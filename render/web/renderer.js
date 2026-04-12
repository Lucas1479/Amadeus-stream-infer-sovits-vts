/**
 * renderer.js — PixiJS 渲染引擎核心
 *
 * 暴露 window.renderApp，Python 端通过 runJavaScript() 调用：
 *   renderApp.setEmotion(emotion)
 *   renderApp.setSpeaking(bool)
 *   renderApp.setMouth(value 0-1)
 *   renderApp.setSubtitle(text)
 *   renderApp.loadSpriteFrames(emotion, [url, ...])
 *   renderApp.loadSpriteClipFrames(emotion, inUrls, loopUrls, outUrls)
 *   renderApp.loadLive2DModel(url)
 *   renderApp.loadTransitionFrames(fromEmotion, toEmotion, [url, ...])
 *   renderApp.setMode('sprite'|'live2d'|'both'|'hybrid')
 *
 * hybrid 模式：静止时显示 Live2D 模型，开始说话时交叉淡入帧动画，
 *              说话结束后淡回 Live2D。适合"Live2D 待机 + 帧动画说话"场景。
 */

(function () {
  "use strict";

  // ---------------------------------------------------------------------------
  // PixiJS Application
  // ---------------------------------------------------------------------------
  const app = new PIXI.Application({
    resizeTo: document.getElementById("canvas-container"),
    backgroundAlpha: 0,          // 透明背景
    autoDensity: true,
    resolution: window.devicePixelRatio || 1,
    antialias: true,
  });
  document.getElementById("canvas-container").appendChild(app.view);

  // ---------------------------------------------------------------------------
  // SpriteRenderer — 静态帧动画（Kur1oR3 风格）
  // ---------------------------------------------------------------------------
  class SpriteRenderer {
    constructor(stage) {
      this.stage = stage;
      this.container = new PIXI.Container();
      stage.addChild(this.container);

      this.sprite = new PIXI.Sprite();
      this.sprite.anchor.set(0.5, 1.0);  // 底部中心对齐
      this.sprite.eventMode = "static";
      this.sprite.cursor = "grab";
      this.container.addChild(this.sprite);
      this._baselineMarker = new PIXI.Graphics();
      this._baselineMarker.visible = false;
      this.container.addChild(this._baselineMarker);

      /** emotion → PIXI.Texture[] */
      this._frames = {};
      /** emotion → { in: PIXI.Texture[], loop: PIXI.Texture[], out: PIXI.Texture[] } */
      this._clips = {};
      /** emotion → { frameIntervalMs?: number, loopMode?: string } */
      this._clipConfigs = {};
      /** emotion → PIXI.Texture[][] (transition from→to) */
      this._transitions = {};

      this._currentEmotion = "normal";
      this._speaking = false;
      this._mouthValue = 0;
      this._frameIdx = 0;
      this._ticker = null;
      this._transitionQueue = [];  // 正在播放的过渡帧序列
      /** { phase: 'in'|'loop'|'out', idx: number } | null */
      this._clipState = null;
      /** { seq: PIXI.Texture[], idx: number, intervalMs: number } | null */
      this._previewState = null;
      /** 默认对齐参数（每个 emotion 独立一份） */
      this._defaultAlign = {
        offsetX: 0,
        offsetY: 0,
        scaleMul: 1.0,
        baselineOffsetPx: 0,
        showBaseline: false,
      };
      this._alignByEmotion = {};
      this._isLockedFn = null;
      this._anchorFn = null;   // () => { x, y, scaleFactor } — 由 RenderApp 注入
      this._baseFrameIntervalMs = 150; // 常规帧节奏
      this._clipFrameIntervalMs = 50;  // 三段式 clip 更细腻

      // ---- 口型同步 ----
      /** emotion → { cx, cy, width, height, curve, closedFrameIdx, openness: number[] } */
      this._mouthConfigs = {};

      // overlay 和 mask 都作为 this.sprite 的子节点。
      // 好处：sprite 被 lock_to_live2d / align 系统移动或缩放时，
      // 子节点自动跟随，完全不需要手动同步坐标。
      // 坐标系：sprite-local，anchor(0.5,1) → 底部中心为原点(0,0)，
      //         纹理占 x∈[-w/2,w/2]，y∈[-h,0]，纹理中心 = (0, -h/2)。

      /** 嘴部遮罩：Graphics，坐标以 sprite-local 纹理像素表示（不含 scale） */
      this._mouthMask = new PIXI.Graphics();
      this.sprite.addChild(this._mouthMask);

      /** 叠加帧 sprite：与 base sprite 相同的 anchor/位置，显示振幅对应的 loop 帧 */
      this._mouthOverlay = new PIXI.Sprite();
      this._mouthOverlay.anchor.set(0.5, 1.0);   // 与 base sprite 对齐
      this._mouthOverlay.x = 0;
      this._mouthOverlay.y = 0;
      this._mouthOverlay.visible = false;
      this._mouthOverlay.mask = this._mouthMask;
      this.sprite.addChild(this._mouthOverlay);

      this._setupAlignInteraction();
      this._startIdleTicker();
    }

    // ---- 资产加载 ----

    loadFrames(emotion, urls) {
      const textures = urls.map(u => {
        const tex = PIXI.Texture.from(u);
        // 贴图加载完成后重新定位（解决初始 height=1 导致的偏移问题）
        if (!tex.baseTexture.valid) {
          tex.baseTexture.on('loaded', () => {
            if (emotion === this._currentEmotion) {
              this._showFrame(this._frameIdx);
            }
          });
        }
        return tex;
      });
      this._frames[emotion] = textures;
      // 若当前情绪就是这个，刷新显示
      if (emotion === this._currentEmotion) this._showFrame(0);
    }

    loadTransitionFrames(fromEmotion, toEmotion, urls) {
      const key = `${fromEmotion}→${toEmotion}`;
      this._transitions[key] = urls.map(u => PIXI.Texture.from(u));
    }

    loadClipFrames(emotion, inUrls, loopUrls, outUrls) {
      const mkTextures = (urls) => (urls || []).map(u => {
        const tex = PIXI.Texture.from(u);
        if (!tex.baseTexture.valid) {
          tex.baseTexture.on('loaded', () => {
            if (emotion === this._currentEmotion) {
              this._showFrame(this._frameIdx);
            }
          });
        }
        return tex;
      });
      this._clips[emotion] = {
        in: mkTextures(inUrls),
        loop: mkTextures(loopUrls),
        out: mkTextures(outUrls),
      };
    }

    setClipConfig(emotion, cfg) {
      if (!emotion || typeof cfg !== "object" || cfg === null) return;
      const oldCfg = this._clipConfigs[emotion] || {};
      const next = { ...oldCfg };
      if (typeof cfg.frameIntervalMs === "number" && Number.isFinite(cfg.frameIntervalMs)) {
        next.frameIntervalMs = Math.max(20, Math.min(200, Math.round(cfg.frameIntervalMs)));
      }
      if (typeof cfg.loopMode === "string") {
        const mode = cfg.loopMode.trim();
        if (mode === "loop_until_speaking_end" || mode === "once_then_hold") {
          next.loopMode = mode;
        }
      }
      this._clipConfigs[emotion] = next;
    }

    // ---- 状态控制 ----

    setEmotion(emotion) {
      if (emotion === this._currentEmotion) return;
      this._applyEmotion(emotion);
    }

    _applyEmotion(emotion) {
      if (emotion === this._currentEmotion) return;
      // 清理 hold 计时器，防止旧计时器在新情绪上误触发
      if (this._outHoldTimer) { clearTimeout(this._outHoldTimer); this._outHoldTimer = null; }
      this._pendingEmotion = null;
      const key = `${this._currentEmotion}→${emotion}`;
      const transFrames = this._transitions[key];
      this._currentEmotion = emotion;
      this._ensureAlignForEmotion(this._currentEmotion);
      this._frameIdx = 0;
      this._clipState = null;
      this._previewState = null;
      // 切换表情时隐藏旧口型遮罩（避免旧 mask 跟新 sprite 错位）
      this._mouthOverlay.visible = false;
      this._mouthMask.clear();

      if (transFrames && transFrames.length > 0) {
        this._playTransition(transFrames, () => {
          this._showFrame(0);
          if (this._speaking) this._startClipSpeaking();
        });
      } else {
        this._showFrame(0);
        if (this._speaking) this._startClipSpeaking();
      }
    }

    setSpeaking(speaking) {
      const prev = this._speaking;
      this._speaking = speaking;
      this._frameIdx = 0;
      if (speaking) {
        // 新说话开始：取消 hold 计时器和预览，立刻接管
        if (this._outHoldTimer) { clearTimeout(this._outHoldTimer); this._outHoldTimer = null; }
        this._previewState = null;
        this._clipState = null;
      }
      if (speaking && !prev) {
        this._startClipSpeaking();
      } else if (!speaking && prev) {
        this._stopClipSpeaking();
      }
    }

    setMouth(value) {
      this._mouthValue = Math.max(0, Math.min(1, value));
      // 调试计数：每 60 次调用打印一次，确认信号持续到达
      this._setMouthCallCount = (this._setMouthCallCount || 0) + 1;
      if (this._setMouthCallCount <= 3 || this._setMouthCallCount % 60 === 0) {
        console.log("[MouthSync] setMouth called #" + this._setMouthCallCount +
          " v=" + value.toFixed(3) + " emotion=" + this._currentEmotion);
      }
      this._updateMouthLayer();
    }

    // ---- 口型同步 ----

    loadMouthConfig(emotion, config) {
      /** 存储检测脚本输出的 { cx, cy, width, height, curve, openness } */
      if (!config || !Array.isArray(config.openness)) {
        console.warn("[MouthSync] loadMouthConfig skipped (invalid config):", emotion, config);
        return;
      }
      this._mouthConfigs[emotion] = config;
      console.log("[MouthSync] loaded config for:", emotion,
        "cx=" + config.cx + " cy=" + config.cy +
        " openness.length=" + config.openness.length +
        " closedIdx=" + config.closedFrameIdx);
    }

    _updateMouthLayer() {
      const cfg = this._mouthConfigs[this._currentEmotion];
      if (!cfg) {
        this._mouthOverlay.visible = false;
        this._mouthMask.clear();
        return;
      }
      const clip = this._getEmotionClip();
      if (!clip || !clip.loop.length) {
        this._mouthOverlay.visible = false;
        this._mouthMask.clear();
        return;
      }

      const v = this._mouthValue;

      // overlay 是 sprite 的子节点，位置(0,0)已自动与 sprite 对齐，无需手动同步。

      // 1. 按振幅查最匹配的 loop 帧（openness 最接近 v 的帧索引）
      const { openness } = cfg;
      const n = Math.min(openness.length, clip.loop.length);
      let bestIdx = 0, bestDist = Infinity;
      for (let i = 0; i < n; i++) {
        const d = Math.abs(openness[i] - v);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      this._mouthOverlay.texture = clip.loop[bestIdx];
      this._mouthOverlay.visible = v > 0.01;

      // 3. 绘制口型遮罩
      //    低振幅时用最小尺寸（0.05），避免遮罩太小看不出效果
      if (v > 0.01) {
        this._drawMouthMask(Math.max(v, 0.05), cfg);
      } else {
        this._mouthMask.clear();
      }
    }

    _drawMouthMask(amplitude, cfg) {
      const g = this._mouthMask;
      g.clear();
      if (amplitude <= 0) return;

      const { cx, cy, width, height } = cfg;
      const tex = this.sprite.texture;
      if (!tex || tex.height <= 1) return;

      // mask 是 sprite 的子节点，坐标系 = sprite-local（纹理像素，不含 scale）。
      // anchor(0.5,1) → 底部中心为原点，纹理中心在 (0, -texH/2)。
      // 检测值 cx/cy 相对纹理中心（正右/正下为正）→ sprite-local:
      //   localX = cx
      //   localY = cy - texH/2
      const th = tex.height;
      const localX = cx;
      const localY = cy - th / 2;

      // 横向：固定宽度 * 1.8 padding，确保嘴角不被截断
      const wHalf = (width / 2) * 1.8;
      // 纵向：基础高度（闭口时也留足空间）+ 振幅扩展
      //   base  = height * 1.0  （完整检测高度，覆盖整个嘴部区域）
      //   extra = height * 1.5 * amplitude  （张嘴时向上/下扩展）
      const hHalf = (height / 2) * (1.0 + 1.5 * amplitude);

      if (hHalf < 0.5) return;

      g.beginFill(0xFFFFFF, 1);
      g.drawEllipse(localX, localY, wHalf, hHalf);
      g.endFill();
    }

    resize(w, h) {
      this.sprite.x = w / 2;
      this.sprite.y = h;
      // 仅在贴图已加载（height > 1）时缩放，避免空贴图 height=1 导致 scale 暴涨
      const th = this.sprite.texture.height;
      if (th > 1) {
        this.sprite.scale.set(h / th);
      } else {
        this.sprite.scale.set(1);
      }
      // 位置变化后重绘口型遮罩
      this._updateMouthLayer();
    }

    // ---- 内部 ----

    _getEmotionFrames() {
      return this._frames[this._currentEmotion] || this._frames["normal"] || null;
    }

    _getEmotionClip() {
      return this._clips[this._currentEmotion] || null;
    }

    _getEmotionClipConfig() {
      const c = this._clipConfigs[this._currentEmotion] || {};
      return {
        frameIntervalMs: (typeof c.frameIntervalMs === "number" ? c.frameIntervalMs : this._clipFrameIntervalMs),
        loopMode: c.loopMode || "loop_until_speaking_end",
      };
    }

    _startClipSpeaking() {
      const clip = this._getEmotionClip();
      if (!clip) return;
      if (clip.in.length > 0) {
        this._clipState = { phase: "in", idx: 0 };
      } else if (clip.loop.length > 0) {
        this._clipState = { phase: "loop", idx: 0 };
      }
    }

    _stopClipSpeaking() {
      // 测试预览期间由预览状态机独占，不接受 speaking stop 打断
      if (this._previewState) return;
      const clip = this._getEmotionClip();
      if (!clip) {
        this._clipState = null;
        this._showFrame(0);
        return;
      }
      if (clip.out.length > 0) {
        this._clipState = { phase: "out", idx: 0 };
      } else {
        this._clipState = null;
        this._showFrame(0);
      }
    }

    _tickClip() {
      const clip = this._getEmotionClip();
      if (!clip || !this._clipState) return false;

      const phase = this._clipState.phase;
      if (phase === "_holding") return true;  // hold 计时中，ticker 保持当前帧不动
      let seq = clip[phase] || [];

      // 如果当前 phase 没帧，按流程自动跳转
      if (seq.length === 0) {
        if (phase === "in") {
          this._clipState = { phase: "loop", idx: 0 };
          return this._tickClip();
        }
        if (phase === "loop") {
          if (!this._speaking) {
            this._clipState = { phase: "out", idx: 0 };
            return this._tickClip();
          }
          return true;
        }
        // phase === "out"
        this._clipState = null;
        this._showFrame(0);
        return true;
      }

      // ── lip sync 模式：loop 阶段冻住底层 sprite 在闭口帧，让 overlay 处理嘴型 ──
      if (phase === "loop" && this._speaking) {
        const mouthCfg = this._mouthConfigs[this._currentEmotion];
        if (mouthCfg) {
          if (!this._lipSyncLoggedOnce) {
            this._lipSyncLoggedOnce = true;
            console.log("[MouthSync] lip-sync mode active for:", this._currentEmotion,
              "closedFrameIdx=" + mouthCfg.closedFrameIdx);
          }
          // 底层 sprite 固定在闭口帧（让 overlay+mask 完全接管嘴部）
          const closedIdx = Math.min(
            typeof mouthCfg.closedFrameIdx === "number" ? mouthCfg.closedFrameIdx : 0,
            seq.length - 1
          );
          this._applyFrame(seq[closedIdx]);
          // idx 不推进：下次 tick 继续持有闭口帧
          return true;
        } else if (!this._noMouthCfgLogged) {
          this._noMouthCfgLogged = true;
          console.warn("[MouthSync] no config for emotion:", this._currentEmotion,
            "configs loaded:", Object.keys(this._mouthConfigs));
        }
      }

      // 逐帧推进（普通时间驱动模式）
      if (this._clipState.idx >= seq.length) {
        if (phase === "in") {
          this._clipState = { phase: this._speaking ? "loop" : "out", idx: 0 };
          return this._tickClip();
        }
        if (phase === "loop") {
          if (this._speaking) {
            const cfg = this._getEmotionClipConfig();
            if (cfg.loopMode === "once_then_hold") {
              this._clipState.idx = Math.max(0, seq.length - 1);
            } else {
              this._clipState.idx = 0;
            }
          } else {
            this._clipState = { phase: "out", idx: 0 };
          }
          return this._tickClip();
        }
        // phase === "out" 完成，停留最后一帧再切回中立
        this._finishOutWithHold(900);
        return true;
      }

      this._applyFrame(seq[this._clipState.idx]);
      this._clipState.idx += 1;
      return true;
    }

    // out 结束后停留最后一帧 holdMs 毫秒再切回中立，让过渡更自然
    _finishOutWithHold(holdMs = 900) {
      if (this._outHoldTimer) clearTimeout(this._outHoldTimer);
      this._clipState = { phase: "_holding" };  // 占位，阻止 ticker 进其他分支
      this._outHoldTimer = setTimeout(() => {
        this._outHoldTimer = null;
        this._clipState = null;
        // hold 结束后若有排队的新情绪，切过去；否则回到当前情绪静止帧
        if (this._pendingEmotion) {
          this._applyEmotion(this._pendingEmotion);
        } else {
          this._showFrame(0);
        }
      }, holdMs);
    }

    playClipPreviewByDuration(durationSec) {
      const clip = this._getEmotionClip();
      if (!clip) return;
      const inSeq = clip.in || [];
      const loopSeq = clip.loop || [];
      const outSeq = clip.out || [];
      if (loopSeq.length === 0) return;

      const cfg = this._getEmotionClipConfig();
      const fi = Math.max(20, Math.round(cfg.frameIntervalMs || this._clipFrameIntervalMs));
      const dur = Math.max(0.2, Number(durationSec) || 0);
      const targetFrames = Math.max(1, Math.round((dur * 1000) / fi));

      const nIn = inSeq.length;
      const nLoop = loopSeq.length;
      const nOut = outSeq.length;
      const nBase = nIn + nLoop + nOut;     // 完整 in+loop+out（通常约 12s）
      const nLoopOut = nLoop + nOut;        // loop+out（通常约 8s）

      const pickMiddleLoop = (count) => {
        if (count <= 0 || nLoop === 0) return [];
        if (count >= nLoop) return [...loopSeq];
        const start = Math.max(0, Math.floor((nLoop - count) / 2));
        return loopSeq.slice(start, start + count);
      };

      const resample = (arr, outCount) => {
        if (!arr || arr.length === 0 || outCount <= 0) return [];
        if (arr.length === outCount) return [...arr];
        if (outCount === 1) return [arr[0]];
        const n = arr.length;
        const out = [];
        for (let i = 0; i < outCount; i++) {
          const idx = Math.round((i * (n - 1)) / (outCount - 1));
          out.push(arr[Math.max(0, Math.min(n - 1, idx))]);
        }
        return out;
      };

      // 策略：in 始终完整播放，loop 按自然速度播整数轮
      //       预览不强制带 out（out 由实时说话流程控制）
      let seq = [];
      const available = targetFrames - nIn;

      if (available <= 0 || nLoop === 0) {
        seq = [...inSeq];
      } else {
        const cycles = Math.max(1, Math.round(available / nLoop));
        const loopPart = [];
        for (let i = 0; i < cycles; i++) loopPart.push(...loopSeq);
        seq = [...inSeq, ...loopPart];
      }

      if (!seq || seq.length === 0) return;

      this._speaking = false;
      this._clipState = null;
      this._previewState = { seq, idx: 0, intervalMs: fi };
      // 返回实际序列时长（ms），供调用方设置精确的回调定时器
      return seq.length * fi + 900;  // +900 为 hold 时间
    }

    _alignKey(emotion) {
      const k = (emotion || this._currentEmotion || "normal");
      return String(k).trim().toLowerCase() || "normal";
    }

    _ensureAlignForEmotion(emotion) {
      const key = this._alignKey(emotion);
      if (!this._alignByEmotion[key]) {
        this._alignByEmotion[key] = { ...this._defaultAlign };
      }
      return this._alignByEmotion[key];
    }

    setAlignTransform(cfg, emotion) {
      if (!cfg || typeof cfg !== "object") return;
      const a = this._ensureAlignForEmotion(emotion);
      if (Number.isFinite(cfg.offsetX)) a.offsetX = Number(cfg.offsetX);
      if (Number.isFinite(cfg.offsetY)) a.offsetY = Number(cfg.offsetY);
      if (Number.isFinite(cfg.scaleMul)) a.scaleMul = Math.max(0.2, Math.min(5.0, Number(cfg.scaleMul)));
      if (Number.isFinite(cfg.baselineOffsetPx)) a.baselineOffsetPx = Number(cfg.baselineOffsetPx);
      if (typeof cfg.showBaseline === "boolean") a.showBaseline = cfg.showBaseline;
      // 锚点感知元数据（由后处理脚本计算，设置后进入 anchor 模式）
      if (Number.isFinite(cfg.textureCx))    a.textureCx    = Number(cfg.textureCx);
      if (Number.isFinite(cfg.textureFeetY)) a.textureFeetY = Number(cfg.textureFeetY);
      if (!emotion || this._alignKey(emotion) === this._alignKey(this._currentEmotion)) {
        this._applyCurrentTransform();
      }
    }

    getAlignTransform(emotion) {
      const a = this._ensureAlignForEmotion(emotion);
      return { ...a };
    }

    nudgeAlignScale(delta) {
      const _an = this._ensureAlignForEmotion(this._currentEmotion);
      const _am = (_an.textureFeetY != null && _an.textureCx != null);
      if (!_am && this._isLockedFn && this._isLockedFn(this._currentEmotion)) return;
      const a = this._ensureAlignForEmotion(this._currentEmotion);
      const next = Math.max(0.2, Math.min(5.0, a.scaleMul + Number(delta || 0)));
      if (Math.abs(next - a.scaleMul) < 1e-6) return;
      a.scaleMul = next;
      this._applyCurrentTransform();
    }

    _setupAlignInteraction() {
      let dragging = false;
      let sx = 0;
      let sy = 0;
      let ox = 0;
      let oy = 0;

      this.sprite.on("pointerdown", (e) => {
        // 锚点感知模式下允许拖拽（调整微调偏移），否则仍检查 lock
        const _a0 = this._ensureAlignForEmotion(this._currentEmotion);
        const _anchorMode = (_a0.textureFeetY != null && _a0.textureCx != null);
        if (!_anchorMode && this._isLockedFn && this._isLockedFn(this._currentEmotion)) return;
        dragging = true;
        this.sprite.cursor = "grabbing";
        sx = e.global.x;
        sy = e.global.y;
        const a = this._ensureAlignForEmotion(this._currentEmotion);
        ox = a.offsetX;
        oy = a.offsetY;
        e.stopPropagation();
      });

      app.stage.on("pointermove", (e) => {
        if (!dragging) return;
        const a = this._ensureAlignForEmotion(this._currentEmotion);
        a.offsetX = ox + (e.global.x - sx);
        a.offsetY = oy + (e.global.y - sy);
        this._applyCurrentTransform();
      });

      const stopDrag = () => {
        dragging = false;
        this.sprite.cursor = "grab";
      };
      app.stage.on("pointerup", stopDrag);
      app.stage.on("pointerupoutside", stopDrag);
    }

    setLockChecker(fn) {
      this._isLockedFn = (typeof fn === "function") ? fn : null;
    }

    setAnchorFn(fn) {
      this._anchorFn = (typeof fn === "function") ? fn : null;
    }

    _drawBaselineMarker() {
      const g = this._baselineMarker;
      if (!g) return;
      const a = this._ensureAlignForEmotion(this._currentEmotion);
      g.clear();
      g.visible = !!a.showBaseline;
      if (!g.visible) return;
      const h = app.screen.height;
      const y = this.sprite.y - a.baselineOffsetPx * this.sprite.scale.y;
      const x1 = Math.max(6, this.sprite.x - h * 0.45);
      const x2 = Math.min(app.screen.width - 6, this.sprite.x + h * 0.45);
      g.lineStyle(2, 0xff4d4f, 0.95);
      g.moveTo(x1, y);
      g.lineTo(x2, y);
      g.beginFill(0xff4d4f, 0.95);
      g.drawCircle(this.sprite.x, y, 3);
      g.endFill();
    }

    _applyCurrentTransform() {
      const texture = this.sprite.texture;
      if (!texture || texture.height <= 0) return;
      const a = this._ensureAlignForEmotion(this._currentEmotion);
      const h = app.screen.height;

      if (a.textureFeetY != null && a.textureCx != null) {
        // ── 锚点感知模式 ─────────────────────────────────────────────────
        // anchor = 角色脚底在屏幕中的世界坐标，来自 Live2D 当前变换
        const anc = this._anchorFn
          ? this._anchorFn()
          : { x: app.screen.width / 2, y: h, scaleFactor: 1.0 };
        const scale = (h / texture.height) * (a.scaleMul || 1.0) * (anc.scaleFactor || 1.0);
        this.sprite.scale.set(scale);
        // 水平：修正贴图中心与角色视觉中心的偏差，offsetX 作为微调
        this.sprite.x = anc.x + (texture.width / 2 - a.textureCx) * scale + (a.offsetX || 0);
        // 垂直：将脚底（textureFeetY）对齐到锚点 y，offsetY 作为微调
        this.sprite.y = anc.y + (texture.height - a.textureFeetY) * scale + (a.offsetY || 0);
      } else {
        // ── 旧有屏幕空间模式（向后兼容） ─────────────────────────────────
        const scale = (h / texture.height) * (a.scaleMul || 1.0);
        this.sprite.scale.set(scale);
        this.sprite.x = app.screen.width / 2 + (a.offsetX || 0);
        this.sprite.y = h + (a.offsetY || 0);
      }
      this._drawBaselineMarker();
    }

    _showFrame(idx) {
      const frames = this._getEmotionFrames();
      if (!frames || frames.length === 0) return;
      this._applyFrame(frames[idx % frames.length]);
    }

    _applyFrame(texture) {
      if (!texture) return;
      this.sprite.texture = texture;
      this._applyCurrentTransform();
    }

    _startIdleTicker() {
      let elapsed = 0;
      app.ticker.add((delta) => {
        if (this._transitionQueue.length > 0) return;  // 过渡中不更新普通帧
        elapsed += app.ticker.deltaMS;
        let intervalMs = this._baseFrameIntervalMs;
        if (this._previewState) {
          intervalMs = this._previewState.intervalMs;
        } else if (this._clipState) {
          const clipCfg = this._getEmotionClipConfig();
          intervalMs = clipCfg.frameIntervalMs;
        }
        if (elapsed < intervalMs) return;
        elapsed = 0;

        if (this._previewState) {
          const p = this._previewState;
          if (p.idx >= p.seq.length) {
            this._previewState = null;
            this._finishOutWithHold(900);
            return;
          }
          this._applyFrame(p.seq[p.idx]);
          p.idx += 1;
          return;
        }

        if (this._clipState && this._tickClip()) return;

        const frames = this._getEmotionFrames();
        if (!frames || frames.length === 0) return;

        if (this._speaking && frames.length > 1) {
          // 说话：循环 frame 1..N-1（frame 0 = 闭嘴）
          this._frameIdx = (this._frameIdx % (frames.length - 1)) + 1;
        } else {
          this._frameIdx = 0;
        }
        this._applyFrame(frames[this._frameIdx]);
      });
    }

    _playTransition(frames, onDone, { startMs = 40, endMs = 100, tailHoldMs = 80 } = {}) {
      this._transitionQueue = [...frames];
      let idx = 0;
      const n = this._transitionQueue.length;
      const step = () => {
        if (idx >= n) {
          this._transitionQueue = [];
          onDone && onDone();
          return;
        }
        this._applyFrame(this._transitionQueue[idx]);
        // 缓出：t=0→1 随帧进度线性插值，最后一帧额外保持 tailHoldMs
        const t = n > 1 ? idx / (n - 1) : 1;
        const delay = startMs + (endMs - startMs) * t + (idx === n - 1 ? tailHoldMs : 0);
        idx++;
        setTimeout(step, delay);
      };
      step();
    }
  }

  // ---------------------------------------------------------------------------
  // Live2DRenderer — pixi-live2d-display 包装（可选）
  // ---------------------------------------------------------------------------
  class Live2DRenderer {
    constructor(stage) {
      this.stage = stage;
      this.container = new PIXI.Container();
      stage.addChild(this.container);
      this._model = null;
      this._available = typeof PIXI.live2d !== "undefined";
      this._loading = false;
      /** 模型加载成功后调用，由 RenderApp 注入 */
      this.onModelLoaded = null;
      /** 用户调整量（叠加在自动适配结果上） */
      this._offsetX = 0;
      this._offsetY = 0;
      this._scaleFactor = 1.0;
      /** 自动适配基础缩放（由 _fitToStage 计算） */
      this._baseScale = 1.0;

      if (!this._available) {
        console.warn("[Live2DRenderer] pixi-live2d-display 未加载，Live2D 不可用");
      }
    }

    isReady() {
      return this._available && this._model !== null;
    }

    // ---- 变换 API（由 Python/GUI 调用）----

    /** 设置位置偏移量与缩放倍率（叠加在自动适配结果之上） */
    setTransform(offsetX, offsetY, scaleFactor) {
      this._offsetX = offsetX;
      this._offsetY = offsetY;
      this._scaleFactor = scaleFactor;
      this._applyTransform();
    }

    getTransform() {
      return { offsetX: this._offsetX, offsetY: this._offsetY, scaleFactor: this._scaleFactor };
    }

    getFollowTransformForSprite() {
      return { offsetX: this._offsetX, offsetY: this._offsetY, scaleMul: this._scaleFactor };
    }

    resetTransform() {
      this._offsetX = 0;
      this._offsetY = 0;
      this._scaleFactor = 1.0;
      this._applyTransform();
    }

    /** 将当前偏移/缩放叠加到模型上（不重算 _baseScale） */
    _applyTransform() {
      if (!this._model) return;
      const w = app.screen.width;
      const h = app.screen.height;
      this._model.x = w / 2 + this._offsetX;
      this._model.y = h + this._offsetY;
      this._model.scale.set(this._baseScale * this._scaleFactor);
    }

    // ---- 拖拽交互 ----

    _setupDrag() {
      if (!this._model) return;
      const model = this._model;
      const self  = this;

      // PixiJS 7 交互模式
      model.eventMode = "static";
      model.cursor    = "grab";

      // stage 也需要 static 才能接收 pointermove（手指/鼠标离开 model 也能跟随）
      app.stage.eventMode = "static";

      let dragging = false, startGX, startGY, startOX, startOY;

      model.on("pointerdown", (e) => {
        dragging = true;
        model.cursor = "grabbing";
        startGX = e.global.x;
        startGY = e.global.y;
        startOX = self._offsetX;
        startOY = self._offsetY;
        e.stopPropagation();
      });

      app.stage.on("pointermove", (e) => {
        if (!dragging) return;
        self._offsetX = startOX + (e.global.x - startGX);
        self._offsetY = startOY + (e.global.y - startGY);
        self._applyTransform();
      });

      const stopDrag = () => { dragging = false; model.cursor = "grab"; };
      app.stage.on("pointerup", stopDrag);
      app.stage.on("pointerupoutside", stopDrag);
    }

    // ---- 加载 ----

    async loadModel(url) {
      if (!this._available) return;
      this._loading = true;
      try {
        const { Live2DModel } = PIXI.live2d;
        const model = await Live2DModel.from(url, { autoInteract: false });
        if (this._model) {
          this.container.removeChild(this._model);
          this._model.destroy();
        }
        this._model = model;
        this.container.addChild(model);
        this._fitToStage();
        this._setupDrag();
        this._loading = false;
        console.log("[Live2DRenderer] 模型加载成功:", url,
          "screen:", app.screen.width, "x", app.screen.height,
          "baseScale:", this._baseScale.toFixed(3));
        if (this.onModelLoaded) this.onModelLoaded();
      } catch (e) {
        this._loading = false;
        console.error("[Live2DRenderer] 模型加载失败:", e);
      }
    }

    setExpression(name) {
      if (!this._model) return;
      try { this._model.expression(name); } catch (e) {}
    }

    setMotion(group, priority) {
      if (!this._model) return;
      try { this._model.motion(group, undefined, priority); } catch (e) {}
    }

    setMouth(value) {
      if (!this._model) return;
      try {
        this._model.internalModel.coreModel.setParameterValueById(
          "PARAM_MOUTH_OPEN_Y", value
        );
      } catch (e) {}
    }

    resize(w, h) {
      if (!this._model) return;
      this._fitToStage();
    }

    _fitToStage() {
      if (!this._model) return;
      const w = app.screen.width;
      const h = app.screen.height;

      // 底部中心对齐（与 SpriteRenderer anchor(0.5,1.0) 一致）
      this._model.anchor.set(0.5, 1.0);

      // 计算基础缩放（按屏幕高度填满），用户缩放倍率叠加在外面
      const mh = (this._model.internalModel && this._model.internalModel.originalHeight)
        || this._model.height || 1;
      this._baseScale = h / mh;

      this._applyTransform();
    }
  }

  // ---------------------------------------------------------------------------
  // Subtitle overlay
  // ---------------------------------------------------------------------------
  class SubtitleOverlay {
    constructor(stage) {
      this.container = new PIXI.Container();
      stage.addChild(this.container);

      this._bg = new PIXI.Graphics();
      this.container.addChild(this._bg);

      this._text = new PIXI.Text("", {
        fontFamily: "Arial, sans-serif",
        fontSize: 16,
        fill: 0xffffff,
        wordWrap: true,
        wordWrapWidth: 400,
        align: "center",
        dropShadow: true,
        dropShadowDistance: 1,
      });
      this._text.anchor.set(0.5, 1);
      this.container.addChild(this._text);
      this.container.visible = false;
    }

    setText(text) {
      this._text.text = text;
      this.container.visible = !!text;
      this._redrawBg();
    }

    resize(w, h) {
      this._text.style.wordWrapWidth = w - 40;
      this._text.x = w / 2;
      this._text.y = h - 10;
      this._redrawBg();
    }

    _redrawBg() {
      const b = this._text.getBounds();
      this._bg.clear();
      if (!this.container.visible) return;
      this._bg.beginFill(0x000000, 0.5);
      this._bg.drawRoundedRect(b.x - 8, b.y - 6, b.width + 16, b.height + 12, 8);
      this._bg.endFill();
    }
  }

  // ---------------------------------------------------------------------------
  // RenderApp — 顶层编排器，暴露给 Python
  // ---------------------------------------------------------------------------
  class RenderApp {
    constructor() {
      this._sprite = new SpriteRenderer(app.stage);
      this._live2d = new Live2DRenderer(app.stage);
      this._subtitle = new SubtitleOverlay(app.stage);
      this._mode = "sprite";  // 'sprite' | 'live2d' | 'both' | 'hybrid'

      // 初始化层级：sprite 在下，live2d 在上，字幕最上
      app.stage.removeChildren();
      app.stage.addChild(this._sprite.container);
      app.stage.addChild(this._live2d.container);
      app.stage.addChild(this._subtitle.container);

      this._live2d.container.visible = false;
      this._sprite.setLockChecker((emotion) => this._isSpriteAlignLocked(emotion));
      // 注入锚点函数：锚点感知模式下 sprite 始终跟随 Live2D 世界位置
      this._sprite.setAnchorFn(() => {
        const t = this._live2d.getFollowTransformForSprite();
        return {
          x: app.screen.width / 2 + (t.offsetX || 0),
          y: app.screen.height + (t.offsetY || 0),
          scaleFactor: Math.max(1e-6, t.scaleMul || 1.0),
        };
      });

      // hybrid 模式内部状态
      this._hybridSpeaking = false;   // 当前是否处于说话状态
      this._crossfadeTick = null;     // 当前进行中的 crossfade ticker（避免重叠）
      this._hybridPreviewTimer = null;
      this._spriteAlignLockByEmotion = {};
      this._spriteAlignLockDeltaByEmotion = {};

      // Live2D 加载完成回调
      this._live2d.onModelLoaded = () => {
        // 启动时 setSpriteAlignLockToLive2D 在模型未就绪时被调用，delta 未被捕获。
        // 模型就绪后在此补全：对所有已 lock 的情绪重新捕获 delta。
        // 此时 sprite 的 offsetX/offsetY 已从 profile 恢复，delta 计算结果正确。
        for (const emo of Object.keys(this._spriteAlignLockByEmotion)) {
          if (this._spriteAlignLockByEmotion[emo]) {
            const align = this._sprite.getAlignTransform(emo);
            // 仅对旧有模式（无 texture 元数据）补捕 delta；锚点模式无需 delta
            if (align.textureFeetY == null) {
              // 若已从持久化数据恢复了 delta，则跳过重新捕获（避免覆盖正确的相对值）
              if (this._spriteAlignLockDeltaByEmotion[emo]) {
                console.log("[RenderApp] onModelLoaded: 跳过重捕（已有持久化 delta）for", emo);
                continue;
              }
              this.setSpriteAlignLockToLive2D(true, emo);
              console.log("[RenderApp] onModelLoaded: 补捕 delta for", emo);
            }
          }
        }
        if (this._mode === "hybrid" && !this._hybridSpeaking) {
          console.log("[RenderApp] Live2D 加载完成，hybrid 模式自动切换到 Live2D 显示");
          this._live2d.container.visible = true;
          this._live2d.container.alpha = 0;
          this._crossfadeTo(this._sprite.container, this._live2d.container, 300);
        }
      };

      // 滚轮缩放：在 live2d / hybrid / both 模式下，滚轮调整模型大小
      app.view.addEventListener("wheel", (e) => {
        // Alt/Shift + 滚轮：直接缩放 sprite 表情层（用于手动对齐）
        if (e.altKey || e.shiftKey) {
          const step = e.deltaY > 0 ? -0.03 : 0.03;
          this._sprite.nudgeAlignScale(step);
          e.preventDefault();
          return;
        }
        if (this._mode !== "sprite") {
          // Live2D 缩放步长调细：默认 0.02，Ctrl 精调 0.01
          const unit = e.ctrlKey ? 0.01 : 0.02;
          const step = e.deltaY > 0 ? -unit : unit;
          const nf = Math.max(0.2, Math.min(5.0, this._live2d._scaleFactor + step));
          this._live2d._scaleFactor = nf;
          this._live2d._applyTransform();
        }
        e.preventDefault();
      }, { passive: false });

      // PixiJS 自己的 resize 事件（resizeTo 生效后触发）比 window.resize 更可靠
      // 注意：回调参数 w/h 是物理像素，改用 app.screen 取逻辑像素
      app.renderer.on('resize', () => {
        const w = app.screen.width;
        const h = app.screen.height;
        this._sprite.resize(w, h);
        this._live2d.resize(w, h);
        this._subtitle.resize(w, h);
        this._sprite._showFrame(this._sprite._frameIdx);
      });

      // ticker 第一帧后强制重算一次（canvas 在此时已有正确尺寸）
      app.ticker.addOnce(() => this._onResize());
      // 持续联动 sprite 到 Live2D：两种模式。
      app.ticker.add(() => {
        const emo = this._currentSpriteEmotionKey();
        const align = this._sprite.getAlignTransform(emo);

        // ── 锚点感知模式：有 texture 元数据 → 每帧重算位置，天然跟随 Live2D ──
        if (align.textureFeetY != null && align.textureCx != null) {
          this._sprite._applyCurrentTransform();
          return;
        }

        // ── 旧 delta 锁定模式（向后兼容，无 texture 元数据的情绪） ──
        if (!this._isSpriteAlignLocked(emo) || !this._live2d.isReady()) return;
        const t = this._live2d.getFollowTransformForSprite();
        const keep = this._sprite.getAlignTransform(emo);
        const d = this._spriteAlignLockDeltaByEmotion[emo] || {
          offsetX: 0,
          offsetY: 0,
          scaleMul: 1.0,
          lockScaleMul: 1.0,
        };
        const lockScale = Math.max(1e-6, Number(d.lockScaleMul) || 1.0);
        const curScale = Math.max(1e-6, Number(t.scaleMul) || 1.0);
        const scaleRatio = curScale / lockScale;
        this._sprite.setAlignTransform({
          // 相对锁定（缩放感知）：偏移随 Live2D 缩放同比例变化，避免漂移
          offsetX: t.offsetX + d.offsetX * scaleRatio,
          offsetY: t.offsetY + d.offsetY * scaleRatio,
          scaleMul: t.scaleMul * d.scaleMul,
          baselineOffsetPx: keep.baselineOffsetPx,
          showBaseline: keep.showBaseline,
        }, emo);
      });
    }

    _currentSpriteEmotionKey() {
      return String((this._sprite && this._sprite._currentEmotion) || "normal").toLowerCase();
    }

    _isSpriteAlignLocked(emotion) {
      const key = String(emotion || this._currentSpriteEmotionKey()).toLowerCase();
      return !!this._spriteAlignLockByEmotion[key];
    }

    // ---- 公共 API ----

    setEmotion(emotion) {
      if (this._mode === "hybrid") {
        // hybrid：两端都更新表情，由当前可见层决定显示效果
        this._sprite.setEmotion(emotion);
        this._live2d.setExpression(emotion);
        return;
      }
      if (this._mode !== "live2d") this._sprite.setEmotion(emotion);
      if (this._mode !== "sprite") this._live2d.setExpression(emotion);
    }

    setSpeaking(speaking) {
      if (this._mode === "hybrid") {
        this._setHybridSpeaking(speaking);
        return;
      }
      if (this._mode !== "live2d") this._sprite.setSpeaking(speaking);
    }

    setMouth(value) {
      if (this._mode !== "live2d") this._sprite.setMouth(value);
      if (this._mode !== "sprite") this._live2d.setMouth(value);
    }

    loadMouthConfig(emotion, config) {
      this._sprite.loadMouthConfig(emotion, config);
    }

    setSubtitle(text) {
      this._subtitle.setText(text);
    }

    loadSpriteFrames(emotion, urls) {
      this._sprite.loadFrames(emotion, urls);
    }

    loadSpriteClipFrames(emotion, inUrls, loopUrls, outUrls) {
      this._sprite.loadClipFrames(emotion, inUrls, loopUrls, outUrls);
    }

    playSpriteClipPreview(durationSec, options) {
      const d = Math.max(0.2, Number(durationSec) || 0);
      const followLive2D = !!(options && options.followLive2D);
      const emo = this._currentSpriteEmotionKey();
      if (this._mode === "hybrid") {
        // 测试预览：在 hybrid 下也要切到 sprite 层，否则用户看不到 in/loop/out。
        if (this._hybridPreviewTimer) {
          clearTimeout(this._hybridPreviewTimer);
          this._hybridPreviewTimer = null;
        }
        this._hybridSpeaking = false; // 这是测试预览，不改变真实 speaking 状态机
        if (this._live2d.isReady()) {
          // followLive2D 开关仅决定是否启用锁定策略；
          // 不再在点测试时强制把对齐吸到 Live2D 绝对值，避免与相对锁定 delta 打架导致回弹。
          this._crossfadeTo(this._live2d.container, this._sprite.container, 180);
        } else {
          this._sprite.container.visible = true;
          this._sprite.container.alpha = 1;
        }
        const actualMs = this._sprite.playClipPreviewByDuration(d) || Math.round(d * 1000) + 1200;
        // 预览结束后回到 Live2D 待机层，用实际序列时长（含 hold）而非请求时长
        this._hybridPreviewTimer = setTimeout(() => {
          this._hybridPreviewTimer = null;
          if (this._mode !== "hybrid") return;
          if (this._live2d.isReady()) {
            this._crossfadeTo(this._sprite.container, this._live2d.container, 180);
          }
        }, actualMs + 220);
        return;
      }
      this._sprite.playClipPreviewByDuration(d);
    }

    setSpriteClipConfig(emotion, cfg) {
      this._sprite.setClipConfig(emotion, cfg);
    }

    setSpriteAlignTransform(cfg, emotion) {
      this._sprite.setAlignTransform(cfg || {}, emotion || null);
    }

    getSpriteAlignTransform(emotion) {
      const result = this._sprite.getAlignTransform(emotion || null);
      // 附带 delta 数据，供 Python 层持久化，重启后无需重新捕获
      const emo = String(emotion || this._currentSpriteEmotionKey()).toLowerCase();
      const d = this._spriteAlignLockDeltaByEmotion[emo];
      if (d) {
        result._lockDx = Number(d.offsetX) || 0;
        result._lockDy = Number(d.offsetY) || 0;
        result._lockDs = Number(d.scaleMul) || 1.0;
        result._lockLockS = Number(d.lockScaleMul) || 1.0;
      }
      return result;
    }

    // 直接注入已保存的 delta，绕过 isReady() 检查。重启恢复时使用。
    setSpriteAlignLockDelta(dx, dy, ds, lockS, emotion) {
      const emo = String(emotion || this._currentSpriteEmotionKey()).toLowerCase();
      this._spriteAlignLockByEmotion[emo] = true;
      this._spriteAlignLockDeltaByEmotion[emo] = {
        offsetX: Number(dx) || 0,
        offsetY: Number(dy) || 0,
        scaleMul: Number(ds) || 1.0,
        lockScaleMul: Number(lockS) || 1.0,
      };
    }

    setSpriteAlignLockToLive2D(locked, emotion) {
      const emo = String(emotion || this._currentSpriteEmotionKey()).toLowerCase();
      const nextLocked = !!locked;
      const prevLocked = !!this._spriteAlignLockByEmotion[emo];
      this._spriteAlignLockByEmotion[emo] = nextLocked;
      // 已经处于锁定状态时，避免重复重算 delta（切换预设会触发），否则会逐步漂移。
      if (nextLocked && prevLocked) {
        return;
      }
      if (nextLocked && this._live2d.isReady()) {
        const t = this._live2d.getFollowTransformForSprite();
        const cur = this._sprite.getAlignTransform(emo);
        const ts = Math.max(1e-6, Number(t.scaleMul) || 1.0);
        const cs = Math.max(1e-6, Number(cur.scaleMul) || 1.0);
        // 记录相对差值：点锁时不改变当前 sprite 位置/大小。
        this._spriteAlignLockDeltaByEmotion[emo] = {
          offsetX: (Number(cur.offsetX) || 0) - (Number(t.offsetX) || 0),
          offsetY: (Number(cur.offsetY) || 0) - (Number(t.offsetY) || 0),
          scaleMul: cs / ts,
          lockScaleMul: ts,
        };
      } else if (!nextLocked) {
        // 解锁只清该 emotion 的 delta，其他表情不受影响。
        delete this._spriteAlignLockDeltaByEmotion[emo];
      }
    }

    loadTransitionFrames(fromEmotion, toEmotion, urls) {
      this._sprite.loadTransitionFrames(fromEmotion, toEmotion, urls);
    }

    async loadLive2DModel(url) {
      await this._live2d.loadModel(url);
    }

    setMode(mode) {
      this._mode = mode;
      if (mode === "hybrid") {
        this._hybridSpeaking = false;
        if (this._live2d.isReady()) {
          // Live2D 已就绪：直接显示 Live2D，隐藏 sprite
          this._live2d.container.visible = true;
          this._live2d.container.alpha = 1;
          this._sprite.container.visible = false;
          this._sprite.container.alpha = 0;
          console.log("[RenderApp] hybrid 模式启动，Live2D 就绪，显示 Live2D");
        } else {
          // Live2D 未就绪（模型还在加载）：先显示 sprite，等 onModelLoaded 回调再切换
          this._sprite.container.visible = true;
          this._sprite.container.alpha = 1;
          this._live2d.container.visible = false;
          this._live2d.container.alpha = 0;
          console.warn("[RenderApp] hybrid 模式启动，Live2D 尚未就绪，暂时显示 sprite，等待模型加载");
        }
      } else {
        this._sprite.container.visible = (mode !== "live2d");
        this._sprite.container.alpha = 1;
        this._live2d.container.visible = (mode !== "sprite");
        this._live2d.container.alpha = 1;
      }
    }

    // ---- hybrid 内部实现 ----

    /**
     * hybrid 模式下切换说话状态：
     *   speaking=true  → Live2D 淡出，sprite 帧动画淡入（开始说话）
     *   speaking=false → sprite 淡出，Live2D 淡入（说话结束，回到待机）
     * 过渡时长 200ms，使用 PixiJS ticker 逐帧插值，平滑无卡顿。
     */
    _setHybridSpeaking(speaking) {
      if (this._hybridSpeaking === speaking) return;
      this._hybridSpeaking = speaking;

      if (speaking) {
        // 开始说话：切到帧动画（无论 Live2D 是否就绪）
        this._sprite.setSpeaking(true);
        if (this._live2d.isReady()) {
          this._crossfadeTo(this._live2d.container, this._sprite.container, 200);
        } else {
          // Live2D 还没就绪，sprite 本来就在显示，只需要启动说话动画
          this._sprite.container.visible = true;
          this._sprite.container.alpha = 1;
        }
      } else {
        // 停止说话：Live2D 就绪则切回 Live2D，否则保持 sprite
        this._sprite.setSpeaking(false);
        if (this._live2d.isReady()) {
          this._crossfadeTo(this._sprite.container, this._live2d.container, 200);
        } else {
          this._sprite.container.visible = true;
          this._sprite.container.alpha = 1;
        }
      }
    }

    /**
     * 从 fromContainer（alpha 1→0）淡出，toContainer（alpha 0→1）淡入。
     * 若上一次 crossfade 还未结束，先中止再开始新的，避免 alpha 错乱。
     */
    _crossfadeTo(fromContainer, toContainer, duration) {
      // 中止上一次未完成的过渡
      if (this._crossfadeTick) {
        app.ticker.remove(this._crossfadeTick);
        this._crossfadeTick = null;
      }

      // 确保目标 container 可见（alpha 从当前值开始平滑，不跳帧）
      toContainer.visible = true;

      let elapsed = 0;
      const tick = (delta) => {
        elapsed += app.ticker.deltaMS;
        const t = Math.min(elapsed / duration, 1);
        // ease-out cubic，过渡更自然
        const eased = 1 - Math.pow(1 - t, 3);
        fromContainer.alpha = 1 - eased;
        toContainer.alpha = eased;

        if (t >= 1) {
          app.ticker.remove(tick);
          this._crossfadeTick = null;
          fromContainer.visible = false;
          fromContainer.alpha = 1;  // 重置 alpha，下次切换时从正确值开始
          toContainer.alpha = 1;
        }
      };
      this._crossfadeTick = tick;
      app.ticker.add(tick);
    }

    // ---- Live2D 变换公开 API（Python 端通过 runJavaScript 调用）----

    /** 设置位置偏移（单位：逻辑像素）与缩放倍率（1.0 = 默认大小） */
    setLive2DTransform(offsetX, offsetY, scaleFactor) {
      this._live2d.setTransform(offsetX, offsetY, scaleFactor);
    }

    /** 返回当前变换参数 {offsetX, offsetY, scaleFactor} */
    getLive2DTransform() {
      return this._live2d.getTransform();
    }

    /** 重置到默认位置/大小 */
    resetLive2DTransform() {
      this._live2d.resetTransform();
    }

    // ---- 内部 ----

    _onResize() {
      const w = app.screen.width;
      const h = app.screen.height;
      this._sprite.resize(w, h);
      this._live2d.resize(w, h);
      this._subtitle.resize(w, h);
      this._sprite._showFrame(this._sprite._frameIdx);
    }
  }

  // 挂载到全局，Python runJavaScript 调用
  window.renderApp = new RenderApp();
  console.log("[RenderEngine] renderer.js 初始化完成");

})();

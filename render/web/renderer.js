/**
 * renderer.js — PixiJS 渲染引擎核心
 *
 * 暴露 window.renderApp，Python 端通过 runJavaScript() 调用：
 *   renderApp.setEmotion(emotion)
 *   renderApp.setSpeaking(bool)
 *   renderApp.setMouth(value 0-1)
 *   renderApp.setSubtitle(text)
 *   renderApp.loadSpriteFrames(emotion, [url, ...])
 *   renderApp.loadLive2DModel(url)
 *   renderApp.loadTransitionFrames(fromEmotion, toEmotion, [url, ...])
 *   renderApp.setMode('sprite'|'live2d'|'both')
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
      this.container.addChild(this.sprite);

      /** emotion → PIXI.Texture[] */
      this._frames = {};
      /** emotion → PIXI.Texture[][] (transition from→to) */
      this._transitions = {};

      this._currentEmotion = "normal";
      this._speaking = false;
      this._mouthValue = 0;
      this._frameIdx = 0;
      this._ticker = null;
      this._transitionQueue = [];  // 正在播放的过渡帧序列

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

    // ---- 状态控制 ----

    setEmotion(emotion) {
      if (emotion === this._currentEmotion) return;
      const key = `${this._currentEmotion}→${emotion}`;
      const transFrames = this._transitions[key];
      const prev = this._currentEmotion;
      this._currentEmotion = emotion;
      this._frameIdx = 0;

      if (transFrames && transFrames.length > 0) {
        this._playTransition(transFrames, () => this._showFrame(0));
      } else {
        this._showFrame(0);
      }
    }

    setSpeaking(speaking) {
      this._speaking = speaking;
      this._frameIdx = 0;
    }

    setMouth(value) {
      this._mouthValue = Math.max(0, Math.min(1, value));
      // 直接映射到帧索引（0 = 闭嘴，>0.5 = 开口帧）
      if (this._speaking) {
        const frames = this._getEmotionFrames();
        if (frames && frames.length > 1) {
          const idx = this._mouthValue > 0.5 ? 1 : 0;
          this._applyFrame(frames[Math.min(idx, frames.length - 1)]);
        }
      }
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
    }

    // ---- 内部 ----

    _getEmotionFrames() {
      return this._frames[this._currentEmotion] || this._frames["normal"] || null;
    }

    _showFrame(idx) {
      const frames = this._getEmotionFrames();
      if (!frames || frames.length === 0) return;
      this._applyFrame(frames[idx % frames.length]);
    }

    _applyFrame(texture) {
      if (!texture) return;
      this.sprite.texture = texture;
      // 使用 app.screen（逻辑 CSS 像素），而非 app.renderer（物理像素）
      // 高 DPI 屏幕下 renderer.width = screen.width × devicePixelRatio，直接用会导致坐标偏移
      const h = app.screen.height;
      if (texture.height > 0) {
        const scale = h / texture.height;
        this.sprite.scale.set(scale);
        this.sprite.x = app.screen.width / 2;
        this.sprite.y = h;
      }
    }

    _startIdleTicker() {
      // 150ms 一帧（与 KurisuCharacter.timer 保持一致）
      let elapsed = 0;
      app.ticker.add((delta) => {
        if (this._transitionQueue.length > 0) return;  // 过渡中不更新普通帧
        elapsed += app.ticker.deltaMS;
        if (elapsed < 150) return;
        elapsed = 0;

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

    _playTransition(frames, onDone) {
      this._transitionQueue = [...frames];
      let idx = 0;
      const step = () => {
        if (idx >= this._transitionQueue.length) {
          this._transitionQueue = [];
          onDone && onDone();
          return;
        }
        this._applyFrame(this._transitionQueue[idx++]);
        setTimeout(step, 50);  // 过渡帧间隔 50ms
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
      if (!this._available) {
        console.warn("[Live2DRenderer] pixi-live2d-display 未加载，Live2D 不可用");
      }
    }

    async loadModel(url) {
      if (!this._available) return;
      try {
        const { Live2DModel } = PIXI.live2d;
        const model = await Live2DModel.from(url);
        if (this._model) {
          this.container.removeChild(this._model);
          this._model.destroy();
        }
        this._model = model;
        this.container.addChild(model);
        this._fitToStage();
        console.log("[Live2DRenderer] 模型加载成功:", url);
      } catch (e) {
        console.error("[Live2DRenderer] 模型加载失败:", e);
      }
    }

    setExpression(name) {
      if (!this._model) return;
      try {
        this._model.expression(name);
      } catch (e) {}
    }

    setMotion(group, priority) {
      if (!this._model) return;
      try {
        this._model.motion(group, undefined, priority);
      } catch (e) {}
    }

    setMouth(value) {
      if (!this._model) return;
      try {
        this._model.internalModel.coreModel.setParameterValueById(
          "ParamMouthOpenY", value
        );
      } catch (e) {}
    }

    resize(w, h) {
      if (!this._model) return;
      this._fitToStage();
    }

    _fitToStage() {
      if (!this._model) return;
      const w = app.renderer.width;
      const h = app.renderer.height;
      this._model.x = w / 2;
      this._model.y = h / 2;
      const scaleX = w / this._model.width;
      const scaleY = h / this._model.height;
      this._model.scale.set(Math.min(scaleX, scaleY));
      this._model.anchor.set(0.5);
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
      this._mode = "sprite";  // 'sprite' | 'live2d' | 'both'

      // 初始化层级：sprite 在下，live2d 在上，字幕最上
      app.stage.removeChildren();
      app.stage.addChild(this._sprite.container);
      app.stage.addChild(this._live2d.container);
      app.stage.addChild(this._subtitle.container);

      this._live2d.container.visible = false;

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
    }

    // ---- 公共 API ----

    setEmotion(emotion) {
      if (this._mode !== "live2d") this._sprite.setEmotion(emotion);
      if (this._mode !== "sprite") this._live2d.setExpression(emotion);
    }

    setSpeaking(speaking) {
      if (this._mode !== "live2d") this._sprite.setSpeaking(speaking);
    }

    setMouth(value) {
      if (this._mode !== "live2d") this._sprite.setMouth(value);
      if (this._mode !== "sprite") this._live2d.setMouth(value);
    }

    setSubtitle(text) {
      this._subtitle.setText(text);
    }

    loadSpriteFrames(emotion, urls) {
      this._sprite.loadFrames(emotion, urls);
    }

    loadTransitionFrames(fromEmotion, toEmotion, urls) {
      this._sprite.loadTransitionFrames(fromEmotion, toEmotion, urls);
    }

    async loadLive2DModel(url) {
      await this._live2d.loadModel(url);
    }

    setMode(mode) {
      this._mode = mode;
      this._sprite.container.visible = (mode !== "live2d");
      this._live2d.container.visible = (mode !== "sprite");
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

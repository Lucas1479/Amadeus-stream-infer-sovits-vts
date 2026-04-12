"""vts/action.py — VTS 动作执行层

负责：
  - VTS 心跳（heartbeat_worker）
  - 表情/动作队列消费（action_worker）
  - 批量表情复位（reset_all_expressions）
  - 动作指令分发（record_actions）

依赖注入（configure()）：
  - vts_manager     : VTSConnectionManager 实例
  - pending_actions : Queue（由 main.py 传入）
  - delegate_fn     : async callable，处理 DELEGATE 标签（_handle_delegate）
"""

import asyncio
import logging
import time
import traceback
from queue import Empty
from threading import Thread, Lock

from tools.tts_text_processor import EMO_PRESETS
from tools.text_utils import _parse_seconds, _pair_ids_values

logger = logging.getLogger(__name__)

# ===== 模块级状态 =====
_active_expressions: set = set()
_active_expr_lock = Lock()

# 心跳/动作队列暂停标志（切换到 PixiJS 渲染模式时置 True，回到 VTS 时置 False）
_paused: bool = False

# ===== 依赖注入占位 =====
_vts_manager = None
_pending_actions = None
_delegate_fn = None


def configure(vts_manager=None, pending_actions=None, delegate_fn=None):
    """注入运行时依赖。在 async main() 初始化完毕后调用。"""
    global _vts_manager, _pending_actions, _delegate_fn
    if vts_manager is not None:
        _vts_manager = vts_manager
    if pending_actions is not None:
        _pending_actions = pending_actions
    if delegate_fn is not None:
        _delegate_fn = delegate_fn


def set_paused(paused: bool) -> None:
    """暂停/恢复 VTS 心跳与动作队列。

    切换到 PixiJS 内置渲染时调用 set_paused(True)，
    切回 VTS 渲染时调用 set_paused(False)。
    """
    global _paused
    _paused = paused
    logger.info(f"[VTSAction] {'⏸ VTS 心跳已暂停' if paused else '▶ VTS 心跳已恢复'}")


# =============================================================================
# 心跳
# =============================================================================

def heartbeat_worker():
    """VTS 心跳工作线程。断联时主动触发后台重连，不阻塞。

    _paused=True 时跳过所有操作（PixiJS 渲染模式下不需要维持 VTS 连接活跃度）。
    """
    while True:
        try:
            time.sleep(3)
            if _paused:
                continue
            if _vts_manager is None:
                continue
            if not _vts_manager.connected:
                _vts_manager._start_background_reconnect()
            else:
                # 先发协议级 PING 帧保持 TCP 连接，再发应用层心跳
                _vts_manager.send_ping()
                _vts_manager.send_heartbeat()
        except Exception as e:
            logger.error(f"心跳线程异常: {e}")


# =============================================================================
# 动作队列消费
# =============================================================================

def action_worker():
    """VTS 动作队列消费线程。

    _paused=True 时丢弃队列中的 VTS 动作（PixiJS 渲染模式下 VTS 不应收到指令）。
    """
    while True:
        try:
            if _pending_actions is None:
                time.sleep(0.1)
                continue
            if _paused:
                # 清空队列积压，避免切回 VTS 时一次性爆发大量旧指令
                try:
                    while not _pending_actions.empty():
                        _pending_actions.get_nowait()
                except Exception:
                    pass
                time.sleep(0.1)
                continue
            act = _pending_actions.get_nowait()
            atype = (act.get("type") or "").upper()
            attrs = act.get("attrs", {})

            if atype == "HOTKEY":
                key = attrs.get("name") or attrs.get("id")
                if key and _vts_manager is not None:
                    _vts_manager.trigger_hotkey(key)
                continue

            if atype == "EXPR":
                name = attrs.get("name") or attrs.get("file")
                if not name:
                    continue
                alias = {
                    # 微笑/Smile
                    "微笑": "Smile.exp3.json",
                    "微笑み": "Smile.exp3.json",
                    "smile": "Smile.exp3.json",
                    "happy": "Smile.exp3.json",
                    "开心": "Smile.exp3.json",
                    "高兴": "Smile.exp3.json",
                    "喜び": "Smile.exp3.json",
                    # 思考/Thinking
                    "思考": "Thinking.exp3.json",
                    "考え": "Thinking.exp3.json",
                    "thinking": "Thinking.exp3.json",
                    "思考中": "Thinking.exp3.json",
                    "考え中": "Thinking.exp3.json",
                    # 生气/Angry
                    "angry": "Angry.exp3.json",
                    "生气": "Angry.exp3.json",
                    "annoyed": "Angry.exp3.json",
                    "愤怒": "Angry.exp3.json",
                    "怒り": "Angry.exp3.json",
                    "mad": "Angry.exp3.json",
                    "furious": "Angry.exp3.json",
                    # 失望/Disappointed
                    "disappointed": "Disappointed.exp3.json",
                    "失望": "Disappointed.exp3.json",
                    "沮丧": "Disappointed.exp3.json",
                    "失落": "Disappointed.exp3.json",
                    "がっかり": "Disappointed.exp3.json",
                    "落ち込む": "Disappointed.exp3.json",
                    "sad": "Disappointed.exp3.json",
                    "sorrow": "Disappointed.exp3.json",
                    "dejected": "Disappointed.exp3.json",
                }
                key_str = str(name).strip()
                mapped = alias.get(key_str.lower()) or alias.get(key_str)
                if mapped:
                    name = mapped
                fade = _parse_seconds(attrs.get("fade"), 0.0)
                dur = _parse_seconds(attrs.get("dur"), 0.0)
                active_str = attrs.get("active")
                active = True if active_str is None else (str(active_str).lower() == "true")
                weight = attrs.get("weight")

                # 表情持续时间由 emotion_presets.json 注册表管理（ExpressionController），
                # action_worker 仅处理直接的 EXPR 指令，不覆盖 dur。

                if _vts_manager is not None:
                    _vts_manager.activate_expression(
                        name,
                        active=True if active else False,
                        fade_time=fade,
                        weight=weight,
                    )
                try:
                    with _active_expr_lock:
                        if active:
                            _active_expressions.add(name)
                        else:
                            _active_expressions.discard(name)
                except Exception:
                    pass
                if active and dur > 0:
                    def _deactivate(name_, fade_):
                        try:
                            logger.info(f"计划到时关闭表情: name={name_}, fade={fade_}")
                            if _vts_manager is not None:
                                _vts_manager.activate_expression(
                                    name_, active=False, fade_time=fade_, weight=0.0
                                )
                            with _active_expr_lock:
                                _active_expressions.discard(name_)
                        except Exception as e:
                            logger.error(f"关闭表情失败: {e}")
                    Thread(
                        target=lambda d=dur, n=name, f=fade: (time.sleep(d), _deactivate(n, f)),
                        daemon=True,
                    ).start()
                continue

            if atype == "PARAM":
                ids_text = attrs.get("id") or attrs.get("ids")
                values_text = attrs.get("value") or attrs.get("values")
                pairs = _pair_ids_values(ids_text, values_text)
                if not pairs:
                    continue
                dur = _parse_seconds(attrs.get("dur"), 0.0)
                fade = _parse_seconds(attrs.get("fade"), 0.0)
                total_fade = fade if fade > 0 else 0.0
                if total_fade <= 0:
                    if _vts_manager is not None:
                        _vts_manager.send_parameters(pairs)
                else:
                    steps = max(1, int(total_fade / 0.05))
                    for step in range(1, steps + 1):
                        ratio = step / float(steps)
                        interp = {pid: val * ratio for pid, val in pairs.items()}
                        if _vts_manager is not None:
                            _vts_manager.send_parameters(interp)
                        time.sleep(0.05)
                if dur > 0:
                    def _reset(pairs_, fade_):
                        try:
                            if fade_ > 0:
                                steps2 = max(1, int(fade_ / 0.05))
                                for step in range(steps2, 0, -1):
                                    ratio = step / float(steps2)
                                    interp = {pid: val * ratio for pid, val in pairs_.items()}
                                    if _vts_manager is not None:
                                        _vts_manager.send_parameters(interp)
                                    time.sleep(0.05)
                            if _vts_manager is not None:
                                _vts_manager.send_parameters({pid: 0.0 for pid in pairs_.keys()})
                        except Exception:
                            pass
                    Thread(
                        target=lambda d=dur, p=pairs, tf=total_fade: (time.sleep(d), _reset(p, tf)),
                        daemon=True,
                    ).start()
                continue

            if atype == "EMO":
                preset = attrs.get("preset")
                if not preset:
                    continue
                cfg = EMO_PRESETS.get(preset)
                dur = _parse_seconds(attrs.get("dur"), 0.0)
                if not cfg:
                    continue
                params_cfg = cfg.get("PARAM") or {}
                expr_cfg = cfg.get("EXPR") or {}
                fade = _parse_seconds(attrs.get("fade"), 0.0)
                if params_cfg and _vts_manager is not None:
                    _vts_manager.send_parameters(params_cfg)
                    if dur > 0:
                        Thread(
                            target=lambda d=dur, pc=params_cfg: (
                                time.sleep(d),
                                _vts_manager.send_parameters({pid: 0.0 for pid in pc.keys()})
                                if _vts_manager else None,
                            ),
                            daemon=True,
                        ).start()
                for name, _eopts in expr_cfg.items():
                    if _vts_manager is not None:
                        _vts_manager.activate_expression(name, active=True, fade_time=fade)
                    with _active_expr_lock:
                        _active_expressions.add(name)
                    if dur > 0:
                        Thread(
                            target=lambda d=dur, n=name, f=fade: (
                                time.sleep(d),
                                _vts_manager.activate_expression(n, active=False, fade_time=f)
                                if _vts_manager else None,
                            ),
                            daemon=True,
                        ).start()
                continue

        except Empty:
            time.sleep(0.02)
            continue
        except Exception as e:
            logger.error(f"动作执行线程异常: {e}")
            logger.error(traceback.format_exc())


# =============================================================================
# 表情复位
# =============================================================================

def reset_all_expressions(fade_time: float = 0.2):
    """关闭当前激活的所有表情。"""
    try:
        with _active_expr_lock:
            targets = list(_active_expressions)
            _active_expressions.clear()
        if targets:
            logger.info(f"复位表情,共 {len(targets)} 个")
        for name in targets:
            try:
                if _vts_manager is not None:
                    _vts_manager.activate_expression(
                        name, active=False, fade_time=fade_time, weight=0.0
                    )
            except Exception as e:
                logger.error(f"复位表情失败: name={name}, err={e}")
    except Exception as e:
        logger.error(f"复位表情总控异常: {e}")


# =============================================================================
# 动作指令分发
# =============================================================================

def record_actions(actions):
    """将解析得到的动作记录到全局队列，等待后续VTS执行模块消费。
    DELEGATE 类型例外：异步触发 OpenClaw 委托，不进入 VTS 队列。
    """
    if not actions:
        return
    for act in actions:
        if act.get("type") == "DELEGATE":
            task = act.get("attrs", {}).get("task", "").strip()
            if task and _delegate_fn is not None:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(_delegate_fn(task))
                        logger.info(f"[OpenClaw] 委托任务已提交: {task[:60]}")
                    else:
                        logger.warning("[OpenClaw] 事件循环未运行，无法提交委托任务")
                except Exception as e:
                    logger.error(f"[OpenClaw] 委托任务提交失败: {e}")
        else:
            if _pending_actions is not None:
                _pending_actions.put(act)
                logger.info(f"解析到表情指令: {act}")

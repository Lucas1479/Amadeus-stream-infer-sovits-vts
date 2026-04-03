"""Runtime compatibility helpers shared by the app entrypoints.

This module keeps platform-specific policy in one place so Mac compatibility
changes do not get reimplemented ad-hoc across `main.py`, GUI code, and the
TTS pipeline.
"""

from __future__ import annotations

import logging
import os
import platform

import torch

logger = logging.getLogger(__name__)

_VALID_TTS_DEVICES = {"auto", "cpu", "cuda", "mps"}


def is_macos_arm64() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def has_mps_support() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


def normalize_tts_device(requested: str | None) -> str:
    value = (requested or "auto").strip().lower()
    if value not in _VALID_TTS_DEVICES:
        logger.warning("未知的 TTS_DEVICE=%s，回退到 auto", requested)
        return "auto"
    return value


def resolve_tts_device(requested: str | None, *, log: logging.Logger | None = None) -> str:
    """Resolve the effective runtime device for TTS.

    Apple Silicon defaults to CPU in `auto` mode on purpose. Upstream GPT-SoVITS
    notes that CPU inference is often the more stable default on macOS, while
    MPS remains available as an explicit opt-in for local experiments.
    """

    log = log or logger
    requested = normalize_tts_device(requested)

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if is_macos_arm64():
            log.info("检测到 Apple Silicon；auto 模式默认回退到 CPU 以优先保证稳定性。")
            return "cpu"
        if has_mps_support():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("TTS_DEVICE=cuda 但当前环境无 CUDA，回退到 CPU。")
        return "cpu"

    if requested == "mps" and not has_mps_support():
        log.warning("TTS_DEVICE=mps 但当前环境无 MPS，回退到 CPU。")
        return "cpu"

    return requested


def supports_cuda_graph(device: str | None = None) -> bool:
    device = normalize_tts_device(
        device or os.environ.get("AMADEUS_RESOLVED_TTS_DEVICE") or os.environ.get("TTS_DEVICE")
    )
    return device == "cuda" and torch.cuda.is_available()


def effective_cuda_graph_enabled(device: str | None = None) -> bool:
    return supports_cuda_graph(device) and os.environ.get("ENABLE_CUDA_GRAPH", "0") == "1"


def should_use_half_precision(device: str | None) -> bool:
    return normalize_tts_device(device) == "cuda" and torch.cuda.is_available()


def get_tts_mode_label(cuda_graph_enabled: bool) -> str:
    return "CUDA Graph ×1" if cuda_graph_enabled else "Parallel ×2"


def configure_torch_runtime(requested: str | None, *, log: logging.Logger | None = None) -> str:
    """Apply platform-specific torch runtime guards and return the resolved device."""

    log = log or logger
    resolved = resolve_tts_device(requested, log=log)
    os.environ["AMADEUS_RESOLVED_TTS_DEVICE"] = resolved

    # Keep MPS opt-in usable on macOS even when individual ops lack native
    # kernels. This makes explicit `TTS_DEVICE=mps` experiments degrade to CPU
    # instead of hard-failing partway through inference.
    if resolved == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        log.info("已启用 PYTORCH_ENABLE_MPS_FALLBACK=1，降低 MPS 路径的算子兼容风险。")

    # CUDA Graph only exists on CUDA. Clamp the environment here so the rest of
    # the code can trust `ENABLE_CUDA_GRAPH` instead of repeating platform
    # checks everywhere.
    if resolved != "cuda" and os.environ.get("ENABLE_CUDA_GRAPH") == "1":
        log.warning("当前 TTS 设备为 %s，禁用 ENABLE_CUDA_GRAPH 以避免进入无效模式。", resolved)
        os.environ["ENABLE_CUDA_GRAPH"] = "0"

    return resolved

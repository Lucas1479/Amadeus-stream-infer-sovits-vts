"""运行时资源检查。

把 TTS 依赖的模型 / 前端目录 / 参考音频集中检查，避免主入口和 GUI
各自维护一套散落的存在性判断，后续做降级模式时也只需要消费一份状态。
"""

from dataclasses import dataclass
from pathlib import Path


_REQUIRED_PRETRAINED_DIRS = (
    ("BERT model directory", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"),
    ("CN-HuBERT directory", "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
    ("BigVGAN directory", "GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x"),
)


@dataclass(frozen=True)
class TTSRuntimeAssetsStatus:
    available: bool
    summary: str
    tooltip: str
    missing: list[str]
    checked_paths: dict[str, str]


def _resolve_path(root_dir: Path, raw_path: str) -> Path | None:
    if not raw_path or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else root_dir / path


def _describe_missing(label: str, path: Path | None) -> str:
    if path is None:
        return f"{label}: not configured"
    return f"{label}: {path}"


def _build_summary(missing_labels: list[str]) -> str:
    if not missing_labels:
        return "Ready: TTS runtime assets are available."

    preview = ", ".join(missing_labels[:3])
    remaining = len(missing_labels) - 3
    if remaining > 0:
        preview = f"{preview} (+{remaining} more)"
    return f"Warning: TTS unavailable — missing {preview}."


def inspect_tts_runtime_assets(
    root_dir: str | Path,
    gpt_model_path: str,
    sovits_model_path: str,
    ref_audio_path: str,
) -> TTSRuntimeAssetsStatus:
    """检查 TTS 运行所需资源是否齐全。"""
    root = Path(root_dir).resolve()
    checks: list[tuple[str, Path | None, str]] = [
        ("GPT model (.ckpt)", _resolve_path(root, gpt_model_path), "file"),
        ("SoVITS model (.pth)", _resolve_path(root, sovits_model_path), "file"),
        ("Reference audio (.wav)", _resolve_path(root, ref_audio_path), "file"),
    ]
    checks.extend((label, root / rel_path, "dir") for label, rel_path in _REQUIRED_PRETRAINED_DIRS)

    missing: list[str] = []
    checked_paths: dict[str, str] = {}
    missing_labels: list[str] = []

    for label, path, expected_kind in checks:
        checked_paths[label] = str(path) if path is not None else "<unset>"
        if path is None:
            missing.append(_describe_missing(label, path))
            missing_labels.append(label)
            continue

        exists = path.is_file() if expected_kind == "file" else path.is_dir()
        if not exists:
            missing.append(_describe_missing(label, path))
            missing_labels.append(label)

    summary = _build_summary(missing_labels)
    if missing:
        tooltip = "\n".join(missing)
    else:
        tooltip = "TTS runtime assets are ready."

    return TTSRuntimeAssetsStatus(
        available=not missing,
        summary=summary,
        tooltip=tooltip,
        missing=missing,
        checked_paths=checked_paths,
    )

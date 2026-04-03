#!/usr/bin/env python3
"""Download the minimum GPT-SoVITS runtime assets required by this repo.

This project depends on two kinds of things:
1. Python packages, which `install.sh` already installs.
2. Runtime model assets, which are too large to keep in git and must be
   downloaded separately.

For a first Mac smoke test we do not require a custom speaker model. Instead,
we download the official GPT-SoVITS v3 base weights plus the text/audio
frontend models that the current inference path expects locally.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "lj1995/GPT-SoVITS"
ALLOW_PATTERNS = [
    "s1v3.ckpt",
    "s2Gv3.pth",
    "models--nvidia--bigvgan_v2_24khz_100band_256x/*",
    "chinese-roberta-wwm-ext-large/*",
    "chinese-hubert-base/*",
]


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    local_dir = project_root / "GPT_SoVITS" / "pretrained_models"
    local_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading official GPT-SoVITS v3 runtime assets...")
    print(f"Source repo: https://huggingface.co/{REPO_ID}")
    print(f"Target dir : {local_dir}")
    print("This may download several GB and can take a while on first run.")

    # We intentionally download into the exact directory layout expected by
    # the current codebase so the user only needs to point .env at s1v3/s2Gv3.
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(local_dir),
        allow_patterns=ALLOW_PATTERNS,
    )

    print("\nDone. Expected files:")
    print(f"- {local_dir / 's1v3.ckpt'}")
    print(f"- {local_dir / 's2Gv3.pth'}")
    print(f"- {local_dir / 'chinese-roberta-wwm-ext-large'}")
    print(f"- {local_dir / 'chinese-hubert-base'}")
    print(f"- {local_dir / 'models--nvidia--bigvgan_v2_24khz_100band_256x'}")


if __name__ == "__main__":
    main()

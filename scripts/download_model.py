"""Pre-download the OmniVoice snapshot into ./models/OmniVoice.

Run this once on a machine with internet before `docker build`. The Dockerfile
then `COPY`s the folder into the image so prod pods don't need network access
to HuggingFace.

Usage:
    pip install huggingface_hub hf_transfer
    python scripts/download_model.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "k2-fsa/OmniVoice"
DEST = Path(__file__).resolve().parent.parent / "models" / "OmniVoice"


def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    DEST.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=REPO_ID, local_dir=str(DEST))
    # HF leaves a resume cache inside the local dir; strip it so it doesn't
    # inflate the Docker context.
    cache_dir = DEST / ".cache"
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
    print(f"downloaded to: {path}")


if __name__ == "__main__":
    main()

"""Convenience entry point: `python main.py`.

Adds `src/` to sys.path so the tts_service package is importable without
installing or setting PYTHONPATH. Inside the Docker image uvicorn is invoked
directly via CMD, so this file is primarily for local runs.

Env vars (see .env.example):
    HOST, PORT, LOG_LEVEL, MOCK_ENGINE, OMNIVOICE_MODE, OMNIVOICE_MODEL_PATH
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import uvicorn  # noqa: E402

from tts_service.settings import settings  # noqa: E402


def main() -> None:
    uvicorn.run(
        "tts_service.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=1,
    )


if __name__ == "__main__":
    main()

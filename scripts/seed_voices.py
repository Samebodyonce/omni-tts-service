"""Seed reference audio for each voice via voice design.

One-shot helper: loads omnivoice-triton, generates a short utterance per
language using voice design attributes (deterministic at temperature=0),
saves the result under voices/<voice_id>.wav, and prints a JSON stanza ready
to paste into voices/voices.json.

Skip this if you already have real recordings.

Usage:
    python scripts/seed_voices.py --mode hybrid --dtype bf16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

SEEDS = [
    {
        "id": "kk_male",
        "lang": "kk",
        "instruct": "male, middle-aged, medium pitch",
        "text": "Сәлеметсіз бе, мен колл-орталықтың операторымын.",
    },
    {
        "id": "ru_male",
        "lang": "ru",
        "instruct": "male, middle-aged, medium pitch",
        "text": "Здравствуйте, я оператор колл-центра, чем могу помочь.",
    },
    {
        "id": "tr_male",
        "lang": "tr",
        "instruct": "male, middle-aged, medium pitch",
        "text": "Merhaba, ben çağrı merkezi operatörüyüm, size nasıl yardımcı olabilirim.",
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--out-dir", default="voices")
    args = ap.parse_args()

    from omnivoice_triton import create_runner

    runner = create_runner(args.mode, dtype=args.dtype)
    runner.load_model()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config: list[dict] = []
    for seed in SEEDS:
        print(f"generating seed for {seed['id']}…")
        result = runner.generate_voice_design(
            text=seed["text"], instruct=seed["instruct"], language=seed["lang"]
        )
        wav_path = out_dir / f"{seed['id']}.wav"
        sf.write(str(wav_path), result["audio"], int(result["sample_rate"]))
        print(f"  wrote {wav_path} ({result['time_s']*1000:.0f} ms)")
        config.append({
            "id": seed["id"],
            "lang": seed["lang"],
            "mode": "clone",
            "ref_audio": f"{seed['id']}.wav",
            "ref_text": seed["text"],
            "description": f"Seeded via voice design: {seed['instruct']}",
        })

    print("\nPaste this into voices/voices.json:\n")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

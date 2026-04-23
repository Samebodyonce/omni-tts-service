"""Concurrent load-test for the TTS service.

Usage:
    python scripts/bench_service.py --url http://localhost:8000 --rps 10 --duration 30
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

import httpx

PHRASES = {
    "ru": [
        "Здравствуйте, вы позвонили в колл-центр.",
        "Пожалуйста, оставайтесь на линии, оператор скоро ответит.",
        "Спасибо за ваш звонок, хорошего дня.",
    ],
    "kk": [
        "Сәлеметсіз бе, біздің колл-орталыққа қоңырау шалдыңыз.",
        "Күте тұрыңыз, оператор жауап береді.",
    ],
    "tr": [
        "Merhaba, çağrı merkezimize hoş geldiniz.",
        "Lütfen hatta kalın, operatörümüz sizi bilgilendirecek.",
    ],
}

VOICES = {"ru": "ru_default", "kk": "kk_default", "tr": "tr_default"}


async def one_request(client: httpx.AsyncClient, url: str) -> tuple[float, int]:
    lang = random.choice(list(PHRASES.keys()))
    payload = {"text": random.choice(PHRASES[lang]), "lang": lang, "voice": VOICES[lang]}
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{url}/tts", json=payload, timeout=30.0)
        return (time.perf_counter() - t0) * 1000.0, r.status_code
    except Exception:
        return (time.perf_counter() - t0) * 1000.0, 0


async def run(url: str, rps: int, duration: float) -> None:
    interval = 1.0 / rps
    results: list[tuple[float, int]] = []
    async with httpx.AsyncClient() as client:
        deadline = time.perf_counter() + duration
        tasks: list[asyncio.Task] = []
        next_fire = time.perf_counter()
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now < next_fire:
                await asyncio.sleep(next_fire - now)
            tasks.append(asyncio.create_task(one_request(client, url)))
            next_fire += interval
        results = await asyncio.gather(*tasks)

    latencies = [ms for ms, _ in results]
    codes: dict[int, int] = {}
    for _, code in results:
        codes[code] = codes.get(code, 0) + 1

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        xs = sorted(xs)
        k = int(len(xs) * p)
        return xs[min(k, len(xs) - 1)]

    print(f"sent={len(results)} duration={duration}s target_rps={rps}")
    print(f"status: {codes}")
    if latencies:
        print(
            "latency ms: "
            f"min={min(latencies):.0f} "
            f"p50={statistics.median(latencies):.0f} "
            f"p90={pct(latencies, 0.9):.0f} "
            f"p99={pct(latencies, 0.99):.0f} "
            f"max={max(latencies):.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rps", type=int, default=5)
    ap.add_argument("--duration", type=float, default=15.0)
    args = ap.parse_args()
    asyncio.run(run(args.url, args.rps, args.duration))


if __name__ == "__main__":
    main()

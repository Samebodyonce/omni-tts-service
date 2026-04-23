"""API integration tests running against the mock engine.

MOCK_ENGINE=1 skips the omnivoice-triton load, so these tests don't need a GPU.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("MOCK_ENGINE", "1")
os.environ.setdefault("PREWARM", "0")
os.environ.setdefault("VOICES_CONFIG", "./voices/voices.json")

from tts_service.main import app  # noqa: E402


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        # lifespan: wait for engine ready
        async with app.router.lifespan_context(app):
            # Give the mock worker thread a beat to hit _ready.set()
            for _ in range(50):
                r = await c.get("/health")
                if r.json()["status"] == "ok":
                    break
                import asyncio
                await asyncio.sleep(0.05)
            yield c


async def test_health_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mock"] is True


async def test_voices_list(client):
    r = await client.get("/voices")
    assert r.status_code == 200
    ids = {v["id"] for v in r.json()["voices"]}
    assert {"kk_default", "ru_default", "tr_default"} <= ids


async def test_tts_pcm(client):
    r = await client.post(
        "/tts", json={"text": "Hello", "lang": "ru", "voice": "ru_default"}
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["x-sample-rate"] == "8000"
    assert r.headers["x-sample-format"] == "s16le"
    assert len(r.content) > 0
    # Even-byte count (int16)
    assert len(r.content) % 2 == 0


async def test_tts_wav(client):
    r = await client.post(
        "/tts?fmt=wav",
        json={"text": "Hello", "lang": "ru", "voice": "ru_default"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.content[:4] == b"RIFF"


async def test_tts_lang_mismatch(client):
    r = await client.post(
        "/tts", json={"text": "Hi", "lang": "kk", "voice": "ru_default"}
    )
    assert r.status_code == 400


async def test_tts_unknown_voice(client):
    r = await client.post(
        "/tts", json={"text": "Hi", "lang": "ru", "voice": "nope"}
    )
    assert r.status_code == 400


async def test_tts_text_too_long(client):
    long_text = "a" * 10_000
    r = await client.post(
        "/tts", json={"text": long_text, "lang": "ru", "voice": "ru_default"}
    )
    assert r.status_code == 400

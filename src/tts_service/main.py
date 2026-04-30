import io
import logging
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path as PathLib
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .engine import QueueFullError, RequestTimeoutError, TTSEngine
from .schemas import HealthResponse, Lang, TTSRequest, VoiceMode, VoicesResponse
from .settings import settings
from .voices import Voice, VoiceRegistry

log = logging.getLogger("tts_service")


def _setup_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    voices = VoiceRegistry()
    voices.load(settings.voices_config, settings.voices_dir)
    engine = TTSEngine(settings=settings, voices=voices)
    await engine.start()
    app.state.engine = engine
    app.state.voices = voices
    try:
        yield
    finally:
        await engine.stop()


app = FastAPI(
    title="Telephony TTS",
    version="0.1.0",
    description="Low-latency TTS (8 kHz raw PCM) for call-center use. Powered by omnivoice-triton.",
    lifespan=lifespan,
)

_STATIC_DIR = PathLib(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    engine: TTSEngine = app.state.engine
    err = engine.load_error()
    status: Literal["ok", "loading", "error"] = (
        "error" if err is not None else ("ok" if engine.is_ready() else "loading")
    )
    return HealthResponse(
        status=status,
        mode=settings.omnivoice_mode,
        mock=settings.mock_engine,
        queue_size=engine.queue_size(),
        queue_maxsize=settings.queue_maxsize,
    )


@app.get("/voices", response_model=VoicesResponse)
async def list_voices() -> VoicesResponse:
    voices: VoiceRegistry = app.state.voices
    return VoicesResponse(voices=voices.list())


@app.post(
    "/tts",
    responses={
        200: {
            "content": {
                "application/octet-stream": {},
                "audio/wav": {},
            },
            "description": "Raw 16-bit LE PCM (8 kHz mono) or WAV.",
        },
        400: {"description": "Bad request (unknown voice, text too long, lang mismatch)"},
        503: {"description": "Queue full or model not ready"},
        504: {"description": "Timed out waiting for GPU worker"},
    },
)
async def tts(
    req: TTSRequest,
    fmt: Literal["pcm", "wav"] = Query("pcm", description="Wire format"),
) -> Response:
    engine: TTSEngine = app.state.engine
    voices: VoiceRegistry = app.state.voices

    if len(req.text) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"text too long ({len(req.text)} > {settings.max_text_length})",
        )
    if not engine.is_ready():
        err = engine.load_error()
        if err is not None:
            raise HTTPException(status_code=503, detail=f"engine failed to load: {err}")
        raise HTTPException(status_code=503, detail="engine still loading")

    try:
        voice = voices.get(req.voice)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unknown voice: {req.voice}")
    if voice.lang != req.lang:
        raise HTTPException(
            status_code=400,
            detail=f"voice {voice.id} is registered for lang={voice.lang}, got {req.lang}",
        )

    try:
        pcm = await engine.synthesize(req.text, voice)
    except QueueFullError:
        raise HTTPException(status_code=503, detail="queue full, retry later")
    except RequestTimeoutError:
        raise HTTPException(status_code=504, detail="gpu worker timed out")

    if fmt == "wav":
        return Response(
            content=_wrap_wav(pcm, settings.sample_rate_out),
            media_type="audio/wav",
            headers={"X-Sample-Rate": str(settings.sample_rate_out)},
        )
    return Response(
        content=pcm,
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(settings.sample_rate_out),
            "X-Sample-Format": "s16le",
            "X-Channels": "1",
        },
    )


@app.post(
    "/tts/generate",
    responses={
        200: {
            "content": {"application/octet-stream": {}, "audio/wav": {}},
            "description": "Raw 16-bit LE PCM (8 kHz mono) or WAV.",
        },
        400: {"description": "Bad request"},
        503: {"description": "Queue full or model not ready"},
        504: {"description": "Timed out waiting for GPU worker"},
    },
)
async def tts_generate(
    text: str = Form(..., description="Текст для синтеза"),
    lang: Lang = Form(..., description="Код языка (kk/ru/tr)"),
    mode: VoiceMode = Form("auto", description="Режим: auto / design / clone"),
    instruct: str | None = Form(None, description="Описание голоса для design"),
    ref_text: str | None = Form(None, description="Транскрипция reference-аудио для clone"),
    ref_audio: UploadFile | None = File(None, description="WAV 3–10 с для clone"),
    fmt: Literal["pcm", "wav"] = Query("wav", description="Wire format"),
) -> Response:
    """Ad-hoc TTS without pre-registering a voice — drives the UI tabs."""
    engine: TTSEngine = app.state.engine

    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"text too long ({len(text)} > {settings.max_text_length})",
        )
    if not engine.is_ready():
        err = engine.load_error()
        if err is not None:
            raise HTTPException(status_code=503, detail=f"engine failed to load: {err}")
        raise HTTPException(status_code=503, detail="engine still loading")

    voice = Voice(id=f"adhoc_{lang}_{mode}", lang=lang, mode=mode)
    tmp_path: PathLib | None = None
    if mode == "design":
        if not instruct:
            raise HTTPException(status_code=400, detail="design mode requires `instruct`")
        voice.instruct = instruct
    elif mode == "clone":
        if ref_audio is None or not ref_audio.filename:
            raise HTTPException(status_code=400, detail="clone mode requires `ref_audio`")
        if not ref_text:
            raise HTTPException(status_code=400, detail="clone mode requires `ref_text`")
        suffix = PathLib(ref_audio.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(await ref_audio.read())
            tmp_path = PathLib(f.name)
        voice.ref_audio_path = tmp_path
        voice.ref_text = ref_text

    try:
        pcm = await engine.synthesize(text, voice)
    except QueueFullError:
        raise HTTPException(status_code=503, detail="queue full, retry later")
    except RequestTimeoutError:
        raise HTTPException(status_code=504, detail="gpu worker timed out")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    if fmt == "wav":
        return Response(
            content=_wrap_wav(pcm, settings.sample_rate_out),
            media_type="audio/wav",
            headers={"X-Sample-Rate": str(settings.sample_rate_out)},
        )
    return Response(
        content=pcm,
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(settings.sample_rate_out),
            "X-Sample-Format": "s16le",
            "X-Channels": "1",
        },
    )


def _wrap_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# Convenience: uniform error JSON shape.
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

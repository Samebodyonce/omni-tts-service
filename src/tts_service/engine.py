"""Single-GPU TTS engine.

Architecture: one background thread owns the CUDA context and runs `omnivoice-triton`
sequentially (CUDA Graph fixes batch shape, so parallel inference on one GPU
breaks the graph). FastAPI handlers enqueue jobs onto an asyncio.Queue and await
a future that the worker thread fulfills via `loop.call_soon_threadsafe`.

Backpressure: queue has a bounded size; overflows raise QueueFull (→ HTTP 503).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from concurrent.futures import Future
from typing import Any

import numpy as np

from .audio import sine_pcm, to_pcm16_8k
from .settings import Settings
from .voices import Voice, VoiceRegistry

log = logging.getLogger(__name__)


@dataclasses.dataclass
class _Job:
    text: str
    voice: Voice
    future: Future
    enqueued_at: float


class TTSEngine:
    def __init__(self, settings: Settings, voices: VoiceRegistry) -> None:
        self.settings = settings
        self.voices = voices
        self._queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=settings.queue_maxsize)
        self._worker_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: Any = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._load_error: Exception | None = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._worker_thread = threading.Thread(
            target=self._worker_main, name="tts-worker", daemon=True
        )
        self._worker_thread.start()
        # Don't block startup on model load — /health will report `loading`.
        log.info("engine worker started (mock=%s mode=%s)",
                 self.settings.mock_engine, self.settings.omnivoice_mode)

    async def stop(self) -> None:
        self._stop.set()
        # Wake the worker if it is idle on the queue.
        try:
            self._queue.put_nowait(
                _Job(text="", voice=None, future=Future(), enqueued_at=0.0)  # type: ignore[arg-type]
            )
        except asyncio.QueueFull:
            pass
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5)
        log.info("engine stopped")

    # ------------------------------------------------------------------ public

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def load_error(self) -> Exception | None:
        return self._load_error

    def queue_size(self) -> int:
        return self._queue.qsize()

    async def synthesize(self, text: str, voice: Voice) -> bytes:
        loop = asyncio.get_running_loop()
        fut: Future = loop.create_future()  # type: ignore[assignment]
        job = _Job(text=text, voice=voice, future=fut, enqueued_at=time.perf_counter())
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as e:
            raise QueueFullError("request queue is full") from e

        try:
            return await asyncio.wait_for(fut, timeout=self.settings.request_timeout)
        except asyncio.TimeoutError as e:
            raise RequestTimeoutError("timed out waiting for GPU worker") from e

    # ------------------------------------------------------------------ worker

    def _worker_main(self) -> None:
        try:
            self._load_runner()
            self._ready.set()
        except Exception as e:
            log.exception("failed to load runner: %s", e)
            self._load_error = e
            # Drain the queue with failures so callers don't hang.
            self._drain_with_error(e)
            return

        if self.settings.prewarm:
            try:
                self._prewarm()
            except Exception:
                log.exception("prewarm failed (continuing)")

        while not self._stop.is_set():
            try:
                job = asyncio.run_coroutine_threadsafe(self._queue.get(), self._loop).result()
            except Exception:
                log.exception("worker queue read failed")
                continue
            if self._stop.is_set():
                break
            if job.voice is None:  # shutdown sentinel
                continue
            self._handle_job(job)

    def _handle_job(self, job: _Job) -> None:
        t0 = time.perf_counter()
        try:
            pcm = self._run_inference(job.text, job.voice)
        except Exception as e:
            log.exception("inference failed voice=%s", job.voice.id)
            self._set_future_exception(job.future, e)
            return

        wait_ms = (t0 - job.enqueued_at) * 1000.0
        infer_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "tts ok voice=%s chars=%d wait_ms=%.1f infer_ms=%.1f bytes=%d",
            job.voice.id, len(job.text), wait_ms, infer_ms, len(pcm),
        )
        self._set_future_result(job.future, pcm)

    def _run_inference(self, text: str, voice: Voice) -> bytes:
        if self.settings.mock_engine:
            # Return ~0.25s of sine, already at the target sample rate.
            return sine_pcm(duration_s=0.25 + 0.02 * len(text), sr=self.settings.sample_rate_out)

        if voice.mode == "clone":
            result = self._runner.generate_voice_clone(
                text=text,
                ref_audio=str(voice.ref_audio_path),
                ref_text=voice.ref_text or "",
                language=voice.lang,
            )
        elif voice.mode == "design":
            result = self._runner.generate_voice_design(
                text=text,
                instruct=voice.instruct or "",
                language=voice.lang,
            )
        else:
            result = self._runner.generate(text=text, language=voice.lang)

        audio: np.ndarray = result["audio"]
        src_sr: int = int(result["sample_rate"])
        return to_pcm16_8k(audio, src_sr=src_sr, dst_sr=self.settings.sample_rate_out)

    def _load_runner(self) -> None:
        if self.settings.mock_engine:
            log.warning("MOCK_ENGINE=1 — skipping omnivoice-triton load")
            return

        from omnivoice_triton import create_runner

        dtype_map = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
        dtype = dtype_map[self.settings.omnivoice_dtype]
        device = f"cuda:{self.settings.cuda_device}"

        model_id = self.settings.omnivoice_model_path
        log.info("loading omnivoice-triton runner mode=%s dtype=%s device=%s model_id=%s",
                 self.settings.omnivoice_mode, dtype, device, model_id)
        self._runner = create_runner(
            self.settings.omnivoice_mode,
            device=device,
            dtype=dtype,
            model_id=model_id,
        )
        self._runner.load_model()
        log.info("runner loaded")

    def _prewarm(self) -> None:
        if self.settings.mock_engine:
            return
        log.info("prewarming runner (one dummy generate per voice)")
        warmup_text = "Hello."
        for voice in self.voices.list():
            full = self.voices.get(voice.id)
            try:
                self._run_inference(warmup_text, full)
                log.info("prewarmed voice=%s", voice.id)
            except Exception:
                log.exception("prewarm voice=%s failed", voice.id)

    def _drain_with_error(self, err: Exception) -> None:
        while True:
            try:
                job = asyncio.run_coroutine_threadsafe(
                    self._queue.get(), self._loop
                ).result(timeout=0.5)
            except Exception:
                return
            if job.voice is not None:
                self._set_future_exception(job.future, err)

    def _set_future_result(self, fut: Future, value: bytes) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(
            lambda f=fut, v=value: f.set_result(v) if not f.done() else None
        )

    def _set_future_exception(self, fut: Future, exc: Exception) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(
            lambda f=fut, e=exc: f.set_exception(e) if not f.done() else None
        )


class QueueFullError(RuntimeError):
    pass


class RequestTimeoutError(RuntimeError):
    pass

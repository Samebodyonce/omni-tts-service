# Telephony TTS

Low-latency TTS service for call-center use. Accepts `{text, lang, voice}`,
returns raw 16-bit LE PCM mono at 8 kHz — the wire format PBX / SIP stacks expect.

Backed by [omnivoice-triton](https://github.com/newgrit1004/omnivoice-triton)
(OmniVoice + OpenAI-Triton kernel fusion). Hybrid mode generates audio at
~3.4× faster than the PyTorch baseline on RTX 5090.

The model is **baked into the image** — no HuggingFace access needed at runtime.

## Supported languages / voices

| Lang | Voice id | Mode |
| --- | --- | --- |
| kk (Kazakh)  | `kk_default` | auto |
| ru (Russian) | `ru_default` | auto |
| tr (Turkish) | `tr_default` | auto |

Each voice can run in one of three modes (see `voices/README.md`):

- **`auto`** — OmniVoice picks a voice automatically. Default until you have recordings.
- **`design`** — describe the voice via `instruct`, e.g. `"male, low pitch"`.
  Voice design was trained on zh/en only — may be unstable for kk/ru/tr.
- **`clone`** — best option. Drop a 3–10 s `.wav` into `voices/`, point
  `voices.json` at it, set `ref_text`, restart.

## Build

### 1. Pre-download the model (one-time, on a box with internet)

```bash
pip install huggingface_hub hf_transfer
python scripts/download_model.py
# → ./models/OmniVoice/  (~2 GB)
```

### 2. Build the image

```bash
docker build -t tts-service:0.1.0 .
```

The final image is ~5 GB (CUDA base + PyTorch cu128 + model weights). Push to
your registry (`docker push ...`) and it's ready for Kubernetes.

### 3. Run locally (requires nvidia-container-toolkit)

```bash
docker run --rm --gpus all -p 8000:8000 \
    -e OMNIVOICE_MODE=hybrid \
    tts-service:0.1.0
```

First pod start takes ~30–60 s while the model loads onto the GPU; `/health`
reports `loading` until it's done.

## Kubernetes

Minimal example — assumes `tts-service:0.1.0` is in a reachable registry and
the cluster has the NVIDIA device plugin installed.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tts
spec:
  replicas: 1                 # bump per GPU node; each replica pins one GPU
  selector:
    matchLabels: { app: tts }
  template:
    metadata:
      labels: { app: tts }
    spec:
      containers:
        - name: tts
          image: your-registry/tts-service:0.1.0
          ports:
            - containerPort: 8000
          env:
            - { name: OMNIVOICE_MODE, value: hybrid }
            - { name: QUEUE_MAXSIZE, value: "64" }
            - { name: REQUEST_TIMEOUT, value: "30" }
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: 6Gi
            requests:
              nvidia.com/gpu: 1
              memory: 4Gi
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 30
            periodSeconds: 5
            failureThreshold: 24   # ~2 min warm-up budget
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 120
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata: { name: tts }
spec:
  selector: { app: tts }
  ports: [{ port: 80, targetPort: 8000 }]
```

Scaling: one replica per GPU. CUDA Graph uses a fixed batch shape, so two
requests can't share a replica — increase replicas (and GPU nodes) to raise RPS.

## API

### `POST /tts`

Request body (JSON):

```json
{ "text": "Здравствуйте, вы позвонили в колл-центр.", "lang": "ru", "voice": "ru_default" }
```

Query params:

- `fmt=pcm` (default) — body is raw `int16 LE` @ 8 kHz mono.
- `fmt=wav` — body is a WAV container around the same PCM.

Response headers on PCM responses:

```
Content-Type: application/octet-stream
X-Sample-Rate: 8000
X-Sample-Format: s16le
X-Channels: 1
```

Examples:

```bash
curl -sS -X POST http://localhost:8000/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Здравствуйте.","lang":"ru","voice":"ru_default"}' \
  --output out.pcm

# listen via ffplay:
ffplay -autoexit -f s16le -ar 8000 -ac 1 out.pcm

# or ask for a WAV:
curl -sS -X POST 'http://localhost:8000/tts?fmt=wav' \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello.","lang":"ru","voice":"ru_default"}' \
  --output out.wav
```

### `GET /voices`

Lists registered voices with their language and mode.

### `GET /health`

```json
{ "status": "ok", "mode": "hybrid", "mock": false, "queue_size": 0, "queue_maxsize": 64 }
```

`status=loading` during cold start (~30–60 s while weights load);
`status=error` if the runner failed — check pod logs.

## Concurrency / throughput

OmniVoice in **Hybrid** mode uses a captured CUDA Graph with a fixed batch
shape (BS=1). Two inferences cannot run in the same graph, so the service
serializes requests through a single GPU worker thread. Backpressure is
applied with a bounded `asyncio.Queue`:

- `QUEUE_MAXSIZE=64` — incoming requests that exceed this return `503`.
- `REQUEST_TIMEOUT=30` — clients waiting longer than this get `504`.

To scale: more replicas, one GPU each.

### Inference mode (`OMNIVOICE_MODE`)

| Mode | Speedup | Notes |
| --- | --- | --- |
| `base` | 1.00× | Plain PyTorch — most portable |
| `triton` | ~1.02× | Kernel fusion only |
| `faster` | ~2.75× | CUDA Graph only — good fallback if `hybrid` misbehaves |
| `hybrid` | ~3.4× | Triton + CUDA Graph — **default**, verified on sm_120 only |

`hybrid` is the recommended prod mode. On untested GPUs (sm_86, sm_89) if
you hit kernel errors, switch to `faster` — keeps most of the speedup.

## Local HTTP testing without a GPU

Set `MOCK_ENGINE=1` — the service skips loading omnivoice-triton and returns a
sine wave. Useful for wiring up clients / PBX integration before the GPU box
is ready.

```bash
MOCK_ENGINE=1 PREWARM=0 PYTHONPATH=src python -m uvicorn tts_service.main:app --port 8000
```

## Adding a cloned voice

1. Record (or cut) a 3–10 s clean sample in the target language, 16 kHz+ mono WAV.
2. Save as `voices/ru_female.wav` (before building the image).
3. Edit `voices/voices.json`:

   ```json
   {
     "id": "ru_female",
     "lang": "ru",
     "mode": "clone",
     "ref_audio": "ru_female.wav",
     "ref_text": "Точная транскрипция reference-аудио."
   }
   ```

4. Rebuild the image. Voice list now exposes `ru_female`.

## Dev

```bash
pip install -e ".[dev]"
pytest
python scripts/bench_service.py --url http://localhost:8000 --rps 5 --duration 15
```

## Layout

```
src/tts_service/
  main.py      # FastAPI app: /tts, /voices, /health
  engine.py    # Single-GPU worker + bounded queue
  voices.py    # Voice registry loaded from voices/voices.json
  audio.py     # 24 kHz → 8 kHz, float → int16 PCM
  settings.py  # Pydantic-settings, reads .env
  schemas.py   # Request/response models
scripts/
  download_model.py  # Pre-download OmniVoice snapshot
  seed_voices.py     # Generate seed reference audio (optional)
  bench_service.py   # Concurrent load-test
models/
  OmniVoice/   # Baked into the image by the Dockerfile
voices/
  voices.json  # Voice registry
```

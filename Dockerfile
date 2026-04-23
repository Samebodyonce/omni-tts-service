# CUDA 12.8 runtime with cuDNN — matches omnivoice-triton's tested stack (PyTorch 2.8 cu128).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Fully offline: don't let HF try to reach the hub at runtime.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/opt/hf-cache

# Python 3.12 from deadsnakes + libsndfile for soundfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

WORKDIR /app

# Torch must come from the cu128 index — pin the exact versions omnivoice-triton expects.
# Kept as its own layer: ~3 GB, never invalidated by source changes.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0+cu128 torchaudio==2.8.0+cu128

# App deps (light, separated so code changes don't invalidate the heavy torch layer).
RUN pip install \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" \
        "pydantic>=2.7" "pydantic-settings>=2.4" \
        "numpy>=1.26" "scipy>=1.13" "soundfile>=0.12" \
        "omnivoice-triton>=0.1"

# Bake the model into the image. `models/OmniVoice/` must exist in the build context.
# Download it before building:  python scripts/download_model.py
COPY models /app/models

COPY src ./src
COPY voices ./voices

ENV PYTHONPATH=/app/src \
    OMNIVOICE_MODEL_PATH=/app/models/OmniVoice

EXPOSE 8000

# Probe from inside the container; K8s will also add its own liveness/readiness.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=20 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5); sys.exit(0 if b'\"ok\"' in r.read() else 1)" || exit 1

CMD ["python", "-m", "uvicorn", "tts_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

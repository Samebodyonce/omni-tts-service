# App image: builds on the pre-baked base (CUDA + Python + torch cu128 + deps).
# Rebuild the base with `docker build -f Dockerfile.base -t samebodyonce/tts-service-base:0.1.0 .`
# whenever heavy deps (torch, omnivoice-triton) change.
FROM samebodyonce/tts-service-base:0.2.0

# Needed by FastAPI's Form/File handling for the /tts/generate UI endpoint.
RUN pip install "python-multipart>=0.0.9"

# Model weights baked in so prod pods don't need network access to HF.
# Pre-download locally:  python scripts/download_model.py
COPY models /app/models

COPY src ./src
COPY voices ./voices

ENV OMNIVOICE_MODEL_PATH=/app/models/OmniVoice

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=20 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5); sys.exit(0 if b'\"ok\"' in r.read() else 1)" || exit 1

CMD ["python", "-m", "uvicorn", "tts_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM python:3.12-slim

# Hugging Face Spaces runs containers as user 1000; the app writes sessions.json,
# checkpoints.db and embedding_cache/ at runtime, so it must own /app.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PAPEER_PUBLIC=1 \
    PORT=7860

WORKDIR /app

# ── Layer 1: runtime deps only (cached until deploy/requirements.txt changes) ────
COPY --chown=user deploy/requirements.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Layer 2: app code and theme ──────────────────────────────────────────────
COPY --chown=user backend/ backend/
COPY --chown=user .streamlit/ .streamlit/
COPY --chown=user app.py .

EXPOSE 7860

# API keys come from the host's secrets as env vars (never baked into the image).
CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.maxUploadSize=10 \
    --browser.gatherUsageStats=false

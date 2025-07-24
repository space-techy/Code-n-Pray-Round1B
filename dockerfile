# ---------- base image ----------
FROM --platform=linux/amd64 python:3.10-slim

# ---------- system & Python deps ----------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- copy local source ----------
COPY main.py          /app/
COPY process_pdfs.py  /app/

# ---------- download & cache HF models ----------
#  1) gtr‑t5‑base (≈ 540 MB)
#  2) cross‑encoder/ms‑marco‑MiniLM‑L‑6‑v2 (≈ 65 MB)

RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
# cache dir = /app/models  (see ENV below)
SentenceTransformer('sentence-transformers/gtr-t5-base').save('./models/sentence-transformers/gtr-t5-base')                   # base encoder
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2').save('./models/cross-encoder/ms-marco-MiniLM-L-6-v2')                        # reranker
PY

# ---------- environment ----------
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/main.py"]

FROM python:3.11-slim

# HF Spaces runs containers as non-root uid 1000 by convention.
RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --user --no-cache-dir "torch>=2.6.0" --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user --no-cache-dir -r requirements.txt

COPY --chown=user app.py .

# HF Spaces convention: listen on 7860
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

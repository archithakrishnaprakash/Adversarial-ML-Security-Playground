# Backend: FastAPI + PyTorch + scikit-learn
FROM python:3.11-slim

WORKDIR /app

# system deps matplotlib needs for its Agg backend at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# --workers 1 matters here, not just for resource reasons: trained models and
# the threat-model store live in this process's memory (see README "Migration
# notes" / MODEL_STORE). Multiple workers would each have their own empty
# model store, so a model trained via one request might 404 on the next.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

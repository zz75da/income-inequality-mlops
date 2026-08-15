FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY train-api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Bring in the whole repo so train-api can invoke ingestion/*.py and
# features/build_features.py as subprocesses (same layout as running locally).
COPY ingestion /app/ingestion
COPY features /app/features
COPY train-api /app/train-api
COPY params.yaml /app/params.yaml

WORKDIR /app/train-api
ENV PYTHONPATH=/app/train-api/services

EXPOSE 5002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5002"]

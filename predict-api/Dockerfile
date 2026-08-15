FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY predict-api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY predict-api /app/predict-api
COPY params.yaml /app/params.yaml

WORKDIR /app/predict-api
EXPOSE 5003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

COPY predict-api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY predict-api /app/predict-api
COPY params.yaml /app/params.yaml
# .dvc/config + dvc.lock are what the cold-start dvc-pull bootstrap in app.py
# needs to resolve the remote and know what to fetch — no data/ volume mount
# on a standalone hosted deploy (Render/Railway/Fly.io), unlike docker-compose.
# .git is NOT copied in here (unlike a naive first attempt at this) — most
# managed build platforms (Render confirmed) deliberately exclude .git from
# what they send to the Docker builder, so `COPY .git` fails there even
# though it works in a local build. The bootstrap instead runs `git init` at
# runtime if needed, which satisfies dvc pull's SCM check identically
# everywhere regardless of the build context — see app.py.
COPY .dvc/config /app/.dvc/config
COPY dvc.lock /app/dvc.lock
COPY dvc.yaml /app/dvc.yaml

WORKDIR /app/predict-api
EXPOSE 5003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]

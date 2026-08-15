FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY predict-api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY predict-api /app/predict-api
COPY params.yaml /app/params.yaml
# .dvc/config + dvc.lock are what the cold-start dvc-pull bootstrap in app.py
# needs to resolve the remote and know what to fetch — no data/ volume mount
# on a standalone hosted deploy (Render/Railway/Fly.io), unlike docker-compose.
# .git is required too: `dvc pull` errors with "not a git repository" without
# it (this repo was dvc-init'd in normal git-tracked mode, not --no-scm) — a
# real gap only found by actually running dvc pull in an isolated container
# with no volume mount, not by reasoning about it. 1.3MB, trivial to include.
COPY .dvc/config /app/.dvc/config
COPY .git /app/.git
COPY dvc.lock /app/dvc.lock
COPY dvc.yaml /app/dvc.yaml

WORKDIR /app/predict-api
EXPOSE 5003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]

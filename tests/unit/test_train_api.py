import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, import_service_app):
    train_app = import_service_app("train-api")

    monkeypatch.setattr(train_app, "JOBS", {})
    monkeypatch.setattr(train_app, "_ACTIVE_JOB", None)

    def fake_run_script(path, job_id):
        train_app.JOBS[job_id]["log"].append({"script": str(path), "returncode": 0})
        return True

    monkeypatch.setattr(train_app, "_run_script", fake_run_script)
    return TestClient(train_app.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_train_invalid_target(client):
    resp = client.post("/train", json={"target": "not-a-real-target"})
    assert resp.status_code == 400


def test_train_starts_job_and_reports_status(client):
    resp = client.post("/train", json={"target": "gini", "run_ingestion": False})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert resp.json()["status"] == "running"

    # allow the background thread to finish
    for _ in range(50):
        status_resp = client.get(f"/train/status/{job_id}")
        if status_resp.json()["status"] in ("success", "failed"):
            break
        time.sleep(0.05)

    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "success"


def test_train_accepts_mobility_rank_target(client):
    resp = client.post("/train", json={"target": "mobility_rank", "run_ingestion": False})
    assert resp.status_code == 202


def test_train_status_unknown_job(client):
    resp = client.get("/train/status/does-not-exist")
    assert resp.status_code == 404

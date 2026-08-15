import json

import pytest


class FakeModelVersion:
    def __init__(self, version):
        self.version = version


class FakeMlflowClient:
    def __init__(self):
        self.transitions = []

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        self.transitions.append(
            {"name": name, "version": version, "stage": stage, "archive_existing_versions": archive_existing_versions}
        )


class FakeTracking:
    def __init__(self, client):
        self._client = client

    def MlflowClient(self):
        return self._client


class FakeMlflow:
    """Stands in for the whole mlflow package — register_and_promote() only
    ever touches this injected object, never `import mlflow` directly, so no
    real mlflow install is needed to test the gate logic."""

    def __init__(self):
        self.registered = []
        self.client = FakeMlflowClient()
        self.tracking = FakeTracking(self.client)

    def register_model(self, model_uri, name):
        self.registered.append((model_uri, name))
        return FakeModelVersion(version="1")


PARAMS = {"promotion_gates": {"gini": {"metric": "r2", "min": 0.4}}}


@pytest.fixture
def common(import_service_module):
    return import_service_module("train-api/services", "common.py")


def test_register_and_promote_promotes_when_gate_passes(tmp_path, monkeypatch, common):
    monkeypatch.setattr(common, "ARTIFACTS_DIR", tmp_path)
    mlflow = FakeMlflow()

    status = common.register_and_promote(mlflow, "gini", "runs:/abc/model_gini", {"r2": 0.6}, PARAMS)

    assert status["stage"] == "Production"
    assert status["passed"] is True
    assert mlflow.client.transitions[0]["stage"] == "Production"
    assert mlflow.client.transitions[0]["archive_existing_versions"] is True


def test_register_and_promote_stays_staging_when_gate_fails(tmp_path, monkeypatch, common):
    monkeypatch.setattr(common, "ARTIFACTS_DIR", tmp_path)
    mlflow = FakeMlflow()

    status = common.register_and_promote(mlflow, "gini", "runs:/abc/model_gini", {"r2": 0.1}, PARAMS)

    assert status["stage"] == "Staging"
    assert status["passed"] is False
    assert mlflow.client.transitions[0]["archive_existing_versions"] is False


def test_register_and_promote_only_updates_its_own_key(tmp_path, monkeypatch, common):
    monkeypatch.setattr(common, "ARTIFACTS_DIR", tmp_path)
    (tmp_path / "registry_status.json").write_text(json.dumps({"mobility": {"stage": "Production"}}))
    mlflow = FakeMlflow()

    common.register_and_promote(mlflow, "gini", "runs:/abc/model_gini", {"r2": 0.6}, PARAMS)

    all_status = json.loads((tmp_path / "registry_status.json").read_text())
    assert all_status["mobility"] == {"stage": "Production"}  # untouched
    assert all_status["gini"]["stage"] == "Production"

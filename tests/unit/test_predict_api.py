import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    import app as predict_app

    class DummyRegressor:
        def predict(self, X):
            return [42.0]

    class DummyClassifier:
        def predict_proba(self, X):
            return [[0.1, 0.2, 0.6, 0.1]]

    class DummyLabelEncoder:
        classes_ = ["High income", "Low income", "Lower middle income", "Upper middle income"]

        def inverse_transform(self, idx):
            return [self.classes_[idx[0]]]

    monkeypatch.setattr(predict_app, "_models", {
        "gini": DummyRegressor(),
        "mobility": DummyRegressor(),
        "income_group": {"model": DummyClassifier(), "label_encoder": DummyLabelEncoder()},
    })
    monkeypatch.setattr(predict_app, "_categorical_mappings", {"region": {"Europe": 0}, "income_group_lag1": {"UNKNOWN": 0}})
    monkeypatch.setattr(predict_app, "record_prediction", lambda *a, **k: None)

    return TestClient(predict_app.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert set(body["models_loaded"]) == {"gini", "mobility", "income_group"}


def test_predict_gini(client):
    resp = client.post("/predict-gini", json={"gdp_per_capita_ppp": 30000, "region": "Europe"})
    assert resp.status_code == 200
    assert resp.json() == {"gini_index": 42.0}


def test_predict_mobility(client):
    resp = client.post("/predict-mobility", json={})
    assert resp.status_code == 200
    assert resp.json() == {"intergen_income_elasticity": 42.0}


def test_predict_income_group(client):
    resp = client.post("/predict-income-group", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["income_group"] == "Lower middle income"
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)


def test_predict_gini_503_when_model_missing(monkeypatch):
    import app as predict_app

    monkeypatch.setattr(predict_app, "_models", {})
    client = TestClient(predict_app.app)
    resp = client.post("/predict-gini", json={})
    assert resp.status_code == 503

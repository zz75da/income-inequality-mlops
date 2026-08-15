import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, import_service_app):
    predict_app = import_service_app("predict-api")

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

    monkeypatch.setattr(
        predict_app,
        "_models",
        {
            "gini": DummyRegressor(),
            "mobility": DummyRegressor(),
            "income_group": {"model": DummyClassifier(), "label_encoder": DummyLabelEncoder()},
        },
    )
    monkeypatch.setattr(
        predict_app, "_categorical_mappings", {"region": {"Europe": 0}, "income_group_lag1": {"UNKNOWN": 0}}
    )
    monkeypatch.setattr(predict_app, "_metrics", {"gini": {"residual_std": 5.0}, "mobility": {"residual_std": 0.1}})
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
    body = resp.json()
    assert body["gini_index"] == 42.0
    assert body["interval_80pct"] == pytest.approx([42.0 - 1.28 * 5.0, 42.0 + 1.28 * 5.0])


def test_predict_mobility(client):
    resp = client.post("/predict-mobility", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["intergen_income_elasticity"] == 42.0
    assert body["interval_80pct"] == pytest.approx([42.0 - 1.28 * 0.1, 42.0 + 1.28 * 0.1])


def test_predict_income_group(client):
    resp = client.post("/predict-income-group", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["income_group"] == "Lower middle income"
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)


def test_explain_gini(client, monkeypatch, import_service_app):
    predict_app = import_service_app("predict-api")
    monkeypatch.setattr(predict_app, "explain", lambda name, X, **kw: {"gdp_per_capita_ppp": 0.5, "region_code": -0.1})

    resp = client.post("/explain-gini", json={"gdp_per_capita_ppp": 30000, "region": "Europe"})
    assert resp.status_code == 200
    assert resp.json() == {"contributions": {"gdp_per_capita_ppp": 0.5, "region_code": -0.1}}


def test_explain_gini_503_when_no_explainer(client, monkeypatch, import_service_app):
    predict_app = import_service_app("predict-api")
    monkeypatch.setattr(predict_app, "explain", lambda name, X, **kw: None)

    resp = client.post("/explain-gini", json={})
    assert resp.status_code == 503


def test_explain_income_group_reports_predicted_class(client, monkeypatch, import_service_app):
    predict_app = import_service_app("predict-api")

    def fake_explain(name, X, class_index=None, **kw):
        assert class_index == 2  # DummyClassifier's predict_proba peaks at index 2
        return {"gdp_per_capita_ppp": 0.3}

    monkeypatch.setattr(predict_app, "explain", fake_explain)

    resp = client.post("/explain-income-group", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["contributions"] == {"gdp_per_capita_ppp": 0.3}
    assert body["explained_class"] == "Lower middle income"


def test_predict_gini_503_when_model_missing(monkeypatch, import_service_app):
    predict_app = import_service_app("predict-api")

    monkeypatch.setattr(predict_app, "_models", {})
    client = TestClient(predict_app.app)
    resp = client.post("/predict-gini", json={})
    assert resp.status_code == 503


def test_predict_mobility_imputes_omitted_fields(monkeypatch, import_service_app):
    """mobility uses RandomForestRegressor, which has no native NaN support —
    a field a caller omits must be median-imputed before it ever reaches
    .predict(), not passed through as NaN."""
    predict_app = import_service_app("predict-api")

    class NaNSensitiveRegressor:
        def predict(self, X):
            assert not X.isna().any().any(), "NaN reached the model — omitted fields weren't imputed"
            return [7.0]

    monkeypatch.setattr(predict_app, "_models", {"mobility": NaNSensitiveRegressor()})
    monkeypatch.setattr(predict_app, "_categorical_mappings", {})
    monkeypatch.setattr(predict_app, "_feature_medians", {"gdp_per_capita_ppp": 12345.0})
    monkeypatch.setattr(predict_app, "record_prediction", lambda *a, **k: None)
    client = TestClient(predict_app.app)

    resp = client.post("/predict-mobility", json={})  # every field omitted
    assert resp.status_code == 200
    assert resp.json() == {"intergen_income_elasticity": 7.0, "interval_80pct": None}

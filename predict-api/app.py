# predict-api/app.py
# ============================================================
# MODULE SUMMARY
# ------------------------------------------------------------
# Role: FastAPI inference service. Loads the 3 trained models
# (gini regressor, mobility regressor, income_group classifier)
# plus the categorical encoding map produced by
# features/build_features.py, and serves predictions from raw
# JSON feature payloads (no need for callers to know about
# pandas .cat.codes internals).
#
# Endpoints:
#   POST /predict-gini            -> {"gini_index": float}
#   POST /predict-mobility        -> {"intergen_income_elasticity": float}
#   POST /predict-income-group    -> {"income_group": str, "probabilities": {...}}
#   POST /reload-artifacts        (reload models from disk after a training run)
#   GET  /drift-status
#   POST /drift-trigger-report
#   GET  /health
#   GET  /metrics
# ============================================================
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from services.drift_monitor import buffer_size, record_prediction, reference_exists, trigger_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("predict-api")

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "data" / "artifacts"

app = FastAPI(title="Income Inequality — Predict API", version="0.1")
Instrumentator().instrument(app).expose(app)

PREDICTION_COUNT = Counter("prediction_total", "Number of predictions served", ["target"])
PREDICTION_LATENCY = Histogram("predict_request_latency_seconds", "Prediction latency", ["target"])

with open(ROOT / "params.yaml") as f:
    PARAMS = yaml.safe_load(f)
NUMERIC_FEATURES = PARAMS["features"]["numeric_features"]
CATEGORICAL_FEATURES = PARAMS["features"]["categorical_features"]

_models: dict = {}
_categorical_mappings: dict = {}


class FeaturePayload(BaseModel):
    # Numeric macro features — all optional so a caller can supply a partial
    # set and let the model's own NaN handling / our median fallback apply.
    gdp_per_capita_ppp: float | None = None
    gdp_growth_pct: float | None = None
    unemployment_rate: float | None = None
    education_expenditure_pct_gdp: float | None = None
    social_spending_pct_gdp: float | None = None
    tax_revenue_pct_gdp: float | None = None
    urban_population_pct: float | None = None
    population_total: float | None = None
    top10_income_share: float | None = None
    bottom50_income_share: float | None = None
    # Categorical
    region: str | None = None
    income_group_lag1: str | None = None

    class Config:
        extra = "allow"  # tolerate extra fields without breaking


def load_artifacts() -> None:
    global _models, _categorical_mappings
    _models = {}
    for name in ("gini", "mobility", "income_group"):
        path = ARTIFACTS_DIR / f"model_{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _models[name] = pickle.load(f)
            logger.info("Loaded model_%s.pkl", name)
        else:
            logger.warning("model_%s.pkl not found at %s — train it first via train-api", name, path)

    mapping_path = ARTIFACTS_DIR / "categorical_mappings.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            _categorical_mappings = json.load(f)
    else:
        logger.warning("categorical_mappings.json not found — categorical features will encode as -1 (unseen)")
        _categorical_mappings = {}


@app.on_event("startup")
def startup_event():
    load_artifacts()


def _build_feature_row(payload: FeaturePayload) -> pd.DataFrame:
    row: dict[str, float | int] = {}
    for col in NUMERIC_FEATURES:
        row[col] = getattr(payload, col, None)
    for col in CATEGORICAL_FEATURES:
        raw_value = getattr(payload, col, None) or "UNKNOWN"
        code = _categorical_mappings.get(col, {}).get(raw_value, -1)
        row[col + "_code"] = code
    return pd.DataFrame([row])


@app.post("/predict-gini")
def predict_gini(payload: FeaturePayload):
    if "gini" not in _models:
        raise HTTPException(status_code=503, detail="model_gini.pkl not loaded — train it first")
    with PREDICTION_LATENCY.labels(target="gini").time():
        X = _build_feature_row(payload)
        pred = float(_models["gini"].predict(X)[0])
        record_prediction(X.iloc[0].to_dict())
    PREDICTION_COUNT.labels(target="gini").inc()
    return {"gini_index": pred}


@app.post("/predict-mobility")
def predict_mobility(payload: FeaturePayload):
    if "mobility" not in _models:
        raise HTTPException(status_code=503, detail="model_mobility.pkl not loaded — train it first")
    with PREDICTION_LATENCY.labels(target="mobility").time():
        X = _build_feature_row(payload)
        pred = float(_models["mobility"].predict(X)[0])
        record_prediction(X.iloc[0].to_dict())
    PREDICTION_COUNT.labels(target="mobility").inc()
    return {"intergen_income_elasticity": pred}


@app.post("/predict-income-group")
def predict_income_group(payload: FeaturePayload):
    if "income_group" not in _models:
        raise HTTPException(status_code=503, detail="model_income_group.pkl not loaded — train it first")
    bundle = _models["income_group"]
    model, label_encoder = bundle["model"], bundle["label_encoder"]
    with PREDICTION_LATENCY.labels(target="income_group").time():
        X = _build_feature_row(payload)
        proba = np.asarray(model.predict_proba(X)[0])
        pred_idx = int(proba.argmax())
        record_prediction(X.iloc[0].to_dict())
    PREDICTION_COUNT.labels(target="income_group").inc()
    return {
        "income_group": label_encoder.inverse_transform([pred_idx])[0],
        "probabilities": {cls: float(p) for cls, p in zip(label_encoder.classes_, proba, strict=False)},
    }


@app.post("/reload-artifacts")
def reload_artifacts():
    load_artifacts()
    return {"status": "reloaded", "models_loaded": list(_models.keys())}


@app.get("/drift-status")
def drift_status():
    return {"buffer_size": buffer_size(), "reference_exists": reference_exists()}


@app.post("/drift-trigger-report")
def drift_trigger_report():
    feature_cols = [c + ("_code" if c in CATEGORICAL_FEATURES else "") for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    path = trigger_report(feature_cols)
    if path is None:
        raise HTTPException(status_code=400, detail="No buffered predictions or no reference data yet")
    return {"report_path": path}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "predict-api", "models_loaded": list(_models.keys())}


@app.get("/")
def root():
    return {
        "status": "API up and running",
        "endpoints": [
            "/predict-gini",
            "/predict-mobility",
            "/predict-income-group",
            "/reload-artifacts",
            "/drift-status",
            "/health",
        ],
    }

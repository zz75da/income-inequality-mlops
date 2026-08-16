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
#   POST /predict-gini            -> {"gini_index": float, "interval_80pct": [lo, hi]}
#   POST /predict-mobility        -> {"intergen_income_elasticity": float, "interval_80pct": [lo, hi]}
#   POST /predict-income-group    -> {"income_group": str, "probabilities": {...}}
#   POST /explain-gini            -> {"contributions": {feature: shap_value, ...}}
#   POST /explain-mobility        -> {"contributions": {feature: shap_value, ...}}
#   POST /explain-income-group    -> {"contributions": {...}, "explained_class": str}
#   POST /reload-artifacts        (reload models from disk after a training run)
#   GET  /drift-status
#   POST /drift-trigger-report
#   GET  /models/registry-status  (MLflow registry stage/version per target, if trained)
#   GET  /health
#   GET  /metrics
# ============================================================
from __future__ import annotations

import json
import logging
import os
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
from services.explain import build_explainers, explain

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
_feature_medians: dict = {}
_metrics: dict = {}


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


def _try_dvc_pull_from_env() -> None:
    """Cold-start bootstrap for a standalone hosted deploy (Render/Railway/
    Fly.io/etc.) with no docker-compose volume mount for data/. Mirrors
    streamlit/app_streamlit.py's _try_dvc_pull_from_secrets(), but reads
    DAGSHUB_USER/DAGSHUB_TOKEN from the environment instead of st.secrets
    (this service has no Streamlit-style secrets manager). Best-effort and
    silent on failure — the existing per-model "not found" warnings already
    cover that case. No-op when any model file already exists (the normal
    docker-compose case), so this never runs an unnecessary dvc pull on
    every /reload-artifacts call in local/dev use.
    """
    import subprocess

    if any((ARTIFACTS_DIR / f"model_{name}.pkl").exists() for name in ("gini", "mobility", "income_group")):
        return
    user = os.getenv("DAGSHUB_USER")
    token = os.getenv("DAGSHUB_TOKEN")
    if not user or not token:
        return

    try:
        # dvc pull needs a .git directory to satisfy its SCM layer (this
        # project's .dvc/config was created via a normal git-tracked `dvc
        # init`, not --no-scm) — errors "not a git repository" without one.
        # Most managed build platforms (Render confirmed; likely others)
        # deliberately exclude .git from what they send to the Docker
        # builder, so copying it in at build time isn't portable. A bare
        # `git init` at runtime satisfies DVC's check just as well — it only
        # needs a .git directory to exist, not any actual history — and
        # works identically everywhere regardless of build-context quirks.
        if not (ROOT / ".git").exists():
            subprocess.run(["git", "init"], cwd=str(ROOT), check=True, capture_output=True, timeout=30)

        subprocess.run(
            ["dvc", "remote", "modify", "origin", "--local", "access_key_id", token],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["dvc", "remote", "modify", "origin", "--local", "secret_access_key", token],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(["dvc", "pull"], cwd=str(ROOT), check=True, capture_output=True, timeout=120)
        logger.info("dvc pull bootstrap succeeded")
    except Exception:
        logger.warning("dvc pull bootstrap failed", exc_info=True)


def load_artifacts() -> None:
    global _models, _categorical_mappings, _feature_medians, _metrics
    _try_dvc_pull_from_env()
    _models = {}
    _metrics = {}
    for name in ("gini", "mobility", "income_group"):
        path = ARTIFACTS_DIR / f"model_{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _models[name] = pickle.load(f)
            logger.info("Loaded model_%s.pkl", name)
        else:
            logger.warning("model_%s.pkl not found at %s — train it first via train-api", name, path)

        metrics_path = ARTIFACTS_DIR / f"metrics_{name}.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                _metrics[name] = json.load(f)

    mapping_path = ARTIFACTS_DIR / "categorical_mappings.json"
    if mapping_path.exists():
        with open(mapping_path) as f:
            _categorical_mappings = json.load(f)
    else:
        logger.warning("categorical_mappings.json not found — categorical features will encode as -1 (unseen)")
        _categorical_mappings = {}

    medians_path = ARTIFACTS_DIR / "feature_medians.json"
    if medians_path.exists():
        with open(medians_path) as f:
            _feature_medians = json.load(f)
    else:
        logger.warning("feature_medians.json not found — a caller omitting a numeric field will fall back to 0.0")
        _feature_medians = {}

    build_explainers(_models)


@app.on_event("startup")
def startup_event():
    load_artifacts()


def _build_feature_row(payload: FeaturePayload) -> pd.DataFrame:
    row: dict[str, float | int] = {}
    for col in NUMERIC_FEATURES:
        value = getattr(payload, col, None)
        row[col] = value if value is not None else _feature_medians.get(col, 0.0)
    for col in CATEGORICAL_FEATURES:
        raw_value = getattr(payload, col, None) or "UNKNOWN"
        code = _categorical_mappings.get(col, {}).get(raw_value, -1)
        row[col + "_code"] = code
    return pd.DataFrame([row])


def _interval_80pct(name: str, pred: float) -> list[float] | None:
    """A fixed-width 80% prediction interval from the held-out test set's
    residual std at training time (metrics_{name}.json's "residual_std").
    Deliberately simple — one global band per target, not per-instance
    heteroscedastic uncertainty (that would need quantile regression or a
    conformal-prediction wrapper). None if the model predates this field."""
    residual_std = _metrics.get(name, {}).get("residual_std")
    if residual_std is None:
        return None
    half_width = 1.28 * residual_std  # ~80% two-tailed normal interval
    return [pred - half_width, pred + half_width]


@app.post("/predict-gini")
def predict_gini(payload: FeaturePayload):
    if "gini" not in _models:
        raise HTTPException(status_code=503, detail="model_gini.pkl not loaded — train it first")
    with PREDICTION_LATENCY.labels(target="gini").time():
        X = _build_feature_row(payload)
        pred = float(_models["gini"].predict(X)[0])
        record_prediction(X.iloc[0].to_dict())
    PREDICTION_COUNT.labels(target="gini").inc()
    return {"gini_index": pred, "interval_80pct": _interval_80pct("gini", pred)}


@app.post("/predict-mobility")
def predict_mobility(payload: FeaturePayload):
    if "mobility" not in _models:
        raise HTTPException(status_code=503, detail="model_mobility.pkl not loaded — train it first")
    with PREDICTION_LATENCY.labels(target="mobility").time():
        X = _build_feature_row(payload)
        pred = float(_models["mobility"].predict(X)[0])
        record_prediction(X.iloc[0].to_dict())
    PREDICTION_COUNT.labels(target="mobility").inc()
    return {"intergen_income_elasticity": pred, "interval_80pct": _interval_80pct("mobility", pred)}


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


@app.post("/explain-gini")
def explain_gini(payload: FeaturePayload):
    if "gini" not in _models:
        raise HTTPException(status_code=503, detail="model_gini.pkl not loaded — train it first")
    X = _build_feature_row(payload)
    contributions = explain("gini", X)
    if contributions is None:
        raise HTTPException(status_code=503, detail="No SHAP explainer available for gini")
    return {"contributions": contributions}


@app.post("/explain-mobility")
def explain_mobility(payload: FeaturePayload):
    if "mobility" not in _models:
        raise HTTPException(status_code=503, detail="model_mobility.pkl not loaded — train it first")
    X = _build_feature_row(payload)
    contributions = explain("mobility", X)
    if contributions is None:
        raise HTTPException(status_code=503, detail="No SHAP explainer available for mobility")
    return {"contributions": contributions}


@app.post("/explain-income-group")
def explain_income_group(payload: FeaturePayload):
    if "income_group" not in _models:
        raise HTTPException(status_code=503, detail="model_income_group.pkl not loaded — train it first")
    bundle = _models["income_group"]
    model, label_encoder = bundle["model"], bundle["label_encoder"]
    X = _build_feature_row(payload)
    proba = np.asarray(model.predict_proba(X)[0])
    pred_idx = int(proba.argmax())
    contributions = explain("income_group", X, class_index=pred_idx)
    if contributions is None:
        raise HTTPException(status_code=503, detail="No SHAP explainer available for income_group")
    return {"contributions": contributions, "explained_class": label_encoder.inverse_transform([pred_idx])[0]}


@app.get("/models/registry-status")
def models_registry_status():
    """Per-target MLflow registry stage/version, as recorded by train-api's
    register_and_promote() after each training run. Reads a plain JSON file
    instead of querying MLflow directly — keeps this service's dependency
    footprint (and its per-request latency/failure surface) unchanged."""
    path = ARTIFACTS_DIR / "registry_status.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


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

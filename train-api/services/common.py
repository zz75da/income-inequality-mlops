"""Shared helpers for the three train_*.py scripts (gini / mobility / income_group)."""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run `make ingest && make features` " "(or `dvc repro`) before training."
        )
    return pd.read_csv(FEATURES_PATH)


def feature_columns(params: dict) -> list[str]:
    feat_cfg = params["features"]
    cols = list(feat_cfg["numeric_features"])
    cols += [c + "_code" for c in feat_cfg["categorical_features"]]
    return cols


def group_train_test_split(df: pd.DataFrame, target: str, params: dict):
    """Split by country_code so no country appears in both train and test."""
    cols = feature_columns(params)
    df = df.dropna(subset=[target]).copy()
    X = df[cols]
    y = df[target]
    groups = df[params["split"]["group_column"]]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=params["split"]["test_size"],
        random_state=params["split"]["random_seed"],
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def save_model(model, name: str) -> Path:
    path = ARTIFACTS_DIR / f"model_{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logging.getLogger("train.common").info("Saved model -> %s", path)
    return path


def save_metrics(metrics: dict, name: str) -> Path:
    path = ARTIFACTS_DIR / f"metrics_{name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.getLogger("train.common").info("Saved metrics -> %s : %s", path, metrics)
    return path


def mlflow_setup():
    """Configure MLflow tracking against DagsHub if credentials are present, else local ./mlruns."""
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "income-inequality")
    mlflow.set_experiment(experiment)
    return mlflow


def register_and_promote(mlflow, name: str, model_uri: str, metrics: dict, params: dict) -> dict:
    """Register the just-logged model as a new MLflow registry version, and
    promote it to the "Production" stage only if its metric clears
    params.yaml's promotion_gates for this target — otherwise it stays in
    "Staging" (still logged and registered, just not live). Mirrors
    monitoring/alert-rules.yml's model-quality thresholds exactly; see that
    file's comment for why they're two hand-synced copies instead of one.

    Also records the outcome in data/artifacts/registry_status.json (touching
    only this target's key — safe because train-api/app.py runs gini/
    mobility/income_group as sequential subprocesses, never concurrently) so
    predict-api can surface registry stage/version without depending on
    mlflow itself.

    Takes `mlflow` as a parameter (the same instance mlflow_setup() returns)
    rather than importing it directly, and reaches MlflowClient via
    `mlflow.tracking.MlflowClient` off that same object — keeps this function
    testable with a plain fake object standing in for the whole mlflow
    package, no real mlflow install required just to test the gate logic.
    """
    logger = logging.getLogger("train.common")
    registered_name = f"income_inequality_{name}"

    try:
        model_version = mlflow.register_model(model_uri, registered_name)
    except Exception:
        logger.exception("Failed to register %s as a model version — skipping promotion", registered_name)
        return {"registered": False}

    gate = params.get("promotion_gates", {}).get(name)
    metric_value = metrics.get(gate["metric"]) if gate else None
    passed = gate is not None and metric_value is not None and metric_value >= gate["min"]
    stage = "Production" if passed else "Staging"

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=registered_name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=passed,
    )
    logger.info(
        "%s v%s -> %s (%s=%s, gate=%s)",
        registered_name,
        model_version.version,
        stage,
        gate["metric"] if gate else "?",
        metric_value,
        gate["min"] if gate else "none configured",
    )

    status = {
        "registered_name": registered_name,
        "version": model_version.version,
        "stage": stage,
        "metric": gate["metric"] if gate else None,
        "metric_value": metric_value,
        "gate_min": gate["min"] if gate else None,
        "passed": passed,
    }
    _update_registry_status(name, status)
    _push_training_metrics(name, metrics)
    return status


def _update_registry_status(name: str, status: dict) -> None:
    path = ARTIFACTS_DIR / "registry_status.json"
    all_status = {}
    if path.exists():
        try:
            all_status = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    all_status[name] = status
    path.write_text(json.dumps(all_status, indent=2))


def _push_training_metrics(name: str, metrics: dict) -> None:
    """Push this run's metrics to Prometheus Pushgateway so
    monitoring/alert-rules.yml's model-quality alerts (which read
    model_final_r2/model_final_accuracy) have real data — they're otherwise
    dead rules with nothing ever feeding them. Non-fatal: a training run
    shouldn't fail just because pushgateway is unreachable."""
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    pushgateway_url = os.getenv("PUSHGATEWAY_URL")
    if not pushgateway_url:
        return

    registry = CollectorRegistry()
    for metric_key in ("r2", "accuracy"):
        if metric_key not in metrics:
            continue
        gauge_name = f"model_final_{metric_key}"
        gauge = Gauge(gauge_name, f"Final {metric_key} of the last training run", ["target"], registry=registry)
        gauge.labels(target=name).set(metrics[metric_key])

    try:
        push_to_gateway(pushgateway_url, job=f"train_{name}", registry=registry)
    except Exception:
        logging.getLogger("train.common").warning("Failed to push metrics to %s", pushgateway_url, exc_info=True)

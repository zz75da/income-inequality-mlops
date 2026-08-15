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
            f"{FEATURES_PATH} not found. Run `make ingest && make features` "
            "(or `dvc repro`) before training."
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

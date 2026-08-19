"""Train the intergenerational rank-correlation mobility regressor (Random Forest).

GDIM's second mobility measure alongside intergen_income_elasticity (used by
train_mobility.py) — same source, same one-value-per-country granularity, same
regression setup. Kept as its own script/model rather than a second output
head on train_mobility.py's model, matching how gini/mobility/income_group are
each already independently trained, registered, and gated.
"""

from __future__ import annotations

import logging

import numpy as np
from common import (
    group_train_test_split,
    load_features,
    load_params,
    mlflow_setup,
    register_and_promote,
    save_metrics,
    save_model,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger("train.mobility_rank")
TARGET = "intergen_rank_correlation"
NAME = "mobility_rank"


def main() -> None:
    params = load_params()
    df = load_features()

    if df[TARGET].notna().sum() < 20:
        logger.warning(
            "Only %d non-null rows for %s — GDIM data likely hasn't been loaded "
            "(see ingestion/ingest_gdim.py). Training will proceed on whatever rows exist, "
            "but results won't be meaningful until GDIM is loaded.",
            df[TARGET].notna().sum(),
            TARGET,
        )

    X_train, X_test, y_train, y_test = group_train_test_split(df, TARGET, params)

    cfg = params["model_mobility_rank"]
    model = RandomForestRegressor(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        random_state=params["split"]["random_seed"],
        n_jobs=-1,
    )

    mlflow = mlflow_setup()
    with mlflow.start_run(run_name="train_mobility_rank"):
        mlflow.log_params({f"mobility_rank_{k}": v for k, v in cfg.items()})
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
            "residual_std": float(np.std(y_test - preds)),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
        model_info = mlflow.sklearn.log_model(model, artifact_path="model_mobility_rank")
        logger.info(
            "intergen_rank_correlation — MAE=%.3f R2=%.3f (train=%d test=%d)",
            metrics["mae"],
            metrics["r2"],
            metrics["n_train"],
            metrics["n_test"],
        )
        register_and_promote(mlflow, NAME, model_info.model_uri, metrics, params)

    save_model(model, NAME)
    save_metrics(metrics, NAME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()

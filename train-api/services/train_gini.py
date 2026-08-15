"""Train the Gini-index regressor (XGBoost) and log the run to MLflow."""
from __future__ import annotations

import logging

from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from common import group_train_test_split, load_features, load_params, mlflow_setup, save_metrics, save_model

logger = logging.getLogger("train.gini")
TARGET = "gini_index"
NAME = "gini"


def main() -> None:
    params = load_params()
    df = load_features()
    X_train, X_test, y_train, y_test = group_train_test_split(df, TARGET, params)

    cfg = params["model_gini"]
    model = XGBRegressor(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        reg_lambda=cfg["reg_lambda"],
        random_state=params["split"]["random_seed"],
        n_jobs=-1,
    )

    mlflow = mlflow_setup()
    with mlflow.start_run(run_name="train_gini"):
        mlflow.log_params({f"gini_{k}": v for k, v in cfg.items()})
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(model, artifact_path="model_gini")
        logger.info("gini_index — MAE=%.3f R2=%.3f (train=%d test=%d)", metrics["mae"], metrics["r2"], metrics["n_train"], metrics["n_test"])

    save_model(model, NAME)
    save_metrics(metrics, NAME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()

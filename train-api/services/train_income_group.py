"""
Train the income-group classifier (XGBoost) — predicts the World Bank
income-group bucket (Low / Lower-middle / Upper-middle / High income) from
macro features.

This substitutes for a genuine individual/household "income bracket" model:
that would require EU-SILC microdata, which is only available under a
restricted research-access agreement with Eurostat (see the note in
ingestion/ingest_eurostat.py), not via any of this project's public APIs. The
WB income-group label is the closest legitimately-obtainable "bracket"
classification target given fully public data sources.
"""
from __future__ import annotations

import logging

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from common import feature_columns, load_features, load_params, mlflow_setup, save_metrics, save_model
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger("train.income_group")
TARGET = "income_group"
NAME = "income_group"


def main() -> None:
    params = load_params()
    df = load_features().dropna(subset=[TARGET]).copy()

    label_encoder = LabelEncoder()
    df["target_encoded"] = label_encoder.fit_transform(df[TARGET])

    cols = feature_columns(params)
    X = df[cols]
    y = df["target_encoded"]
    groups = df[params["split"]["group_column"]]

    splitter = GroupShuffleSplit(n_splits=1, test_size=params["split"]["test_size"], random_state=params["split"]["random_seed"])
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    cfg = params["model_income_group"]
    model = XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        random_state=params["split"]["random_seed"],
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    mlflow = mlflow_setup()
    with mlflow.start_run(run_name="train_income_group"):
        mlflow.log_params({f"income_group_{k}": v for k, v in cfg.items()})
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "macro_f1": float(f1_score(y_test, preds, average="macro")),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "classes": label_encoder.classes_.tolist(),
        }
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(model, artifact_path="model_income_group")
        logger.info("income_group — acc=%.3f macro_f1=%.3f\n%s", metrics["accuracy"], metrics["macro_f1"],
                     classification_report(y_test, preds, target_names=label_encoder.classes_))

    save_model({"model": model, "label_encoder": label_encoder}, NAME)
    save_metrics(metrics, NAME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()

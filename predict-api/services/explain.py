"""
SHAP TreeExplainer wrapper for predict-api's 3 tree-based models
(XGBRegressor, RandomForestRegressor, XGBClassifier).

TreeExplainer computes exact Shapley values directly from the tree
structure — no background dataset needed (unlike KernelExplainer), and fast
enough to run per-request rather than needing to be precomputed.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("predict-api.explain")

_explainers: dict = {}


def build_explainers(models: dict) -> None:
    """Build one TreeExplainer per loaded model — called once from
    load_artifacts() after models are (re)loaded, not per-request."""
    import shap

    global _explainers
    _explainers = {}
    for name, model in models.items():
        estimator = model["model"] if isinstance(model, dict) else model
        try:
            _explainers[name] = shap.TreeExplainer(estimator)
        except Exception:
            logger.warning("Could not build a SHAP explainer for %s", name, exc_info=True)


def explain(name: str, x_row: pd.DataFrame, top_n: int = 5, class_index: int | None = None) -> dict[str, float] | None:
    """Top-N feature contributions for a single-row prediction, largest
    absolute contribution first. None if no explainer is available (model
    not loaded, or SHAP couldn't build one for it).

    `class_index` selects which class's contributions to return for a
    classifier (SHAP gives one contribution set per class); ignored for
    regressors. Callers should pass the model's own predicted class so the
    explanation matches what the user was actually shown.
    """
    explainer = _explainers.get(name)
    if explainer is None:
        return None

    raw = explainer.shap_values(x_row)
    if isinstance(raw, list):
        # Older SHAP API for multi-class: one (n_samples, n_features) array per class.
        row_values = raw[class_index or 0][0]
    elif raw.ndim == 3:
        # Newer SHAP API for multi-class: (n_samples, n_features, n_classes).
        row_values = raw[0, :, class_index or 0]
    else:
        # Regression: (n_samples, n_features).
        row_values = raw[0]

    contributions = dict(zip(x_row.columns, (float(v) for v in row_values), strict=False))
    ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return dict(ranked[:top_n])

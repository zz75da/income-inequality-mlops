"""
Streamlit UI for the Income Inequality MLOps project.

Five pages:
  - Explore: choropleth of the latest Gini index per country + per-country
    time series, sourced straight from data/processed/features.csv (no API
    call needed — this is descriptive data, not a model prediction).
  - Predict: form -> calls predict-api's /predict-gini, /predict-mobility,
    /predict-income-group with the entered macro features, showing formatted
    results with confidence intervals and a SHAP "why this prediction" panel.
  - Model Performance: reads data/artifacts/metrics_*.json + params.yaml +
    registry_status.json directly (no predict-api call) — stays functional
    even if predict-api is down.
  - About: architecture, data sources, and the real debugging stories behind
    this project's known limitations.
  - Drift Reports: lists and embeds the Evidently HTML reports predict-api
    has generated under data/artifacts/drift_reports/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import yaml

import streamlit as st

PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://localhost:5003")
GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL", "https://github.com/zz75da/income-inequality-mlops")
DAGSHUB_URL = os.getenv("DAGSHUB_URL", "https://dagshub.com/zz75da/income-inequality-mlops")
DAGSHUB_MLFLOW_URL = os.getenv("DAGSHUB_MLFLOW_URL", f"{DAGSHUB_URL}.mlflow/#/experiments")

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"
DRIFT_REPORTS_DIR = ARTIFACTS_DIR / "drift_reports"
PARAMS_PATH = ROOT / "params.yaml"

# --- Color choices (see dataviz skill: sequential for magnitude, diverging
# for polarity, fixed categorical order, status via icon+label not color
# alone) ---
SEQUENTIAL_SCALE = "Blues"  # magnitude (Gini, income shares) — one hue, light->dark
CATEGORICAL_COLORS = [  # fixed order for country time series, not auto-cycled
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#9D755D",
]
SHAP_POSITIVE = "#E45756"  # pushes the prediction up
SHAP_NEGATIVE = "#4C78A8"  # pushes the prediction down

TARGETS = {
    "gini": {"label": "Gini index", "metric_label": "R²", "metric_key": "r2"},
    "mobility": {"label": "Mobility (intergen. elasticity)", "metric_label": "R²", "metric_key": "r2"},
    "income_group": {"label": "Income group", "metric_label": "Accuracy", "metric_key": "accuracy"},
}

st.set_page_config(page_title="Income Inequality MLOps", layout="wide", page_icon="📊")


@st.cache_data(ttl=300)
def load_features() -> pd.DataFrame | None:
    if not FEATURES_PATH.exists():
        return None
    return pd.read_csv(FEATURES_PATH)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def load_params() -> dict | None:
    if not PARAMS_PATH.exists():
        return None
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def predict_api_reachable() -> bool:
    """Short-timeout health probe so the Predict page can degrade gracefully
    instead of every button click raising/timing out. Explore/Performance/
    About never call predict-api at all, so they're unaffected either way."""
    try:
        resp = requests.get(f"{PREDICT_API_URL}/health", timeout=2)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def page_explore(df: pd.DataFrame) -> None:
    st.header("Global income inequality — explore")

    latest = df.sort_values("year").groupby("country_code", as_index=False).last()
    metric = st.selectbox(
        "Metric",
        [
            c
            for c in ["gini_index", "top10_income_share", "bottom50_income_share", "intergen_income_elasticity"]
            if c in latest.columns
        ],
    )
    fig = px.choropleth(
        latest,
        locations="country_code",
        color=metric,
        hover_name="country_name",
        color_continuous_scale=SEQUENTIAL_SCALE,
        projection="natural earth",
        title=f"Latest {metric} by country",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Country time series")
    countries = sorted(df["country_name"].dropna().unique())
    selected = st.multiselect(
        "Countries (up to 8 — more than that gets hard to read)",
        countries,
        default=countries[:3] if countries else [],
        max_selections=8,
    )
    if selected:
        subset = df[df["country_name"].isin(selected)]
        fig2 = px.line(
            subset,
            x="year",
            y=metric,
            color="country_name",
            markers=True,
            color_discrete_sequence=CATEGORICAL_COLORS,
        )
        fig2.update_traces(line_width=2, marker_size=6)
        st.plotly_chart(fig2, use_container_width=True)


def _shap_bar_chart(contributions: dict[str, float], title: str) -> go.Figure:
    items = sorted(contributions.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [SHAP_POSITIVE if v >= 0 else SHAP_NEGATIVE for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Contribution to prediction",
        margin=dict(l=0, r=0, t=40, b=0),
        height=300,
    )
    return fig


def page_predict() -> None:
    st.header("Predict")

    if not predict_api_reachable():
        st.warning(
            f"predict-api isn't reachable at {PREDICT_API_URL} — the other pages (Explore, "
            "Model Performance, About) don't need it and still work, but predictions need the "
            "service running. See the README's Quick Start."
        )
        return

    st.caption(f"Calling predict-api at {PREDICT_API_URL}")

    col1, col2 = st.columns(2)
    with col1:
        gdp_per_capita_ppp = st.number_input("GDP per capita (PPP, current intl $)", value=30000.0)
        gdp_growth_pct = st.number_input("GDP growth (%)", value=2.0)
        unemployment_rate = st.number_input("Unemployment rate (%)", value=6.0)
        education_expenditure_pct_gdp = st.number_input("Education spending (% GDP)", value=4.5)
        social_spending_pct_gdp = st.number_input("Social spending (% GDP)", value=15.0)
    with col2:
        tax_revenue_pct_gdp = st.number_input("Tax revenue (% GDP)", value=20.0)
        urban_population_pct = st.number_input("Urban population (%)", value=70.0)
        population_total = st.number_input("Population", value=10_000_000.0)
        top10_income_share = st.number_input("Top 10% income share", value=0.30)
        bottom50_income_share = st.number_input("Bottom 50% income share", value=0.15)

    region = st.text_input("Region", value="Europe & Central Asia")
    income_group_lag1 = st.selectbox(
        "Previous income group", ["Low income", "Lower middle income", "Upper middle income", "High income", "UNKNOWN"]
    )

    payload = dict(
        gdp_per_capita_ppp=gdp_per_capita_ppp,
        gdp_growth_pct=gdp_growth_pct,
        unemployment_rate=unemployment_rate,
        education_expenditure_pct_gdp=education_expenditure_pct_gdp,
        social_spending_pct_gdp=social_spending_pct_gdp,
        tax_revenue_pct_gdp=tax_revenue_pct_gdp,
        urban_population_pct=urban_population_pct,
        population_total=population_total,
        top10_income_share=top10_income_share,
        bottom50_income_share=bottom50_income_share,
        region=region,
        income_group_lag1=income_group_lag1,
    )

    if not st.button("Predict all 3 targets", type="primary"):
        return

    targets = [
        ("gini", "Gini index", "/predict-gini", "/explain-gini", "gini_index"),
        (
            "mobility",
            "Intergenerational mobility",
            "/predict-mobility",
            "/explain-mobility",
            "intergen_income_elasticity",
        ),
        ("income_group", "Income group", "/predict-income-group", "/explain-income-group", None),
    ]

    cols = st.columns(3)
    for col, (_name, label, predict_ep, explain_ep, value_key) in zip(cols, targets, strict=False):
        with col:
            try:
                resp = requests.post(f"{PREDICT_API_URL}{predict_ep}", json=payload, timeout=15)
                resp.raise_for_status()
                body = resp.json()
            except requests.RequestException as exc:
                st.error(f"{label} request failed: {exc}")
                continue

            if value_key:
                pred = body[value_key]
                interval = body.get("interval_80pct")
                delta = None
                if interval:
                    delta = f"80% interval: {interval[0]:.2f} – {interval[1]:.2f}"
                st.metric(label, f"{pred:.2f}", delta=delta, delta_color="off")
            else:
                st.metric(label, body["income_group"])
                proba_df = pd.DataFrame(
                    {"class": list(body["probabilities"].keys()), "probability": list(body["probabilities"].values())}
                ).sort_values("probability", ascending=False)
                st.dataframe(proba_df, hide_index=True, use_container_width=True)

            with st.expander("Why this prediction?"):
                try:
                    exp_resp = requests.post(f"{PREDICT_API_URL}{explain_ep}", json=payload, timeout=15)
                    exp_resp.raise_for_status()
                    exp_body = exp_resp.json()
                    fig = _shap_bar_chart(exp_body["contributions"], f"Top feature contributions — {label}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "SHAP values: how much each feature pushed this specific prediction up or down "
                        "from the model's average prediction."
                    )
                except requests.RequestException as exc:
                    st.info(f"Explanation unavailable: {exc}")


def _gate_status_badge(passed: bool) -> str:
    # Status conveyed by icon + text, never color alone.
    return "✅ Production" if passed else "🟡 Staging"


def page_performance() -> None:
    st.header("Model Performance")
    st.caption("Reads local training artifacts directly — works even if predict-api is down.")

    params = load_params()
    gates = params.get("promotion_gates", {}) if params else {}

    for name, cfg in TARGETS.items():
        metrics = load_json(ARTIFACTS_DIR / f"metrics_{name}.json")
        registry = load_json(ARTIFACTS_DIR / "registry_status.json") or {}
        target_registry = registry.get(name)

        st.subheader(cfg["label"])
        if metrics is None:
            st.info(f"model_{name} hasn't been trained yet.")
            continue

        cols = st.columns(4)
        cols[0].metric(cfg["metric_label"], f"{metrics.get(cfg['metric_key'], float('nan')):.3f}")
        if "mae" in metrics:
            cols[1].metric("MAE", f"{metrics['mae']:.3f}")
        elif "macro_f1" in metrics:
            cols[1].metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        cols[2].metric("Train / test rows", f"{metrics.get('n_train', '?')} / {metrics.get('n_test', '?')}")

        if target_registry:
            cols[3].markdown(f"**Registry:** {_gate_status_badge(target_registry['passed'])}")
            gate = gates.get(name)
            if gate:
                st.caption(
                    f"Promotion gate: {gate['metric']} ≥ {gate['min']} — "
                    f"actual {target_registry['metric_value']:.3f} "
                    f"({'clears' if target_registry['passed'] else 'below'} the bar). "
                    f"Registered as `{target_registry['registered_name']}` v{target_registry['version']}."
                )
        else:
            cols[3].markdown("**Registry:** not registered yet")

        if params:
            model_cfg = params.get(f"model_{name}", {})
            if model_cfg:
                with st.expander("Hyperparameters"):
                    st.json(model_cfg)

        st.divider()


def page_about() -> None:
    st.header("About this project")

    st.markdown(
        """
Predicts three country-level income-inequality indicators — the Gini index, intergenerational
mobility, and World Bank income-group bracket — from public macroeconomic data, built as a
complete MLOps pipeline: ingestion, DVC-versioned data, scheduled retraining, MLflow experiment
tracking with gated model promotion, drift monitoring, and this Streamlit UI.

**Data sources:** World Bank Open Data, OECD Income Distribution Database, Eurostat EU-SILC
aggregates, and the World Bank's GDIM mobility database — all public, unauthenticated APIs.
"""
    )

    st.subheader("Architecture")
    st.markdown(
        """
```mermaid
flowchart LR
    subgraph ingest["Ingestion (Airflow, @monthly)"]
        WB[World Bank] --> M[merge_sources.py]
        OECD --> M
        EU[Eurostat] --> M
        GDIM --> M
        M --> F[build_features.py]
    end
    F --> T1[train_gini.py]
    F --> T2[train_mobility.py]
    F --> T3[train_income_group.py]
    T1 & T2 & T3 --> MLF[(MLflow Registry\\non DagsHub)]
    MLF --> P[predict-api]
    P --> S[Streamlit]
    P -.drift buffer.-> P
```
"""
    )

    st.subheader("Challenges solved")
    st.markdown(
        """
Real debugging stories from building this, not tutorial-perfect assumptions:

- **OECD's SDMX-JSON schema drift.** The code assumed `data.structure`; the live API actually
  returns `data.structures[0]`. Caught by finally running ingestion against the real endpoint.
- **WID.world's public API was retired.** Confirmed via live testing (404, no documented public
  replacement) — the site now uses a private, key-gated endpoint. Handled as a best-effort
  source rather than a hard failure.
- **A broken `dvc.yaml`** used `required: false`, which isn't valid DVC syntax — it silently
  blocked every `dvc` command (`add`, `repro`, `push`) until this was caught.
- **GDIM's real schema was completely different** from the originally-assumed manual-spreadsheet
  approach — found the actual public World Bank Data Catalog CSV and its real columns
  (`BETA`/`COR`, not `icm_transmission`/`icm_rank`), unblocking the whole mobility target.
- **Eurostat's country codes weren't ISO3** (`EL` for Greece, `UK` for the UK, plus aggregates
  like `EU27_2020`) — they were silently failing to join against the other three sources. A new
  pandera schema-validation step caught 976 orphaned rows; fixing the join lifted gini's R² from
  0.31 to 0.60.

See the README's Known Limitations for the ones still open.
"""
    )

    st.subheader("Links")
    st.markdown(
        f"- [GitHub repository]({GITHUB_REPO_URL})\n"
        f"- [DagsHub project]({DAGSHUB_URL})\n"
        f"- [MLflow experiments]({DAGSHUB_MLFLOW_URL})"
    )


def page_drift() -> None:
    st.header("Drift reports")
    if not DRIFT_REPORTS_DIR.exists():
        st.info("No drift reports yet — predict-api generates one once its prediction buffer fills up.")
        return
    reports = sorted(DRIFT_REPORTS_DIR.glob("drift_report_*.html"), reverse=True)
    if not reports:
        st.info("No drift reports yet.")
        return
    selected = st.selectbox("Report", [r.name for r in reports])
    report_path = DRIFT_REPORTS_DIR / selected
    st.components.v1.html(report_path.read_text(), height=800, scrolling=True)


def render_sidebar() -> None:
    st.sidebar.title("Income Inequality MLOps")
    st.sidebar.markdown(f"[GitHub]({GITHUB_REPO_URL}) · [DagsHub]({DAGSHUB_URL}) · [MLflow]({DAGSHUB_MLFLOW_URL})")


def main() -> None:
    render_sidebar()
    page = st.sidebar.radio("Page", ["Explore", "Predict", "Model Performance", "About", "Drift Reports"])

    if page == "Explore":
        df = load_features()
        if df is None:
            st.warning(
                "data/processed/features.csv not found — run the ingestion + feature pipeline first "
                "(`make ingest && make features`)."
            )
        else:
            page_explore(df)
    elif page == "Predict":
        page_predict()
    elif page == "Model Performance":
        page_performance()
    elif page == "About":
        page_about()
    else:
        page_drift()


if __name__ == "__main__":
    main()

"""
Streamlit UI for the Income Inequality MLOps project.

Five pages (About first — landing page for a first-time visitor):
  - About: architecture, data sources, and the real debugging stories behind
    this project's known limitations.
  - Explore: choropleth of the latest Gini index per country + per-country
    time series, sourced straight from data/processed/features.csv (no API
    call needed — this is descriptive data, not a model prediction).
  - Predict: form -> calls predict-api's /predict-gini, /predict-mobility,
    /predict-income-group with the entered macro features, showing formatted
    results with confidence intervals and a SHAP "why this prediction" panel.
  - Model Performance: reads data/artifacts/metrics_*.json + params.yaml +
    registry_status.json directly (no predict-api call) — stays functional
    even if predict-api is down.
  - Drift Reports: lists and embeds predict-api's Evidently HTML reports via
    GET /drift-reports and /drift-reports/{name} — fetched over HTTP, not
    read off local disk, since predict-api and this app run on separate
    hosts (e.g. Render / Streamlit Cloud) with no shared filesystem.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import yaml

import streamlit as st
import streamlit.components.v1 as components

# Must be the first Streamlit command in the script — before even touching
# st.secrets in _config() below, which otherwise trips "set_page_config()
# must be called as the first Streamlit command" (a real crash this caused
# once _config() started reading st.secrets ahead of it).
st.set_page_config(page_title="Income Inequality MLOps", layout="wide", page_icon="📊")


def _config(key: str, default: str) -> str:
    """Read a config value from Streamlit Cloud's secrets manager first (its
    Secrets UI populates st.secrets, not necessarily plain OS environment
    variables), falling back to a real env var (docker-compose/local run),
    then the given default."""
    try:
        value = st.secrets.get(key)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(key, default)


PREDICT_API_URL = _config("PREDICT_API_URL", "http://localhost:5003")
GITHUB_REPO_URL = _config("GITHUB_REPO_URL", "https://github.com/zz75da/income-inequality-mlops")
DAGSHUB_URL = _config("DAGSHUB_URL", "https://dagshub.com/zz75da/income-inequality-mlops")
DAGSHUB_MLFLOW_URL = _config("DAGSHUB_MLFLOW_URL", f"{DAGSHUB_URL}.mlflow/#/experiments")

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
ARTIFACTS_DIR = ROOT / "data" / "artifacts"
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


def _try_dvc_pull_from_secrets() -> None:
    """Cold-start bootstrap for a hosted deploy (e.g. Streamlit Community
    Cloud): data/ is DVC-tracked, not committed, so a fresh checkout has no
    features.csv. If DagsHub credentials are configured in the platform's
    secrets manager, attempt a `dvc pull` before giving up — best-effort,
    silent on failure (the caller's existing "run the pipeline first"
    warning covers that case either way). No-op locally/in Docker, where
    features.csv already exists and this is never reached.
    """
    import subprocess

    try:
        user = st.secrets.get("DAGSHUB_USER")
        token = st.secrets.get("DAGSHUB_TOKEN")
    except Exception:
        return  # no st.secrets configured (e.g. local run without secrets.toml)
    if not user or not token:
        return

    try:
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
    except Exception:
        logging.getLogger("streamlit.bootstrap").warning("dvc pull bootstrap failed", exc_info=True)


@st.cache_data(ttl=300)
def load_features() -> pd.DataFrame | None:
    if not FEATURES_PATH.exists():
        _try_dvc_pull_from_secrets()
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


@st.cache_data(ttl=300)
def load_categorical_mappings() -> dict:
    """category -> code maps written by build_features.py, keyed by column
    name (e.g. "region", "income_group_lag1"). Predict page dropdowns are
    sourced from this so a selected label always matches a code the model
    actually saw in training — free text let users send an unseen category
    (e.g. missing the trailing space in the World Bank's real region labels)
    that predict-api silently encodes as -1."""
    return load_json(ARTIFACTS_DIR / "categorical_mappings.json") or {}


def render_mermaid(diagram: str, height: int = 320) -> None:
    """st.markdown() only puts a ```mermaid fence into a plain code block —
    unlike GitHub, Streamlit has no built-in Mermaid renderer. Render it
    client-side via mermaid.js from a CDN instead (Streamlit Cloud has
    outbound internet access, so this works there the same as locally)."""
    components.html(
        f"""
        <div class="mermaid">{diagram}</div>
        <script type="module">
            import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
            mermaid.initialize({{ startOnLoad: true, theme: "neutral" }});
        </script>
        """,
        height=height,
        scrolling=True,
    )


def predict_api_reachable() -> bool:
    """Health probe so the Predict page can degrade gracefully instead of
    every button click raising/timing out. Explore/Performance/About never
    call predict-api at all, so they're unaffected either way.

    Two-stage: a quick 2s check covers the common case (already warm). A
    free-tier host like Render's spins the service down after ~15min idle
    and can take 30-60s to cold-start the next request, so a single short
    timeout would near-permanently report "unreachable" on an idle demo —
    fall back to a longer wait, with a spinner explaining why, before
    giving up for real."""
    try:
        resp = requests.get(f"{PREDICT_API_URL}/health", timeout=2)
        if resp.status_code == 200:
            return True
    except requests.RequestException:
        pass

    with st.spinner("Waking up predict-api (free-tier host cold start can take up to a minute)..."):
        try:
            resp = requests.get(f"{PREDICT_API_URL}/health", timeout=75)
            return resp.status_code == 200
        except requests.RequestException:
            return False


def page_explore(df: pd.DataFrame) -> None:
    st.header("Global income inequality — explore")

    latest = df.sort_values("year").groupby("country_code", as_index=False).last()
    candidate_metrics = ["gini_index", "top10_income_share", "bottom50_income_share", "intergen_income_elasticity"]
    # A column existing isn't enough — WID.world's public API being retired
    # (see About page) leaves top10/bottom50_income_share present but 100%
    # null, which renders as a blank, unexplained map with no color at all.
    available_metrics = [c for c in candidate_metrics if c in latest.columns and latest[c].notna().any()]
    unavailable_metrics = [c for c in candidate_metrics if c in latest.columns and c not in available_metrics]
    metric = st.selectbox("Metric", available_metrics)
    if unavailable_metrics:
        st.caption(
            f"Not shown (no data currently available): {', '.join(unavailable_metrics)} — "
            "sourced from WID.world, whose public API was retired. See the About page's "
            "Challenges Solved section."
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
    default_countries = [
        c
        for c in ["United States", "China", "Japan", "Brazil", "Germany", "United Kingdom", "France"]
        if c in countries
    ]
    selected = st.multiselect(
        "Countries (up to 8 — more than that gets hard to read)",
        countries,
        default=default_countries,
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
            "service running. If this is the hosted demo, the free-tier host may still be "
            "cold-starting after this waited up to 75s — try reloading the page in a moment. "
            "Otherwise see the README's Quick Start."
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

    mappings = load_categorical_mappings()
    region_options = sorted(mappings.get("region", {}).keys()) or [
        "East Asia & Pacific",
        "Europe & Central Asia",
        "Latin America & Caribbean",
        "Middle East, North Africa, Afghanistan & Pakistan",
        "North America",
        "South Asia",
        "Sub-Saharan Africa",
    ]
    income_group_options = sorted(mappings.get("income_group_lag1", {}).keys()) or [
        "Low income",
        "Lower middle income",
        "Upper middle income",
        "High income",
        "UNKNOWN",
    ]
    default_region_idx = (
        region_options.index("Europe & Central Asia") if "Europe & Central Asia" in region_options else 0
    )
    region = st.selectbox("Region", region_options, index=default_region_idx)
    income_group_lag1 = st.selectbox("Previous income group", income_group_options)

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

    st.subheader("Understanding the predictions")
    st.markdown(
        """
Each target is a **country-year** statistic — an aggregate estimate for a whole country in a given
year, not a prediction about any individual person. Think of it the way a national unemployment
rate or GDP figure works: informative about the population, silent about any one household.
"""
    )

    with st.expander("Gini index — what it measures and how good the model is"):
        st.markdown(
            """
**What it measures:** income inequality within a country, on a 0-100 scale. 0 would mean everyone
has exactly the same income; 100 would mean one person has all of it. Real countries mostly fall
between roughly 25 (e.g. several Nordic countries) and 63 (some of the most unequal economies) —
so a few points of difference is meaningful, not noise.

**Model quality here — R² 0.565, MAE 5.34:**
- **R² (coefficient of determination)** is the share of country-to-country variation in Gini the
  model explains from macro features (GDP per capita, unemployment, urbanization, tax/social
  spending, etc.) — 1.0 would be a perfect fit, 0 would mean "no better than always guessing the
  average." **0.565 means the model explains about 56% of why some countries are more unequal than
  others.** The other 44% comes from things this dataset doesn't capture — labor law, union
  density, informal-sector size, historical land distribution — so treat predictions as a
  macro-driven estimate, not the full picture.
- **MAE (mean absolute error)** is the average miss size, in actual Gini points: **a typical
  prediction is off by about 5.3 points.** For context, that's roughly the gap between France and
  the UK's Gini index — real, but not the difference between an equal and an unequal society.
"""
        )

    with st.expander("Mobility (intergenerational income elasticity) — what it measures and how good the model is"):
        st.markdown(
            """
**What it measures:** how strongly a child's income is tied to their parents' income, on roughly a
0-1 scale. **Lower = more mobility** (your income is less determined by your parents' — e.g.
Nordic countries cluster near 0.15-0.2). **Higher = less mobility** (income is more inherited —
several countries with high inequality sit above 0.4-0.5). This is the least intuitive of the
three targets and the one where "good" performance looks different from a typical regression.

**Model quality here — R² 0.350, MAE 0.102:**
- The elasticity itself only spans a narrow real-world range (roughly 0.1-0.6), so even a model
  that's doing genuinely useful work will show a lower R² than Gini's — there's simply less
  variance available to explain, and this metric is sparser and noisier in the source data (GDIM)
  to begin with. **0.35 in this context is a meaningfully-better-than-average fit, not a weak one**
  — read it relative to this target's ceiling, not against Gini's 0.565.
- **MAE 0.102** means a typical prediction is off by about 0.1 elasticity points — on a 0.1-0.6
  scale, enough to blur adjacent countries but still usually correct about which broad tier
  (low/medium/high mobility) a country falls into.
"""
        )

    with st.expander("Income group — what it measures and how good the model is"):
        st.markdown(
            """
**What it measures:** a classification, not a number — which World Bank income bracket a country
falls into (Low / Lower-middle / Upper-middle / High income), based on gross national income per
capita. Unlike Gini or mobility, there's no partial credit for "close": the model is either right
or wrong about the bracket.

**Model quality here — Accuracy 0.992, Macro F1 0.991:**
- **Accuracy** is simply the fraction of correct predictions — 99.2% here. That number alone can be
  misleading on an imbalanced dataset (if most rows were "High income," always guessing "High
  income" would already score well without learning anything).
- **Macro F1** guards against exactly that: it averages the F1 score (precision/recall balance)
  **per class, unweighted**, so a model that ignores a rare class gets penalized even if overall
  accuracy stays high. **0.991 essentially matching accuracy means the model isn't leaning on the
  common classes — it's genuinely distinguishing all four brackets,** not just the easy ones. This
  is the most reliable of the three targets, which makes sense: GNI-per-capita brackets are a
  close functional match for the GDP-per-capita feature already in the training data.
"""
        )

    st.markdown(
        f"""
**Train/test rows and Registry status** (shown per-target on the Model Performance page): the
train/test split is how "unseen data" performance above was measured — the model never saw the
test rows during training, so those numbers estimate how it'd do on a genuinely new country-year.
Registry status reflects a promotion gate (see the README's
[Design Scope]({GITHUB_REPO_URL}#design-scope) section) — a model only reaches `Production` in the
MLflow registry if its metric clears a minimum bar; falling short routes it to `Staging` instead of
silently shipping a worse model.
"""
    )

    st.subheader("Architecture")
    render_mermaid(
        """
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

    st.subheader("About the author")
    st.markdown(
        """
Built end-to-end — ingestion through deployment — as a portfolio piece demonstrating full-lifecycle
MLOps: reproducible data/model versioning, gated experiment tracking, monitored serving, and a
usable frontend, not just a notebook with a good R². Reach out on LinkedIn if you'd like to talk
about it.
"""
    )

    st.subheader("Links")
    st.markdown(
        f"- [LinkedIn](https://www.linkedin.com/in/zzeghoud)\n"
        f"- [GitHub repository]({GITHUB_REPO_URL})\n"
        f"- [DagsHub project]({DAGSHUB_URL})\n"
        f"- [MLflow experiments]({DAGSHUB_MLFLOW_URL})"
    )


def page_drift() -> None:
    st.header("Drift reports")
    st.caption(
        "Served over HTTP from predict-api's GET /drift-reports, not read off local disk — "
        "predict-api and this app run on separate hosts with no shared filesystem, so a report "
        "generated by predict-api is only ever visible here through its API."
    )
    if not predict_api_reachable():
        st.warning(
            f"predict-api isn't reachable at {PREDICT_API_URL} — drift reports are generated and "
            "served by predict-api, so this page needs it running. See the README's Quick Start."
        )
        return

    try:
        resp = requests.get(f"{PREDICT_API_URL}/drift-reports", timeout=10)
        resp.raise_for_status()
        reports = resp.json().get("reports", [])
    except requests.RequestException as e:
        st.error(f"Failed to list drift reports from predict-api: {e}")
        return

    if not reports:
        st.info("No drift reports yet — predict-api generates one once its prediction buffer fills up.")
        return

    selected = st.selectbox("Report", reports)
    try:
        report_resp = requests.get(f"{PREDICT_API_URL}/drift-reports/{selected}", timeout=10)
        report_resp.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Failed to fetch drift report from predict-api: {e}")
        return
    st.components.v1.html(report_resp.text, height=800, scrolling=True)


def render_sidebar() -> None:
    st.sidebar.title("Income Inequality MLOps")
    st.sidebar.markdown(
        f"[GitHub]({GITHUB_REPO_URL}) · [DagsHub]({DAGSHUB_URL}) · [MLflow]({DAGSHUB_MLFLOW_URL}) · "
        "[LinkedIn](https://www.linkedin.com/in/zzeghoud)"
    )


def main() -> None:
    render_sidebar()
    page = st.sidebar.radio("Page", ["About", "Explore", "Predict", "Model Performance", "Drift Reports"])

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

"""
Streamlit UI for the Income Inequality MLOps project.

Three pages:
  - Explore: choropleth of the latest Gini index per country + per-country
    time series, sourced straight from data/processed/features.csv (no API
    call needed — this is descriptive data, not a model prediction).
  - Predict: form -> calls predict-api's /predict-gini, /predict-mobility,
    /predict-income-group with the entered macro features.
  - Drift Reports: lists and embeds the Evidently HTML reports predict-api
    has generated under data/artifacts/drift_reports/.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests

import streamlit as st

PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://localhost:5003")
ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
DRIFT_REPORTS_DIR = ROOT / "data" / "artifacts" / "drift_reports"

st.set_page_config(page_title="Income Inequality MLOps", layout="wide")


@st.cache_data(ttl=300)
def load_features() -> pd.DataFrame | None:
    if not FEATURES_PATH.exists():
        return None
    return pd.read_csv(FEATURES_PATH)


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
        color_continuous_scale="RdYlGn_r",
        projection="natural earth",
        title=f"Latest {metric} by country",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Country time series")
    countries = sorted(df["country_name"].dropna().unique())
    selected = st.multiselect("Countries", countries, default=countries[:3] if countries else [])
    if selected:
        subset = df[df["country_name"].isin(selected)]
        fig2 = px.line(subset, x="year", y=metric, color="country_name", markers=True)
        st.plotly_chart(fig2, use_container_width=True)


def page_predict() -> None:
    st.header("Predict")
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

    if st.button("Predict all 3 targets"):
        for label, endpoint in [
            ("Gini index", "/predict-gini"),
            ("Intergenerational income elasticity", "/predict-mobility"),
            ("Income group", "/predict-income-group"),
        ]:
            try:
                resp = requests.post(f"{PREDICT_API_URL}{endpoint}", json=payload, timeout=15)
                resp.raise_for_status()
                st.success(f"{label}: {resp.json()}")
            except requests.RequestException as exc:
                st.error(f"{label} request failed: {exc}")


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


def main() -> None:
    st.sidebar.title("Income Inequality MLOps")
    page = st.sidebar.radio("Page", ["Explore", "Predict", "Drift Reports"])

    if page == "Explore":
        df = load_features()
        if df is None:
            st.warning(
                "data/processed/features.csv not found — run the ingestion + feature pipeline first (`make ingest && make features`)."
            )
        else:
            page_explore(df)
    elif page == "Predict":
        page_predict()
    else:
        page_drift()


if __name__ == "__main__":
    main()

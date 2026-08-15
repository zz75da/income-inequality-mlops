# Income Inequality MLOps Platform

[![CI — Tests & DVC Sync](https://github.com/zz75da/income-inequality-mlops/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/zz75da/income-inequality-mlops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** [zz75da](https://github.com/zz75da)

End-to-end MLOps platform predicting country-level income inequality from public macroeconomic data.
FastAPI microservices, Airflow orchestration, DVC/DagsHub data+model versioning, MLflow experiment
tracking, and Prometheus/Grafana monitoring — scoped for a tabular, low-QPS, periodically-refreshed
dataset rather than a high-throughput production simulation.

## Table of Contents

- [Prediction Targets](#prediction-targets)
- [Data Sources](#data-sources)
- [Architecture](#architecture)
- [Design Scope](#design-scope)
- [Quick Start](#quick-start)
- [Service Endpoints](#service-endpoints)
- [Repository Structure](#repository-structure)
- [Environment Variables](#environment-variables)
- [Known Limitations](#known-limitations)

---

## Prediction Targets

Three models, trained from the same country-year feature panel:

| Target | Type | Model | Column |
|---|---|---|---|
| Gini index | Regression | XGBoost | `gini_index` |
| Intergenerational income mobility | Regression | Random Forest | `intergen_income_elasticity` |
| Income-group bracket | Classification (4 classes) | XGBoost | `income_group` |

The third target substitutes the World Bank's Low/Lower-middle/Upper-middle/High income
classification for a genuine household income-bracket model — see
[Known Limitations](#known-limitations) for why.

## Data Sources

All four ingestion scripts hit public, unauthenticated REST/bulk-download endpoints — no
paid API keys required (WID.world has an optional free key that only raises rate limits):

| Source | What it provides | Access |
|---|---|---|
| [World Bank Open Data](https://api.worldbank.org/v2/) | Gini index, GDP, unemployment, education/tax/social spending, income-group classification | Public REST API |
| [OECD Income Distribution Database](https://data-explorer.oecd.org/vis?fs[0]=Topic,1%7CSociety%23SOC%23%7CInequality%23SOC_INE%23&df[id]=DSD_WISE_IDD%40DF_IDD) | Gini index (OECD member countries, alternate methodology) | Public SDMX API |
| [WID.world](https://wid.world/data/) | Top-10% / bottom-50% pre-tax income shares | Public REST API (currently retired — best-effort, see limitations) |
| [Eurostat EU-SILC](https://ec.europa.eu/eurostat/web/microdata/collections-research/european-union-statistics-on-income-and-living-conditions) | Gini, S80/S20 ratio, at-risk-of-poverty rate (aggregate indicators only — see limitations) | Public JSON-stat API |
| World Bank GDIM | Intergenerational mobility (education-based proxy) | Public file download (data catalog, no auth) |

`ingestion/*.py` each document their exact request shape; `ingestion/merge_sources.py` joins
them into one country-year panel, coalescing the three Gini variants (World Bank > OECD >
Eurostat priority) into a single `gini_index` column.

## Architecture

```
┌─────────────┐        ┌──────────────┐  POST /train {run_ingestion:true}
│  Streamlit  │───────►│  predict-api │        │
│    :8501    │        │    :5003     │        ▼
└──────┬──────┘        │  gini · mob. │  ┌────────────────┐  monthly   ┌──────────────┐
       │               │  · inc.grp   │  │   Airflow      │◄──────────►│  train-api   │
       │ POST /predict-*│  Evidently  │  │    :8080       │  poll      │    :5002     │
       └───────────────►│  drift buf. │  └────────────────┘  status    │  ingestion   │
                        └──────┬───────┘         │                     │  -> features │
                               │ /reload-artifacts│                     │  -> 3 trains │
                               └───────────────────────────────────────►│  MLflow log  │
                                                                         └──────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│  Prometheus (:9090) ◄── scrapes train-api + predict-api                  │
│  Grafana    (:3000) ──► request rate, latency, model quality dashboards  │
│  Alertmanager(:9093)──► service-down / model-quality / drift alerts      │
└──────────────────────────────────────────────────────────────────────────┘
```

MLflow experiment tracking + model registry runs on DagsHub, not as a local container. DVC
versions `data/raw` and `data/artifacts` against the same DagsHub S3-compatible remote.

## Design Scope

This project's data is a few thousand country-year rows of tabular macro indicators, refreshed
on each source's own annual/biennial release calendar — a fundamentally different shape from a
high-throughput, high-QPS production workload, and the stack is scoped accordingly rather than
over-built:

- **Kept deliberately simple:** one feature table feeds three sklearn/XGBoost models — no
  multi-model fan-out, no GPU-bound encoders, no auth gateway (single-tenant demo scope).
- **Full MLOps surface anyway:** DVC + DagsHub, MLflow tracking, FastAPI train-api/predict-api
  split, Airflow orchestration, Docker Compose, Prometheus/Grafana/Alertmanager, pytest
  unit+integration suites, GitHub Actions CI — none of that gets cut just because the data is small.
- **Cadence matches the data, not a generic default:** Airflow runs `@monthly` instead of
  on-demand, matching how often the underlying sources actually publish new figures; Evidently
  drift monitoring watches macro-feature drift rather than prediction-confidence drift, since
  that's the more meaningful signal for a slowly-refreshed macro panel.

## Quick Start

### 1 — Clone and configure

```bash
git clone https://github.com/zz75da/income-inequality-mlops.git
cd income-inequality-mlops
cp .env.template .env       # fill in DAGSHUB_USER, DAGSHUB_TOKEN
dvc init                    # if not already committed
dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"
dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"
```

### 2 — Pull raw data (first run: ingest from scratch instead)

```bash
pip install -r requirements.txt
make ingest        # calls all 4 ingest_*.py scripts + merge_sources.py
make features       # build_features.py
# or, once data/raw is DVC-tracked from a previous run:
dvc pull
```

### 3 — Train locally (optional — otherwise train-api does this)

```bash
make dvc-repro      # runs the full DVC pipeline: ingest -> merge -> features -> train x3
```

### 4 — Start the stack

```bash
docker compose build
docker compose up -d
docker compose ps
curl http://localhost:5002/health   # train-api
curl http://localhost:5003/health   # predict-api
```

### 5 — Trigger training via the API (alternative to step 3)

```bash
curl -X POST http://localhost:5002/train -H "Content-Type: application/json" \
  -d '{"target": "all", "run_ingestion": true}'
curl http://localhost:5002/train/status/<job_id>
```

Or open `http://localhost:8080`, enable `income_inequality_pipeline`, and trigger it manually.

## Service Endpoints

### train-api — Training

```
POST /train              {"target": "gini|mobility|income_group|all", "run_ingestion": bool}
                          -> 202 {"job_id": "...", "status": "running"} | 409 if busy
GET  /train/status/{id}  -> {"status": "success|running|failed", "log": [...]}
GET  /health
GET  /metrics
```

### predict-api — Inference

```
POST /predict-gini            {feature fields...} -> {"gini_index": float}
POST /predict-mobility        {feature fields...} -> {"intergen_income_elasticity": float}
POST /predict-income-group    {feature fields...} -> {"income_group": str, "probabilities": {...}}
POST /reload-artifacts        (reload models after a training run)
GET  /drift-status
POST /drift-trigger-report
GET  /health
GET  /metrics
```

See `predict-api/app.py`'s `FeaturePayload` for the full list of accepted feature fields
(mirrors `params.yaml`'s `features.numeric_features` / `categorical_features`).

## Repository Structure

```
income-inequality-mlops/
├── ingestion/
│   ├── common.py                    # shared HTTP retry + long-CSV schema helper
│   ├── ingest_worldbank.py          # World Bank Open Data API
│   ├── ingest_oecd.py               # OECD SDMX API (generic SDMX-JSON flattener)
│   ├── ingest_eurostat.py           # Eurostat JSON-stat API
│   ├── ingest_wid.py                # WID.world API
│   ├── ingest_gdim.py               # GDIM (auto-downloads from World Bank data catalog)
│   └── merge_sources.py             # joins all sources -> merged_panel.csv
├── features/
│   └── build_features.py            # clean/impute/encode -> features.csv + categorical_mappings.json
├── train-api/
│   ├── app.py                       # /train async job runner
│   └── services/
│       ├── common.py                # shared param/feature loading, group split, MLflow setup
│       ├── train_gini.py
│       ├── train_mobility.py
│       └── train_income_group.py
├── predict-api/
│   ├── app.py                       # 3-model inference + Evidently drift buffer
│   └── services/drift_monitor.py
├── streamlit/app_streamlit.py       # Explore (choropleth) · Predict · Drift Reports pages
├── airflow/dags/income_pipeline_dag.py  # @monthly re-ingest + retrain + reload
├── monitoring/                      # Prometheus, Alertmanager, Grafana dashboards+provisioning
├── tests/                           # unit + integration pytest suites
├── .github/workflows/ci.yml         # tests + DVC remote status check
├── dvc.yaml                         # ingest -> merge -> features -> train x3 pipeline
├── params.yaml                      # all tunable hyperparameters + feature list
└── docker-compose.yml               # postgres, airflow, train-api, predict-api, streamlit,
                                      # prometheus, grafana, alertmanager, pushgateway
```

## Environment Variables

Copy `.env.template` to `.env` — see that file for the full list (DagsHub/MLflow credentials,
Airflow Fernet/secret keys, Grafana admin password, optional WID API key).

## Known Limitations

- **Verified end-to-end against live sources (2026-08).** World Bank, OECD, and Eurostat
  ingestion all run cleanly against the live APIs. One real drift was caught and fixed: OECD's
  SDMX-JSON response nests the dimension structure under `data.structures[0]`, not
  `data.structure` as originally coded — `ingestion/ingest_oecd.py` now matches the real shape.
- **WID.world's public REST API (`/api/v3.php`) has been retired** and returns 404 with no
  documented public replacement — the site and official R package now call a private,
  key-gated AWS API Gateway endpoint not obtainable through public signup. `ingest_wid.py` is
  therefore treated as best-effort: its failure doesn't abort a training run, and
  `top10_income_share`/`bottom50_income_share` fall back to median imputation when absent. See
  the script's docstring for details.
- **No true household income-bracket target.** EU-SILC microdata (individual/household rows)
  requires a restricted research-access agreement with Eurostat — it is not available through
  any free API. `income_group` (World Bank's Low/Lower-middle/Upper-middle/High classification)
  is used as the closest legitimately-obtainable "bracket" target instead.
- **GDIM mobility uses an education-based proxy, not true income mobility.** `ingest_gdim.py`
  auto-downloads the public World Bank Data Catalog CSV (no manual step, no auth) and uses its
  `BETA`/`COR` columns (intergenerational persistence/rank-correlation of years-of-schooling) as
  the `intergen_income_elasticity`/`intergen_rank_correlation` features — there's no free dataset
  with this country coverage using actual income. If the dataset's Data Catalog resource id ever
  changes, the script falls back to `data/raw/gdim_manual_download.csv` if present, then degrades
  to an empty output (same best-effort pattern as `ingest_wid.py`) rather than failing the job.
- **World Bank income-group classification is a current snapshot, not a historical series** via
  the API; a true year-by-year series requires manually downloading the OGHIST spreadsheet
  (documented in `ingest_worldbank.py`).

---

## Author & License

**Author:** [zz75da](https://github.com/zz75da)

Licensed under the [MIT License](LICENSE) — Copyright © 2026 zz75da.

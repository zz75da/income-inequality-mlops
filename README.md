# Income Inequality MLOps Platform

[![CI — Tests & DVC Sync](https://github.com/zz75da/income-inequality-mlops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/zz75da/income-inequality-mlops/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** [zz75da](https://github.com/zz75da)

End-to-end MLOps platform predicting country-level income inequality from public macroeconomic data.
Same technical stack family as [rakuten_mlops_services](https://github.com/zz75da/rakuten_z) (FastAPI
microservices, Airflow, DVC/DagsHub, MLflow, Prometheus/Grafana), scoped down for a tabular,
low-QPS, periodically-refreshed dataset instead of a high-throughput multimodal classifier.

## Table of Contents

- [Prediction Targets](#prediction-targets)
- [Data Sources](#data-sources)
- [Architecture](#architecture)
- [Why This Differs From rakuten_mlops_services](#why-this-differs-from-rakuten_mlops_services)
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
| [WID.world](https://wid.world/data/) | Top-10% / bottom-50% pre-tax income shares | Public REST API |
| [Eurostat EU-SILC](https://ec.europa.eu/eurostat/web/microdata/collections-research/european-union-statistics-on-income-and-living-conditions) | Gini, S80/S20 ratio, at-risk-of-poverty rate (aggregate indicators only — see limitations) | Public JSON-stat API |
| World Bank GDIM | Intergenerational income elasticity | Manual download (no API) |

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

MLflow experiment tracking + model registry runs on DagsHub (same pattern as
rakuten_mlops_services), not as a local container. DVC versions `data/raw` and `data/artifacts`
against the same DagsHub S3-compatible remote.

## Why This Differs From rakuten_mlops_services

Rakuten's stack (85k-row multimodal text+image classification, 4 parallel text encoders,
JWT gateway, K8s HPA proof-of-concept) was scoped for a high-throughput production simulation.
This project's data is a few thousand country-year rows of tabular macro indicators refreshed
on each source's own annual/biennial release calendar — copying that stack wholesale would add
dead weight. What changed:

- **Dropped:** `gate-api` JWT auth, `clip-encoder`/`minilm-encoder` microservices, OCR, GradCAM,
  K8s manifests. No multi-encoder fan-out — one feature table feeds three simple sklearn/XGBoost
  models.
- **Kept as-is:** DVC + DagsHub, MLflow tracking, FastAPI train-api/predict-api split, Airflow
  orchestration, Docker Compose, Prometheus/Grafana/Alertmanager, pytest unit+integration suites,
  GitHub Actions CI.
- **Retargeted:** Airflow runs `@monthly` instead of on-demand (matches source release cadence);
  Evidently drift monitoring watches macro-feature drift instead of prediction-confidence drift.

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

Note: `ingestion/ingest_gdim.py` requires manually downloading the World Bank's GDIM
spreadsheet first (see the script's docstring) — everything else runs unattended.

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
│   ├── ingest_gdim.py               # GDIM (manual download, no public API)
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

- **Sandbox-authored, not sandbox-tested end-to-end.** The four `ingest_*.py` scripts are
  written against each source's documented, stable public API contract, but the environment
  this project was scaffolded in has no outbound access to `api.worldbank.org`,
  `sdmx.oecd.org`, `ec.europa.eu`, or `wid.world`. `ingestion/merge_sources.py` and
  `features/build_features.py` *have* been exercised end-to-end against synthetic fixtures
  (see `tests/integration/test_workflow.py`) and confirmed to coalesce/impute/encode
  correctly. Run the ingestion scripts from a machine with normal internet access; if an
  API response shape has drifted since this was written, the error will point at exactly
  which parse step failed.
- **No true household income-bracket target.** EU-SILC microdata (individual/household rows)
  requires a restricted research-access agreement with Eurostat — it is not available through
  any free API. `income_group` (World Bank's Low/Lower-middle/Upper-middle/High classification)
  is used as the closest legitimately-obtainable "bracket" target instead.
- **GDIM mobility data requires a one-time manual download** (no public API for it at all) —
  see `ingestion/ingest_gdim.py`.
- **World Bank income-group classification is a current snapshot, not a historical series** via
  the API; a true year-by-year series requires manually downloading the OGHIST spreadsheet
  (documented in `ingest_worldbank.py`).

---

## Author & License

**Author:** [zz75da](https://github.com/zz75da)

Licensed under the [MIT License](LICENSE) — Copyright © 2026 zz75da.

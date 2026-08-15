# Income Inequality MLOps — convenience targets
COMPOSE = docker compose

.PHONY: up down logs restart build ingest features train test lint dvc-repro dvc-push observability grafana-dash

# --- Full stack ---
up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) down && $(COMPOSE) up -d

logs:
	$(COMPOSE) logs -f

build:
	$(COMPOSE) build

# --- Data pipeline (run locally, outside containers) ---
ingest:
	python ingestion/ingest_worldbank.py
	python ingestion/ingest_oecd.py
	python ingestion/ingest_wid.py
	python ingestion/ingest_eurostat.py
	python ingestion/merge_sources.py

features:
	python features/build_features.py

dvc-repro:
	dvc repro

dvc-push:
	dvc push

# --- Tests ---
test:
	pytest tests/unit -v
	pytest tests/integration -v -m integration

lint:
	python -m py_compile $$(find . -name "*.py" -not -path "./.venv/*" -not -path "./airflow/logs/*")

# --- Observability only ---
observability:
	$(COMPOSE) up -d prometheus grafana pushgateway

grafana-dash:
	@echo "Importing Grafana dashboard..."
	docker cp monitoring/grafana_dashboards/income_inequality_dashboard.json $$(docker ps -qf "name=grafana"):/var/lib/grafana/dashboards/income_inequality_dashboard.json
	@echo "Dashboard imported. http://localhost:3000 (admin/admin)"

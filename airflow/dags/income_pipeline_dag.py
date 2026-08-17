"""
Income Inequality MLOps — scheduled ingestion + retrain DAG.

Unlike a real-time product-catalog pipeline retrained on demand, the four
data sources here (World Bank, OECD, Eurostat, WID) each
publish new figures on their own annual/biennial release calendars. This DAG
therefore runs on a conservative monthly schedule and simply no-ops
usefully (re-pulls return mostly unchanged data) between real upstream
releases — cheap insurance against silently going stale, without needing to
track each source's exact release calendar.

Flow: call train-api's POST /train with run_ingestion=true (re-pulls all 5
sources including GDIM, rebuilds features, retrains all 3 models), poll
GET /train/status/{id} until done, then hit predict-api's
POST /reload-artifacts so the running inference service picks up the new
models without a restart.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta

import requests

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger("airflow.income_pipeline")

TRAIN_API_URL = os.getenv("TRAIN_API_URL", "http://train-api:5002")
PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://predict-api:5003")
POLL_INTERVAL_SECONDS = 30
MAX_WAIT_SECONDS = int(os.getenv("TRAIN_MAX_WAIT_SECONDS", str(2 * 60 * 60)))  # 2h ceiling

default_args = {
    "owner": "income-inequality-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def trigger_and_wait_for_training(**context) -> None:
    resp = requests.post(f"{TRAIN_API_URL}/train", json={"target": "all", "run_ingestion": True}, timeout=30)
    if resp.status_code == 409:
        raise RuntimeError("train-api already has a job running — will retry next scheduled interval")
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    logger.info("Started training job %s", job_id)

    waited = 0
    while waited < MAX_WAIT_SECONDS:
        status_resp = requests.get(f"{TRAIN_API_URL}/train/status/{job_id}", timeout=30)
        status_resp.raise_for_status()
        status = status_resp.json()["status"]
        if status == "success":
            logger.info("Training job %s succeeded", job_id)
            return
        if status == "partial_success":
            # Some targets trained and were saved/logged to MLflow even though
            # others failed (e.g. a transient upstream API outage) — still worth
            # reloading predict-api so the targets that DID train go live, instead
            # of retrying/failing the whole DAG run over one target's bad luck.
            logger.warning("Training job %s partially succeeded: %s", job_id, status_resp.json())
            return
        if status == "failed":
            raise RuntimeError(f"Training job {job_id} failed: {status_resp.json()}")
        time.sleep(POLL_INTERVAL_SECONDS)
        waited += POLL_INTERVAL_SECONDS

    raise TimeoutError(f"Training job {job_id} did not finish within {MAX_WAIT_SECONDS}s")


def reload_predict_api(**context) -> None:
    resp = requests.post(f"{PREDICT_API_URL}/reload-artifacts", timeout=30)
    resp.raise_for_status()
    logger.info("predict-api reloaded: %s", resp.json())


with DAG(
    dag_id="income_inequality_pipeline",
    description="Monthly re-ingest of World Bank/OECD/Eurostat/WID data and retrain of all 3 models",
    default_args=default_args,
    schedule_interval="@monthly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["income-inequality", "mlops"],
) as dag:
    train_task = PythonOperator(
        task_id="trigger_and_wait_for_training",
        python_callable=trigger_and_wait_for_training,
    )

    reload_task = PythonOperator(
        task_id="reload_predict_api",
        python_callable=reload_predict_api,
    )

    train_task >> reload_task

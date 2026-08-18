"""Daily PSI monitoring DAG.

Computes the Population Stability Index of the current scoring population against
the training baseline and fails (alerts) if PSI exceeds the threshold (0.25).
The computation is unit-tested in ``jobs/psi.py``; this DAG only schedules it.
"""
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import sys

# Make the job modules importable (mounted at /opt/airflow/jobs in the container).
JOBS_DIR = os.getenv("AIRFLOW_JOBS_DIR", "/opt/airflow/jobs")
if JOBS_DIR not in sys.path:
    sys.path.insert(0, JOBS_DIR)


def _run_psi_check(**_context):
    import psi_monitoring
    return psi_monitoring.run_psi_check()


with DAG(
    dag_id="psi_monitoring",
    schedule_interval="@daily",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["monitoring", "model-risk", "psi"],
) as dag:
    psi_check = PythonOperator(
        task_id="compute_population_stability_index",
        python_callable=_run_psi_check,
    )

    psi_check

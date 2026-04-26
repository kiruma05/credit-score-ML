from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

def log_start():
    print(" Starting Fraud Detection model training pipeline...")

def log_end():
    print(" Fraud Detection model training finished successfully.")

with DAG(
    dag_id='fraud_detection_dag',
    default_args={'owner': 'frank', 'start_date': datetime(2025, 10, 8)},
    schedule_interval=None,
    catchup=False,
    tags=['fraud', 'mlflow', 'training']
) as dag:

    start_task = PythonOperator(task_id="start", python_callable=log_start)

    train_fraud_model = BashOperator(
        task_id='train_fraud_model',
        bash_command='python /opt/airflow/jobs/train_fraud_model.py'
    )

    end_task = PythonOperator(task_id="end", python_callable=log_end)

    start_task >> train_fraud_model >> end_task

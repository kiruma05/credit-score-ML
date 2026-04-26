import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


with DAG(
        dag_id='pandas_model_training',
        default_args={
            "owner": "Rama",
    "start_date": airflow.utils.dates.days_ago(1)
        },
        schedule_interval=None,
        start_date=airflow.utils.dates.days_ago(1),
        catchup=False,
        tags=['pandas', 'sklearn', 'training'],
) as dag:
    # Task 1: A simple PythonOperator to log the start of the process

    start_task = PythonOperator(
        task_id="start",
        python_callable=lambda: print("Starting the Pandas/Scikit-learn model training pipeline...")
    )



    train_models_task = BashOperator(
        task_id='train_pandas_sklearn_models',
        # This command runs the python script from the mounted directory
        # The Airflow container has all the necessary env vars from docker-compose
        bash_command='python /opt/airflow/jobs/train_models_pandas.py',
    )

    end_task = PythonOperator(
        task_id="end",
        python_callable=lambda: print("Pandas/Scikit-learn model training pipeline finished successfully.")
    )

    start_task >> train_models_task >> end_task

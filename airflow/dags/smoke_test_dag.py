from airflow.models.dag import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime
import os

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

with DAG(
        dag_id='mlflow_infrastructure_smoke_test',
        schedule_interval=None,
        start_date=datetime(2023, 1, 1),
        catchup=False,
        tags=['test', 'mlops', 'smoke-test'],
) as dag:
    run_smoke_test = SparkSubmitOperator(
        task_id="run_mlflow_connectivity_test",
        conn_id="spark_default",
        # Point to our new, simple script
        application="/opt/bitnami/spark/jobs/smoke_test.py",
        # conf={
        #     'spark.master': 'spark://spark-master:7077',
        #     'spark.pyspark.python': '/opt/bitnami/python/bin/python',
        #     'spark.pyspark.driver.python': '/home/airflow/.local/bin/python',
        # },
        env_vars={
            'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI,
            'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY_ID,
            'AWS_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY,
            'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000'
        },
        verbose=True
    )


    run_smoke_test

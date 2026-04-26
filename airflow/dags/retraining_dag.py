from airflow.models.dag import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

# MLflow URI is passed via environment variable from the docker-compose file
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
SPARK_MASTER_URL = "spark://spark-master:7077"
air = '/home/airflow/.local/bin/python'

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


with DAG(
    dag_id='model_retraining_with_spark_operator',
    schedule_interval=None, # Set to None for manual triggers
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['credit-scoring', 'spark-submit-operator'],
) as dag:

    start = PythonOperator(
        task_id="start",
        python_callable = lambda: print("Jobs started")
    )

    # This operator submits the job to the Spark Master defined in the connection
    submit_spark_job = SparkSubmitOperator(
        task_id="spark_train_submit_task",
        # This conn_id must be configured in the Airflow UI
        conn_id="spark_default",
        # Path to the script inside the Airflow container's volume
        application="/opt/bitnami/spark/jobs/train_models.py",
        # Explicitly set the master in the configuration
        conf={
            'spark.master': 'spark://spark-master:7077',

            # <<< START OF THE FIX >>>
            # Path for the Python on the SPARK WORKER nodes
            'spark.pyspark.python': '/opt/bitnami/python/bin/python',

            # Path for the Python on the DRIVER (the Airflow container)
            # Replace this with the path you found in Step 1
            'spark.pyspark.driver.python': '/home/airflow/.local/bin/python',

                                           "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
    "spark.hadoop.fs.s3a.access.key": AWS_ACCESS_KEY_ID,
    "spark.hadoop.fs.s3a.secret.key": AWS_SECRET_ACCESS_KEY,
    "spark.hadoop.fs.s3a.path.style.access": "true",
    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",

    # Pass credentials to executors
    "spark.executor.extraJavaOptions": f"-Dcom.amazonaws.services.s3.enableV4=true -Daws.accessKeyId={AWS_ACCESS_KEY_ID} -Daws.secretAccessKey={AWS_SECRET_ACCESS_KEY}",
    "spark.driver.extraJavaOptions": f"-Dcom.amazonaws.services.s3.enableV4=true -Daws.accessKeyId={AWS_ACCESS_KEY_ID} -Daws.secretAccessKey={AWS_SECRET_ACCESS_KEY}",
       "spark.driver.memory": "8g",
            # cpu
            "spark.executor.cores": "8",
        },
        # Pass the MLflow URI to the Spark script's environment
        env_vars={
    'MLFLOW_TRACKING_URI': MLFLOW_TRACKING_URI,
            'AWS_ACCESS_KEY_ID': AWS_ACCESS_KEY_ID,
            'AWS_SECRET_ACCESS_KEY': AWS_SECRET_ACCESS_KEY,
            'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000',
            'BOTO_CONFIG': '/dev/null',
            'AWS_METADATA_SERVICE_TIMEOUT': '5',
            'AWS_METADATA_SERVICE_NUM_ATTEMPTS': '2',
        },
        verbose=True
    )

    end = PythonOperator(
        task_id="end",
        python_callable = lambda: print("Jobs completed successfully")
    )

    start >> submit_spark_job >> end
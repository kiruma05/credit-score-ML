FROM apache/airflow:2.11.0-python3.10

USER root
RUN apt-get update && \
    apt-get install -y gcc python3-dev openjdk-17-jdk && \
    apt-get clean

# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
# COPY ./data /opt/bitnami/spark/data
# COPY ./airflow/dags /opt/airflow/dags
# COPY ./logs /opt/airflow/logs
# COPY ./airflow/jobs /opt/airflow/jobs
# COPY ./data /opt/airflow/data


USER airflow

RUN pip install apache-airflow apache-airflow-providers-apache-spark mlflow==2.7.1 pyspark==3.3.3
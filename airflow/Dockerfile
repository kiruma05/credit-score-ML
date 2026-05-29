FROM apache/airflow:2.11.0-python3.10

ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64

USER airflow

RUN pip install apache-airflow apache-airflow-providers-apache-spark mlflow==2.7.1 pyspark==3.3.3
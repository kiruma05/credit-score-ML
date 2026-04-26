# Start from the same base Spark image we were using
FROM bitnamilegacy/spark:3.3.3
LABEL authors="kiruma"

# Switch to root user to install packages
USER root

RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/* && \
    wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.2/hadoop-aws-3.3.2.jar -P ${SPARK_HOME}/jars/ && \
    wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar -P ${SPARK_HOME}/jars/

# --- INSTRUCTION 2: Install Python's package manager ---
RUN apt-get update && \
    apt-get install -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

# --- INSTRUCTION 3: Install Python libraries ---
RUN pip install \
    mlflow==2.7.1 \
    boto3 \
    pyspark==3.3.3 \
    scikit-learn \
    pandas \
    numpy

# Switch back to the non-root user that the base image uses
USER 1001

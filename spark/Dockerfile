# Start from the same base Spark image we were using
FROM bitnamilegacy/spark:3.3.3
LABEL authors="kiruma"

# Switch to root user to install packages
USER root

# --- INSTRUCTION 1: Fetch the Hadoop-AWS / S3A jars ---
# The bitnami base image ships without wget or curl, and the Debian bullseye
# security pool intermittently 404s, so we avoid needing a downloader at all:
# Docker's native remote ADD lets BuildKit fetch the jars with its own HTTP
# client (no apt/wget/curl). --chmod=644 makes them world-readable so the
# non-root spark user (uid 1001) can load them onto the classpath.
ADD --chmod=644 https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.2/hadoop-aws-3.3.2.jar /opt/bitnami/spark/jars/
ADD --chmod=644 https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar /opt/bitnami/spark/jars/

# --- INSTRUCTION 2: Install Python's package manager ---
# Retry once on a transient mirror/CDN failure before giving up.
RUN (apt-get update && apt-get install -y --no-install-recommends python3-pip) || \
    (sleep 5 && apt-get update && apt-get install -y --no-install-recommends --fix-missing python3-pip); \
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

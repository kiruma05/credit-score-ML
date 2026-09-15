# Start from the same base Spark image we were using
FROM bitnamilegacy/spark:3.3.3
LABEL authors="kiruma"

# Switch to root user to install packages
USER root

# --- INSTRUCTION 1: Fetch the Hadoop-AWS / S3A jars ---
# Download with a tool already present in the base image (curl) instead of
# apt-installing wget. This avoids a hard dependency on the Debian bullseye
# security pool, which intermittently 404s specific point-release .debs when
# the CDN index is out of sync with the pool.
RUN set -eux; \
    JARDIR="${SPARK_HOME}/jars"; \
    HADOOP_AWS="https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.2/hadoop-aws-3.3.2.jar"; \
    AWS_SDK="https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar"; \
    if command -v curl >/dev/null 2>&1; then \
        curl -fSL "$HADOOP_AWS" -o "$JARDIR/hadoop-aws-3.3.2.jar"; \
        curl -fSL "$AWS_SDK"    -o "$JARDIR/aws-java-sdk-bundle-1.12.367.jar"; \
    else \
        wget -q "$HADOOP_AWS" -P "$JARDIR/"; \
        wget -q "$AWS_SDK"    -P "$JARDIR/"; \
    fi

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

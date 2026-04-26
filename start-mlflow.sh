#!/bin/sh
set -e
: "${POSTGRES_USER?ERROR: POSTGRES_USER is not set}"
: "${POSTGRES_PASSWORD?ERROR: POSTGRES_PASSWORD is not set}"
: "${POSTGRES_DB?ERROR: POSTGRES_DB is not set}"
: "${AWS_ACCESS_KEY_ID?ERROR: AWS_ACCESS_KEY_ID is not set}"
: "${AWS_SECRET_ACCESS_KEY?ERROR: AWS_SECRET_ACCESS_KEY is not set}"
echo "All required environment variables are set."
echo "Installing Python dependencies..."
pip install mlflow==2.9.2 psycopg2-binary boto3
echo "Starting MLflow server..."
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}" \
    --default-artifact-root "s3://mlflow/"
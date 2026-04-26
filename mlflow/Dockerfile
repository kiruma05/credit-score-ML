# Start from the same Python base image
FROM python:3.10-slim
LABEL authors="rama"

# Set the working directory
WORKDIR /app

# Install dependencies. By installing them in the image,
# we don't need to do it every time the container starts.
RUN pip install \
    mlflow==2.7.1 \
    psycopg2-binary \
    boto3 \
    click==8.1.3 \
    Werkzeug==2.2.2 \
    scikit-learn==1.5.2 \
    cloudpickle \
    pandas==2.1.4 \
    psutil==7.0.0

# Expose the port that MLflow will run on
EXPOSE 5000

# The command to run when the container starts.
# We use the 'exec' form to make it the main process.
# Environment variables for the URI and artifact root will be passed in from docker-compose.
CMD [ \
    "mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "$(BACKEND_STORE_URI)", \
    "--default-artifact-root", "$(ARTIFACT_ROOT)" \
]

# ui cmd
# CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "6060"]



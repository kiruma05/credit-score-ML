import os
import sys
import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from urllib import request, error


def check_mlflow_connection(uri, timeout=15):
    """Checks if the MLflow tracking server is reachable."""
    print(f"--- Checking MLflow connection to {uri} (timeout: {timeout}s) ---")
    try:
        health_uri = f"{uri}/health"
        response = request.urlopen(health_uri, timeout=timeout)
        if response.status == 200:
            print("--- MLflow connection successful! ---")
            return
    except Exception as e:
        print(f"CRITICAL: Failed to connect to MLflow server at {uri}. Error: {e}")
        sys.exit(1)


def run_smoke_test():
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    if not MLFLOW_TRACKING_URI:
        print("CRITICAL: MLFLOW_TRACKING_URI environment variable is not set.")
        sys.exit(1)

    check_mlflow_connection(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    spark = SparkSession.builder.appName("SmokeTest").getOrCreate()

    model_name = "SmokeTestModel"

    try:
        print(f"\n--- STEP 1: LOGGING DUMMY ARTIFACT ---")
        with mlflow.start_run(run_name="SmokeTestLog") as run:
            with open("dummy_model.txt", "w") as f:
                f.write("This is a smoke test model.")

            print("Logging dummy_model.txt to artifact_path 'test_model'")
            mlflow.log_artifact("dummy_model.txt", artifact_path="test_model")
            run_id = run.info.run_id
            print(f"Successfully created run with ID: {run_id}")

        # --- START OF THE FIX ---
        # Manually register the model in a more robust way

        print(f"\n--- STEP 2: REGISTERING THE MODEL ---")
        artifact_uri = f"runs:/{run_id}/test_model"

        # Check if the model name exists, create it if not.
        try:
            client.get_registered_model(name=model_name)
            print(f"Registered model '{model_name}' already exists.")
        except mlflow.exceptions.MlflowException:
            print(f"Registered model '{model_name}' does not exist. Creating it.")
            client.create_registered_model(name=model_name)

        # Create a new version for this model from our artifact
        print(f"Creating a new version for model '{model_name}' from artifact URI: {artifact_uri}")
        new_version = client.create_model_version(
            name=model_name,
            source=artifact_uri,
            run_id=run_id
        )
        print(f"Successfully created model version {new_version.version} for '{model_name}'")
        # --- END OF THE FIX ---

    except Exception as e:
        print(f"CRITICAL: Failed during the LOGGING/REGISTRATION step. Error: {e}")
        spark.stop()
        sys.exit(1)

    # 3. Load the model artifact back
    try:
        print(f"\n--- STEP 3: LOADING MODEL ARTIFACT ---")
        model_uri_to_load = f"models:/{model_name}/{new_version.version}"
        print(f"Attempting to download artifacts from: {model_uri_to_load}")

        local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri_to_load)
        print(f"Successfully downloaded artifacts to local path: {local_path}")

        with open(os.path.join(local_path, "dummy_model.txt"), "r") as f:
            content = f.read()
            print(f"Content of downloaded file: '{content}'")
            if "smoke test model" in content:
                print("--- SMOKE TEST PASSED! ---")
            else:
                raise ValueError("Downloaded file content does not match!")

    except Exception as e:
        print(f"CRITICAL: Failed during the LOADING step. Error: {e}")
        spark.stop()
        sys.exit(1)

    spark.stop()


if __name__ == '__main__':
    run_smoke_test()
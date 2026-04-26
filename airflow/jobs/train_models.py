import os
import sys
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import RandomForestClassifier  # Corrected import
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from urllib import request, error


def check_mlflow_connection(uri, timeout=20):
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


def log_and_register_model(client, model, model_name, artifact_path, run_id):
    """Logs a model artifact and registers it in the MLflow Model Registry."""
    print(f"\n>>> LOGGING & REGISTERING: {model_name}")

    # 1. Log the model as an artifact within the current run
    mlflow.spark.log_model(model, artifact_path)
    print(f"Logged artifact to path: {artifact_path}")

    # 2. Register the model using the client
    artifact_uri = f"runs:/{run_id}/{artifact_path}"
    try:
        client.get_registered_model(name=model_name)
        print(f"Registered model '{model_name}' already exists.")
    except mlflow.exceptions.MlflowException:
        print(f"Registered model '{model_name}' does not exist. Creating it.")
        client.create_registered_model(name=model_name)

    # 3. Create a new version for this model from our artifact
    print(f"Creating new version for '{model_name}' from artifact URI: {artifact_uri}")
    new_version = client.create_model_version(
        name=model_name,
        source=artifact_uri,
        run_id=run_id
    )
    print(f"Successfully created model version {new_version.version} for '{model_name}'")


def train():
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    if not MLFLOW_TRACKING_URI:
        print("CRITICAL: MLFLOW_TRACKING_URI environment variable is not set.")
        sys.exit(1)

    check_mlflow_connection(MLFLOW_TRACKING_URI)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    spark = SparkSession.builder.appName("CreditScoringTraining").getOrCreate()

    data_path = "/opt/bitnami/spark/data/customer_data.csv"
    data = spark.read.csv(data_path, header=True, inferSchema=True).fillna(0)

    (train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

    # --- Feature Engineering Pipeline Definitions ---
    categorical_cols = [f.name for f in data.schema.fields if isinstance(f.dataType, T.StringType)]
    categorical_cols = [c for c in categorical_cols if c not in ['customer_id', 'nida', 'risk_category_target']]
    numerical_cols = [f.name for f in data.schema.fields if isinstance(f.dataType, (T.IntegerType, T.DoubleType))]

    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep") for col in
                categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{col}_indexed", outputCol=f"{col}_encoded") for col in categorical_cols]
    label_indexer = StringIndexer(inputCol="risk_category_target", outputCol="label")

    assembler_inputs = [f"{col}_encoded" for col in categorical_cols] + numerical_cols
    vector_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # ---- Model 1: Credit Score (Regression) ----
    with mlflow.start_run(run_name="CreditScore_GBT_Regressor") as run:
        print(">>> TRAINING: CreditScorePredictor")
        gbt_score = GBTRegressor(featuresCol="features", labelCol="payment_history_score", maxIter=10)
        pipeline_score = Pipeline(stages=indexers + encoders + [vector_assembler, gbt_score])
        model_score = pipeline_score.fit(train_data)

        print(">>> EVALUATING: CreditScorePredictor")
        predictions_score = model_score.transform(test_data)
        evaluator_rmse = RegressionEvaluator(labelCol="payment_history_score", predictionCol="prediction",
                                             metricName="rmse")
        rmse = evaluator_rmse.evaluate(predictions_score)

        print(f">>> METRIC (CreditScore RMSE): {rmse}")
        mlflow.log_metric("rmse", rmse)
        log_and_register_model(client, model_score, "CreditScorePredictor", "credit_score_model", run.info.run_id)

    # ---- Model 2: Risk Category (Classification) ----
    with mlflow.start_run(run_name="RiskCategory_RandomForest_Classifier") as run:
        print(">>> TRAINING: RiskCategoryPredictor (using RandomForest)")
        # Use RandomForestClassifier as it supports multiclass classification
        rf_risk = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20)

        pipeline_risk = Pipeline(stages=indexers + encoders + [label_indexer, vector_assembler, rf_risk])
        model_risk = pipeline_risk.fit(train_data)

        print(">>> EVALUATING: RiskCategoryPredictor (using RandomForest)")
        predictions_risk = model_risk.transform(test_data)
        evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        f1_score = evaluator_f1.evaluate(predictions_risk)

        print(f">>> METRIC (RiskCategory F1): {f1_score}")
        mlflow.log_metric("f1_score", f1_score)
        log_and_register_model(client, model_risk, "RiskCategoryPredictor", "risk_category_model", run.info.run_id)

    # ---- Model 3: Recommended Credit Limit (Regression) ----
    with mlflow.start_run(run_name="CreditLimit_GBT_Regressor") as run:
        print(">>> TRAINING: CreditLimitPredictor")
        # Create a target column for the credit limit model
        data_with_limit_target = data.withColumn("limit_target", F.col("monthly_income") * 3)
        (train_limit_data, test_limit_data) = data_with_limit_target.randomSplit([0.8, 0.2], seed=42)

        gbt_limit = GBTRegressor(featuresCol="features", labelCol="limit_target", maxIter=10)
        pipeline_limit = Pipeline(stages=indexers + encoders + [vector_assembler, gbt_limit])
        model_limit = pipeline_limit.fit(train_limit_data)

        print(">>> EVALUATING: CreditLimitPredictor")
        predictions_limit = model_limit.transform(test_limit_data)
        evaluator_rmse_limit = RegressionEvaluator(labelCol="limit_target", predictionCol="prediction",
                                                   metricName="rmse")
        rmse_limit = evaluator_rmse_limit.evaluate(predictions_limit)

        print(f">>> METRIC (CreditLimit RMSE): {rmse_limit}")
        mlflow.log_metric("rmse", rmse_limit)
        log_and_register_model(client, model_limit, "CreditLimitPredictor", "credit_limit_model", run.info.run_id)

    print("\n>>> All models trained and logged successfully. Stopping Spark session. <<<")
    spark.stop()


if __name__ == '__main__':
    train()
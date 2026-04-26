import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from sklearn.preprocessing import LabelEncoder

# Load transaction data
df = pd.read_csv("/opt/airflow/data/transactions.csv")

# Encode categorical columns (object dtype)
cat_cols = df.select_dtypes(include='object').columns.tolist()

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders to file
with open("/opt/airflow/data/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Define features and target
X = df.drop("fraud_bool", axis=1)
y = df["fraud_bool"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="FraudDetectorTraining") as run:
    # Log model parameters
    n_estimators = 100
    random_state = 42
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.set_tag("model_name", "FraudDetector")
    mlflow.set_tag("dataset", "transactions.csv")
    
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # Predict and evaluate
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
    })

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="fraud_model")

    # Register model
    mlflow.register_model(
        f"runs:/{run.info.run_id}/fraud_model",
        "FraudDetector"
    )

print(f"Model training run completed: {run.info.run_id}")

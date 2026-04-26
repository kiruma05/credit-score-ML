import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Generate dummy data
def generate_dummy_fraud_data(num_records=5000, seed=42):  # Increased records for better training
    np.random.seed(seed)
    df = pd.DataFrame({
        "customer_id": [f"cust_{i}" for i in range(num_records)],
        "nida": [f"nida_{1000 + i}" for i in range(num_records)],
        "income": np.round(np.random.beta(2, 5, num_records), 3),
        "name_email_similarity": np.random.uniform(0, 1, num_records).round(3),
        "prev_address_months_count": np.random.randint(-1, 381, num_records),
        "current_address_months_count": np.random.randint(-1, 407, num_records),
        "customer_age": np.random.choice(range(20, 70, 10), num_records),
        "days_since_request": np.random.randint(0, 79, num_records),
        "intended_balcon_amount": np.random.randint(-1, 109, num_records),
        "payment_type": np.random.choice([f"type_{i}" for i in range(1, 6)], num_records),
        "velocity_6h": np.random.randint(-211, 24763, num_records),
        "velocity_24h": np.random.randint(1329, 9527, num_records),
        "velocity_4w": np.random.randint(2779, 7043, num_records),
        "bank_branch_count_8w": np.random.randint(0, 2522, num_records),
        "date_of_birth_distinct_emails_4w": np.random.randint(0, 43, num_records),
        "employment_status": np.random.choice([f"emp_status_{i}" for i in range(1, 8)], num_records),
        "credit_risk_score": np.random.randint(-176, 388, num_records),
        "email_is_free": np.random.choice(["free", "paid"], num_records),
        "housing_status": np.random.choice([f"house_{i}" for i in range(1, 8)], num_records),
        "phone_mobile_valid": np.random.choice([0, 1], num_records, p=[0.1, 0.9]),
        "bank_months_count": np.random.randint(-1, 32, num_records),
        "has_other_cards": np.random.choice([0, 1], num_records, p=[0.7, 0.3]),
        "proposed_credit_limit": np.random.randint(200, 2001, num_records),
        "foreign_request": np.random.choice([0, 1], num_records, p=[0.9, 0.1]),
        "source": np.random.choice(["INTERNET", "MOBILE"], num_records, p=[0.6, 0.4]),
        "session_length_in_minutes": np.random.randint(-1, 108, num_records),
        "device_os": np.random.choice(["Windows", "Macintosh", "Linux", "X11", "Other"], num_records),
        "keep_alive_session": np.random.choice([0, 1], num_records, p=[0.3, 0.7]),
        "device_distinct_emails_8w": np.random.randint(0, 4, num_records),
        "device_fraud_count": np.random.randint(0, 2, num_records),
        "month": np.random.randint(1, 13, num_records),
    })

    # Target variable
    fraud_score = (
        (1 - df["name_email_similarity"]) * 2 +
        (df["velocity_6h"] / 25000) +
        (df["foreign_request"] * 3) +
        (df["email_is_free"] == "free").astype(int) +
        (df["device_fraud_count"]) +
        (df["credit_risk_score"] < 0).astype(int)
    )

    threshold = np.percentile(fraud_score, 90)
    df["fraud_bool"] = (fraud_score > threshold).astype(int)

    # Balance classes for better model performance
    fraud_df = df[df['fraud_bool'] == 1]
    non_fraud_df = df[df['fraud_bool'] == 0].sample(n=len(fraud_df) * 3, random_state=42)
    df = pd.concat([fraud_df, non_fraud_df])

    return df

df = generate_dummy_fraud_data()

# Encode categorical features
categorical_features = [
    'customer_id', 'nida', 'payment_type', 'employment_status',
    'email_is_free', 'housing_status', 'phone_mobile_valid',
    'has_other_cards', 'foreign_request', 'source', 'device_os',
    'keep_alive_session', 'device_fraud_count', 'month'
]
encoders = {}
for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# Train model
features = [col for col in df.columns if col != 'fraud_bool']
X = df[features]
y = df['fraud_bool']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test)))

# Save model and encoders
os.makedirs('data', exist_ok=True)
with open('data/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('data/label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Model retrained and saved. Fraud rate in training data:", df['fraud_bool'].mean())
</parameter
</xai:function_call
import pandas as pd
import numpy as np
import random


def generate_dummy_data(num_records=2000):
    data = {
        # ... (all existing feature columns are the same) ...
        'customer_id': [f'cust_{i}' for i in range(num_records)],
        'nida': [f'nida_{1000 + i}' for i in range(num_records)],
        'age': np.random.randint(22, 65, size=num_records),
        'married': np.random.choice(['YES', 'NO'], size=num_records, p=[0.6, 0.4]),
        'education': np.random.choice(['Graduate', 'Post-Graduate', 'Undergraduate'], size=num_records),
        'dependents': np.random.randint(0, 5, size=num_records),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], size=num_records),
        'spouse_employment_status': np.random.choice(['Employed', 'Unemployed', 'Not Applicable'], size=num_records),
        'monthly_income': np.random.uniform(30000, 150000, size=num_records).round(2),
        'residense_status': np.random.choice(['Owned', 'Rented'], size=num_records),
        'vehicle_ownership_status': np.random.choice(['YES', 'NO'], size=num_records),
        'vehicle_cat': np.random.choice(['Car', 'Motorcycle', 'None'], size=num_records),
        'credit_history_length_months': np.random.randint(6, 240, size=num_records),
        'payment_history_score': np.random.randint(300, 850, size=num_records),
        'total_outstanding_debt': np.random.uniform(0, 50000, size=num_records).round(2),
        'credit_utilization_ratio': np.random.uniform(0.05, 0.95, size=num_records).round(2),
        'number_of_late_payments_36': np.random.randint(0, 10, size=num_records),
        'active_loans': np.random.randint(0, 5, size=num_records),
        'avg_monthly_balance': np.random.uniform(1000, 75000, size=num_records).round(2),
        'savings_account_balance': np.random.uniform(500, 200000, size=num_records).round(2),
        'requested_amount': np.random.uniform(5000, 100000, size=num_records).round(2),
        'loan_purpose': np.random.choice(['Home', 'Education', 'Personal', 'Business'], size=num_records),
        'previous_collateral_value': np.random.uniform(0, 250000, size=num_records).round(2),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.6, size=num_records).round(2)
    }
    df = pd.DataFrame(data)

    # --- START OF NEW TARGET GENERATION LOGIC ---
    # Create a latent "true risk" score
    true_risk = (df['debt_to_income_ratio'] * 2) + \
                (df['credit_utilization_ratio']) - \
                (df['payment_history_score'] / 1000) - \
                (df['monthly_income'] / 100000)

    # 1. Target for Risk Probability Model (is_high_risk: 0 or 1)
    df['is_high_risk'] = (true_risk > np.percentile(true_risk, 70)).astype(int)  # Top 30% are high risk

    # 2. Target for Loan Approval Model (is_approved: 0 or 1)
    df['is_approved'] = (df['payment_history_score'] > 640).astype(int)  # Simple rule for approval

    # 3. Target for Fraud Status Model (is_fraud: 0 or 1)
    # Make fraud rare and based on unusual patterns
    fraud_condition = (df['savings_account_balance'] < 1000) & (df['requested_amount'] > 90000)
    df['is_fraud'] = np.where(fraud_condition, 1,
                              (np.random.rand(num_records) < 0.02).astype(int))  # Base 2% fraud rate
    # --- END OF NEW TARGET GENERATION LOGIC ---

    df.to_csv('data/customer_data.csv', index=False)
    print("Dummy data with new target columns generated at data/customer_data.csv")


if __name__ == '__main__':
    generate_dummy_data()
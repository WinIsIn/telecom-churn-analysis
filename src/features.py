"""Feature engineering for KiwiTel churn prediction."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['charges_per_month_ratio'] = df['total_charges'] / (df['tenure_months'] + 1)
    df['is_high_value'] = (df['monthly_charges'] > 100).astype(int)
    df['support_call_bucket'] = pd.cut(
        df['num_support_calls'],
        bins=[-1, 0, 2, 4, 8],
        labels=['None', 'Low', 'Medium', 'High'],
    )

    contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
    df['contract_risk_score'] = df['contract_type'].map(contract_risk)

    # Encode categorical columns
    cat_cols = [
        'gender', 'region', 'contract_type', 'payment_method',
        'internet_service', 'has_streaming', 'has_security',
        'has_tech_support', 'support_call_bucket',
    ]
    le = LabelEncoder()
    for col in cat_cols:
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    return df


def get_feature_columns() -> list[str]:
    """Return the feature column names used for modelling."""
    return [
        'age', 'tenure_months', 'monthly_charges', 'total_charges',
        'num_support_calls', 'late_payments',
        'charges_per_month_ratio', 'is_high_value', 'contract_risk_score',
        'gender_enc', 'region_enc', 'contract_type_enc',
        'payment_method_enc', 'internet_service_enc',
        'has_streaming_enc', 'has_security_enc', 'has_tech_support_enc',
    ]

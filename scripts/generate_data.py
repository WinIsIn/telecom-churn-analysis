"""
Generate realistic NZ telecom customer dataset for KiwiTel churn analysis.
Produces data/raw/customers.csv with 2,000 rows.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 2000

regions = ['Auckland', 'Wellington', 'Christchurch', 'Hamilton', 'Other']
region_weights = [0.35, 0.20, 0.18, 0.12, 0.15]

contract_types = ['Month-to-month', 'One year', 'Two year']
contract_weights = [0.55, 0.25, 0.20]

payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
payment_weights = [0.35, 0.15, 0.25, 0.25]

internet_services = ['DSL', 'Fiber optic', 'No']
internet_weights = [0.40, 0.45, 0.15]

genders = ['Male', 'Female']

customer_id = [f'KWT-{str(i).zfill(5)}' for i in range(1, N + 1)]
age = np.random.randint(18, 75, N)
gender = np.random.choice(genders, N)
region = np.random.choice(regions, N, p=region_weights)
tenure_months = np.random.randint(1, 73, N)
monthly_charges = np.round(np.random.uniform(29, 199, N), 2)
total_charges = np.round(monthly_charges * tenure_months * np.random.uniform(0.92, 1.05, N), 2)
contract_type = np.random.choice(contract_types, N, p=contract_weights)
payment_method = np.random.choice(payment_methods, N, p=payment_weights)
internet_service = np.random.choice(internet_services, N, p=internet_weights)
has_streaming = np.random.choice(['Yes', 'No'], N, p=[0.44, 0.56])
has_security = np.random.choice(['Yes', 'No'], N, p=[0.38, 0.62])
has_tech_support = np.random.choice(['Yes', 'No'], N, p=[0.35, 0.65])
num_support_calls = np.random.choice(range(9), N, p=[0.30, 0.25, 0.18, 0.12, 0.07, 0.04, 0.02, 0.01, 0.01])
late_payments = np.random.choice(range(6), N, p=[0.50, 0.22, 0.14, 0.08, 0.04, 0.02])

# Churn probability with realistic business logic — strong signal for model learning
def compute_churn_prob(i):
    p = 0.05  # base rate

    # Contract type — month-to-month churns ~3x more (strong signal)
    if contract_type[i] == 'Month-to-month':
        p += 0.28
    elif contract_type[i] == 'One year':
        p += 0.08

    # High charges + low tenure (strong interaction)
    if monthly_charges[i] > 120 and tenure_months[i] < 12:
        p += 0.28
    elif monthly_charges[i] > 100 and tenure_months[i] < 18:
        p += 0.16
    elif monthly_charges[i] > 80 and tenure_months[i] < 6:
        p += 0.18

    # Long-tenure customers are much more loyal
    if tenure_months[i] > 48:
        p -= 0.16
    elif tenure_months[i] > 36:
        p -= 0.10
    elif tenure_months[i] > 24:
        p -= 0.06

    # Support calls (strong signal)
    if num_support_calls[i] >= 5:
        p += 0.32
    elif num_support_calls[i] >= 3:
        p += 0.22
    elif num_support_calls[i] >= 1:
        p += 0.07

    # Fiber optic churns more (competitive market)
    if internet_service[i] == 'Fiber optic':
        p += 0.12
    elif internet_service[i] == 'No':
        p -= 0.08

    # Region effects
    if region[i] in ('Auckland', 'Wellington'):
        p -= 0.05
    elif region[i] == 'Other':
        p += 0.03

    # Late payments
    if late_payments[i] >= 3:
        p += 0.18
    elif late_payments[i] >= 1:
        p += 0.07

    # Tech support / security reduce churn
    if has_tech_support[i] == 'Yes':
        p -= 0.07
    if has_security[i] == 'Yes':
        p -= 0.06

    # Electronic check payers churn more
    if payment_method[i] == 'Electronic check':
        p += 0.06

    return min(max(p, 0.01), 0.97)

churn_probs = np.array([compute_churn_prob(i) for i in range(N)])

# Rank-based assignment: top 26% by probability become churners.
# Add small noise so the boundary isn't perfectly sharp (realistic uncertainty).
target_rate = 0.26
noise = np.random.normal(0, 0.04, N)
churn_score = churn_probs + noise
threshold = np.percentile(churn_score, (1 - target_rate) * 100)
churn_raw = (churn_score >= threshold).astype(int)
churn = np.where(churn_raw == 1, 'Yes', 'No')

df = pd.DataFrame({
    'customer_id': customer_id,
    'age': age,
    'gender': gender,
    'region': region,
    'tenure_months': tenure_months,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'contract_type': contract_type,
    'payment_method': payment_method,
    'internet_service': internet_service,
    'has_streaming': has_streaming,
    'has_security': has_security,
    'has_tech_support': has_tech_support,
    'num_support_calls': num_support_calls,
    'late_payments': late_payments,
    'churn': churn,
})

out = Path(__file__).parent.parent / 'data' / 'raw' / 'customers.csv'
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(f"Generated {N} customer records -> {out}")
print(f"Churn rate: {(df['churn'] == 'Yes').mean():.1%}")
print(df['churn'].value_counts())

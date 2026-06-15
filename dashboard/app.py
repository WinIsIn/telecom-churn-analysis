"""
KiwiTel Churn Analysis — Streamlit Interactive Dashboard
Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features import engineer_features, get_feature_columns
from src.viz import (
    churn_by_contract,
    churn_by_region,
    churn_by_support_bucket,
    churn_pie,
    confusion_matrix_heatmap,
    feature_importance_bar,
    monthly_charges_box,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='KiwiTel Churn Analysis',
    layout='wide',
)

DATA_PATH = Path(__file__).parent.parent / 'data' / 'raw' / 'customers.csv'
MODEL_PATH = Path(__file__).parent.parent / 'model.pkl'

CHURN_COLOR = '#E74C3C'
RETAIN_COLOR = '#2ECC71'


# ── Data / model loaders ─────────────────────────────────────────────────────
def _ensure_data():
    """Generate customers.csv if it doesn't exist (e.g. on Streamlit Cloud)."""
    if DATA_PATH.exists():
        return
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Import and run the generation script directly — no subprocess needed
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'generate_data',
        Path(__file__).parent.parent / 'scripts' / 'generate_data.py',
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


@st.cache_data
def load_data():
    _ensure_data()
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_model_metrics(_payload, _df):
    """Recompute test-set metrics from saved model."""
    df_feat = engineer_features(_df)
    features = get_feature_columns()
    X = df_feat[features]
    y = (_df['churn'] == 'Yes').astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = _payload['model']
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    acc = (y_pred == y_test).mean()
    cm = confusion_matrix(y_test, y_pred).tolist()
    importances = dict(zip(features, rf.feature_importances_))
    top10 = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])
    return acc, auc, cm, top10


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title('KiwiTel Analytics')
st.sidebar.markdown('Customer Churn Intelligence Platform')
st.sidebar.divider()
st.sidebar.markdown('**Data:** 2,000 NZ customers  \n**Model:** Random Forest  \n**Built with:** Python + Streamlit')

# ── Load resources ────────────────────────────────────────────────────────────
df = load_data()
payload = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title('📡 KiwiTel — Customer Churn Analysis')
st.markdown('*Data-driven insights to reduce churn and protect revenue for New Zealand\'s telecom market*')
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(['📊 Overview', '🔍 Churn Drivers', '🤖 Predictive Model', '🎯 Customer Risk Lookup'])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header('Business Overview')

    total = len(df)
    churned = (df['churn'] == 'Yes').sum()
    churn_rate = churned / total * 100
    revenue_at_risk = df[df['churn'] == 'Yes']['monthly_charges'].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Customers', f'{total:,}')
    col2.metric('Churned Customers', f'{churned:,}')
    col3.metric('Churn Rate', f'{churn_rate:.1f}%', delta=f'{churn_rate - 20:.1f}% vs industry avg', delta_color='inverse')
    col4.metric('Monthly Revenue at Risk', f'NZD ${revenue_at_risk:,.0f}')

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(churn_pie(df), use_container_width=True)
    with c2:
        st.plotly_chart(churn_by_region(df), use_container_width=True)

    st.subheader('Dataset Sample')
    st.dataframe(df.head(10), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHURN DRIVERS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header('Churn Driver Analysis')
    st.markdown('Identifying the key factors that drive customer churn at KiwiTel.')

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(churn_by_contract(df), use_container_width=True)
    with c2:
        st.plotly_chart(monthly_charges_box(df), use_container_width=True)

    st.plotly_chart(churn_by_support_bucket(df), use_container_width=True)

    st.subheader('Internet Service Impact')
    grp_internet = (
        df.groupby('internet_service')['churn']
        .apply(lambda s: (s == 'Yes').mean() * 100)
        .reset_index()
        .rename(columns={'churn': 'churn_rate_pct'})
    )
    fig_inet = px.bar(
        grp_internet, x='internet_service', y='churn_rate_pct',
        color='churn_rate_pct', color_continuous_scale=['#2ECC71', '#E74C3C'],
        title='Churn Rate by Internet Service (%)',
        labels={'churn_rate_pct': 'Churn Rate (%)', 'internet_service': 'Internet Service'},
    )
    fig_inet.update_coloraxes(showscale=False)
    st.plotly_chart(fig_inet, use_container_width=True)

    st.subheader('High-Risk Segment: Month-to-Month + Fiber + Tenure < 12 months')
    seg = df[
        (df['contract_type'] == 'Month-to-month') &
        (df['internet_service'] == 'Fiber optic') &
        (df['tenure_months'] < 12)
    ]
    seg_churn = (seg['churn'] == 'Yes').mean() * 100
    seg_rev = seg[seg['churn'] == 'Yes']['monthly_charges'].sum()
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric('Segment Size', f'{len(seg):,} customers')
    sc2.metric('Segment Churn Rate', f'{seg_churn:.1f}%')
    sc3.metric('Revenue at Risk', f'NZD ${seg_rev:,.0f}/mo')


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE MODEL
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header('Predictive Model Performance')

    if payload is None:
        st.warning('Model not trained yet. Run: `python src/model.py` from the project root.')
    else:
        acc, auc, cm, top10 = get_model_metrics(payload, df)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric('Model', 'Random Forest')
        mc2.metric('Accuracy', f'{acc:.1%}')
        mc3.metric('AUC-ROC', f'{auc:.4f}', delta='> 0.75 target ✓' if auc > 0.75 else '⚠ below target')

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(feature_importance_bar(top10), use_container_width=True)
        with c2:
            st.plotly_chart(confusion_matrix_heatmap(cm), use_container_width=True)

        st.subheader('Model Notes')
        st.markdown("""
- **Algorithm:** Random Forest (200 trees, max depth 8)
- **Training split:** 80 / 20 with stratified sampling
- **Target variable:** `churn` (binary: Yes / No)
- **Features used:** 17 engineered features including contract risk score, charges ratio, and support call buckets
- **Baseline:** ~74% accuracy from predicting majority class — model must beat this
""")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — CUSTOMER RISK LOOKUP
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header('Customer Churn Risk Lookup')
    st.markdown('Enter customer details to predict churn probability and get retention recommendations.')

    if payload is None:
        st.warning('Model not trained yet. Run: `python src/model.py` from the project root.')
    else:
        with st.form('risk_form'):
            col1, col2, col3 = st.columns(3)

            with col1:
                tenure = st.slider('Tenure (months)', 1, 72, 12)
                monthly_chg = st.slider('Monthly Charges (NZD)', 29, 199, 79)
                support_calls = st.slider('Support Calls', 0, 8, 1)

            with col2:
                contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
                internet = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
                region = st.selectbox('Region', ['Auckland', 'Wellington', 'Christchurch', 'Hamilton', 'Other'])

            with col3:
                age = st.slider('Age', 18, 74, 35)
                late_pay = st.slider('Late Payments', 0, 5, 0)
                payment = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
                has_stream = st.selectbox('Has Streaming', ['No', 'Yes'])
                has_sec = st.selectbox('Has Security', ['No', 'Yes'])
                has_support = st.selectbox('Has Tech Support', ['No', 'Yes'])

            submitted = st.form_submit_button('🔍 Predict Churn Risk', use_container_width=True)

        if submitted:
            total_chg = monthly_chg * tenure * 0.98
            input_df = pd.DataFrame([{
                'customer_id': 'LOOKUP',
                'age': age,
                'gender': 'Male',
                'region': region,
                'tenure_months': tenure,
                'monthly_charges': monthly_chg,
                'total_charges': total_chg,
                'contract_type': contract,
                'payment_method': payment,
                'internet_service': internet,
                'has_streaming': has_stream,
                'has_security': has_sec,
                'has_tech_support': has_support,
                'num_support_calls': support_calls,
                'late_payments': late_pay,
                'churn': 'No',
            }])

            feat_df = engineer_features(input_df)
            features = get_feature_columns()
            X_input = feat_df[features]

            rf = payload['model']
            prob = rf.predict_proba(X_input)[0][1]
            prob_pct = prob * 100

            if prob_pct < 30:
                risk_level = 'Low'
                risk_color = '#2ECC71'
                risk_icon = '✅'
            elif prob_pct < 60:
                risk_level = 'Medium'
                risk_color = '#F39C12'
                risk_icon = '⚠️'
            else:
                risk_level = 'High'
                risk_color = '#E74C3C'
                risk_icon = '🚨'

            st.divider()
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f'### {risk_icon} Churn Risk: **{risk_level}**')
                st.markdown(f'<h2 style="color:{risk_color}">{prob_pct:.1f}% probability</h2>', unsafe_allow_html=True)

            with rc2:
                gauge = px.pie(
                    values=[prob_pct, 100 - prob_pct],
                    names=['Churn Risk', 'Safe'],
                    hole=0.65,
                    color_discrete_sequence=[risk_color, '#ECF0F1'],
                )
                gauge.update_layout(showlegend=False, height=200, margin=dict(t=0, b=0, l=0, r=0))
                gauge.update_traces(textinfo='none')
                st.plotly_chart(gauge, use_container_width=True)

            if risk_level == 'High':
                st.error('**Retention Recommendations for High-Risk Customer:**')
                recs = []
                if contract == 'Month-to-month':
                    recs.append('📋 **Offer contract upgrade incentive** — discount for switching to 1 or 2-year plan')
                if support_calls >= 3:
                    recs.append('📞 **Proactive service call** — assign a dedicated account manager to resolve outstanding issues')
                if internet == 'Fiber optic' and monthly_chg > 100:
                    recs.append('💰 **Loyalty pricing review** — offer a 10-15% discount on fiber plan to match competitor pricing')
                if tenure < 12:
                    recs.append('🎁 **New customer loyalty bundle** — add free security or streaming package for 3 months')
                if late_pay >= 2:
                    recs.append('💳 **Payment plan restructure** — offer flexible billing date or direct debit discount')
                recs.append('📊 **Executive churn dashboard alert** — flag this customer for monthly retention review')
                for r in recs:
                    st.markdown(f'- {r}')
            elif risk_level == 'Medium':
                st.warning('**Suggested Actions for Medium-Risk Customer:**')
                st.markdown('- 📧 Send satisfaction survey and loyalty reward offer')
                st.markdown('- 📱 Enrol in proactive support notifications')
            else:
                st.success('Customer appears stable. Continue standard engagement cadence.')

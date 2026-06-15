"""Reusable Plotly chart functions for KiwiTel churn analysis."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

CHURN_COLOR = '#E74C3C'
RETAIN_COLOR = '#2ECC71'
COLOR_MAP = {'Yes': CHURN_COLOR, 'No': RETAIN_COLOR}


def churn_pie(df: pd.DataFrame) -> go.Figure:
    counts = df['churn'].value_counts().reset_index()
    counts.columns = ['churn', 'count']
    fig = px.pie(
        counts, values='count', names='churn',
        color='churn', color_discrete_map=COLOR_MAP,
        title='Churn vs Retained',
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def churn_by_region(df: pd.DataFrame) -> go.Figure:
    grp = (
        df.groupby('region')['churn']
        .apply(lambda s: (s == 'Yes').mean() * 100)
        .reset_index()
        .rename(columns={'churn': 'churn_rate_pct'})
        .sort_values('churn_rate_pct', ascending=True)
    )
    fig = px.bar(
        grp, x='churn_rate_pct', y='region', orientation='h',
        title='Churn Rate by Region (%)',
        color='churn_rate_pct', color_continuous_scale=['#2ECC71', '#E74C3C'],
        labels={'churn_rate_pct': 'Churn Rate (%)', 'region': 'Region'},
    )
    fig.update_coloraxes(showscale=False)
    return fig


def churn_by_contract(df: pd.DataFrame) -> go.Figure:
    grp = (
        df.groupby('contract_type')['churn']
        .apply(lambda s: (s == 'Yes').mean() * 100)
        .reset_index()
        .rename(columns={'churn': 'churn_rate_pct'})
        .sort_values('churn_rate_pct', ascending=False)
    )
    fig = px.bar(
        grp, x='contract_type', y='churn_rate_pct',
        title='Churn Rate by Contract Type (%)',
        color='churn_rate_pct', color_continuous_scale=['#2ECC71', '#E74C3C'],
        labels={'churn_rate_pct': 'Churn Rate (%)', 'contract_type': 'Contract Type'},
    )
    fig.update_coloraxes(showscale=False)
    return fig


def monthly_charges_box(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x='churn', y='monthly_charges',
        color='churn', color_discrete_map=COLOR_MAP,
        title='Monthly Charges Distribution (NZD): Churned vs Retained',
        labels={'monthly_charges': 'Monthly Charges (NZD)', 'churn': 'Churned'},
    )
    return fig


def churn_by_support_bucket(df: pd.DataFrame) -> go.Figure:
    bucket_order = ['0 calls', '1-2 calls', '3-4 calls', '5+ calls']

    def bucket(n):
        if n == 0:
            return '0 calls'
        elif n <= 2:
            return '1-2 calls'
        elif n <= 4:
            return '3-4 calls'
        return '5+ calls'

    tmp = df.copy()
    tmp['support_bucket'] = tmp['num_support_calls'].apply(bucket)
    grp = (
        tmp.groupby('support_bucket')['churn']
        .apply(lambda s: (s == 'Yes').mean() * 100)
        .reindex(bucket_order)
        .reset_index()
        .rename(columns={'churn': 'churn_rate_pct'})
    )
    fig = px.bar(
        grp, x='support_bucket', y='churn_rate_pct',
        title='Churn Rate by Support Calls Bucket (%)',
        color='churn_rate_pct', color_continuous_scale=['#2ECC71', '#E74C3C'],
        labels={'churn_rate_pct': 'Churn Rate (%)', 'support_bucket': 'Support Calls'},
        category_orders={'support_bucket': bucket_order},
    )
    fig.update_coloraxes(showscale=False)
    return fig


def feature_importance_bar(importances: dict) -> go.Figure:
    items = sorted(importances.items(), key=lambda x: x[1])
    labels, values = zip(*items)
    fig = go.Figure(go.Bar(
        x=list(values), y=list(labels), orientation='h',
        marker_color='#3498DB',
    ))
    fig.update_layout(title='Top 10 Feature Importances (Random Forest)',
                      xaxis_title='Importance', yaxis_title='Feature')
    return fig


def confusion_matrix_heatmap(cm: list[list[int]]) -> go.Figure:
    labels = ['Retained', 'Churned']
    fig = px.imshow(
        cm, x=labels, y=labels,
        color_continuous_scale='RdYlGn_r',
        title='Confusion Matrix',
        labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
        text_auto=True,
    )
    return fig

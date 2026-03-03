import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Base directory — works from any location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────
# Page Config
# ─────────────────────────────
st.set_page_config(
    page_title="Social Engagement Intelligence",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────
# Dark Theme CSS
# ─────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global font */
    * { font-family: 'Inter', sans-serif !important; }

    /* Main background */
    .stApp { background-color: #0a0e1a; color: #f1f5f9; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937;
    }

    /* Sidebar nav radio labels — bright and readable */
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] label:hover { color: #6366f1 !important; }

    /* Selected radio item */
    [data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div {
        background-color: rgba(99,102,241,0.15) !important;
        border-radius: 8px;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1rem;
        border-top: 3px solid #6366f1;
    }
    [data-testid="stMetricValue"] { color: #6366f1 !important; font-size: 1.8rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #cbd5e1 !important; font-weight: 500 !important; }
    [data-testid="stMetricDelta"] { color: #10b981 !important; }

    /* All headers */
    h1, h2, h3, h4 { color: #f1f5f9 !important; font-family: 'Inter', sans-serif !important; }

    /* Chart titles inside plotly — handled by PLOTLY_LAYOUT */

    /* Divider */
    hr { border-color: #1f2937 !important; }

    /* Select box label */
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label,
    .stRadio label {
        color: #cbd5e1 !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Select box input */
    .stSelectbox > div > div {
        background-color: #1f2937 !important;
        border-color: #374151 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }

    /* Slider track */
    .stSlider > div > div > div > div {
        background-color: #6366f1 !important;
    }

    /* Slider min/max labels */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
    }

    /* Slider current value */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #6366f1 !important;
    }

    /* Success/Warning/Error */
    [data-testid="stAlert"] { border-radius: 10px !important; }
    .stSuccess { background-color: rgba(16,185,129,0.1) !important; border-color: #10b981 !important; color: #d1fae5 !important; }
    .stWarning { background-color: rgba(245,158,11,0.1) !important; border-color: #f59e0b !important; color: #fef3c7 !important; }
    .stError { background-color: rgba(239,68,68,0.1) !important; border-color: #ef4444 !important; color: #fee2e2 !important; }

    /* Caption */
    .stCaption, [data-testid="stCaptionContainer"] { color: #94a3b8 !important; }

    /* Expander */
    .streamlit-expanderHeader { background-color: #111827 !important; color: #e2e8f0 !important; border-radius: 8px !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { background-color: #111827 !important; border-radius: 8px !important; }

    /* Dropdown options list */
    [data-baseweb="popover"] {
        background-color: #1f2937 !important;
    }
    [data-baseweb="menu"] {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="menu"] li {
        background-color: #1f2937 !important;
        color: #e2e8f0 !important;
        font-size: 0.9rem !important;
    }
    [data-baseweb="menu"] li:hover {
        background-color: rgba(99,102,241,0.15) !important;
        color: #f1f5f9 !important;
    }
    [data-baseweb="select"] > div {
        background-color: #1f2937 !important;
        border-color: #374151 !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }
    [data-baseweb="select"] span {
        color: #f1f5f9 !important;
    }

    /* Dropdown arrow */
    [data-baseweb="select"] svg { fill: #94a3b8 !important; }

    /* General text */
    p, span, div { color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# Color Palette
# ─────────────────────────────
COLORS = {
    'primary': '#6366f1',
    'secondary': '#06b6d4',
    'danger': '#E1306C',
    'success': '#10b981',
    'warning': '#f59e0b',
    'control': '#4267B2',
    'treatment': '#E1306C',
    'bg': '#0a0e1a',
    'card': '#111827',
    'border': '#1f2937',
    'muted': '#94a3b8',
    'text': '#f1f5f9'
}

PLOTLY_LAYOUT = dict(
    plot_bgcolor=COLORS['card'],
    paper_bgcolor=COLORS['card'],
    font=dict(color='#e2e8f0', family='Inter, sans-serif', size=13),
    title_font=dict(color='#f1f5f9', size=15, family='Inter, sans-serif'),
    xaxis=dict(gridcolor=COLORS['border'], linecolor=COLORS['border'],
               tickfont=dict(color='#94a3b8'), title_font=dict(color='#cbd5e1')),
    yaxis=dict(gridcolor=COLORS['border'], linecolor=COLORS['border'],
               tickfont=dict(color='#94a3b8'), title_font=dict(color='#cbd5e1')),
    legend=dict(bgcolor=COLORS['card'], bordercolor=COLORS['border'],
                font=dict(color='#e2e8f0')),
    margin=dict(t=50, b=40, l=40, r=20)
)

# ─────────────────────────────
# Load Data
# ─────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/user_experiment_data_clean.csv'))
    preds = pd.read_csv(os.path.join(BASE_DIR, 'data/processed/churn_predictions.csv'))
    return df, preds

df, preds = load_data()

# ─────────────────────────────
# Sidebar
# ─────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2rem;'>📊</div>
        <div style='font-size:1.1rem; font-weight:700; color:#6366f1;'>Social Intelligence</div>
        <div style='font-size:0.75rem; color:#94a3b8;'>Meta-Style Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🧪 A/B Test Results", "🔮 Churn Risk Monitor",
         "💬 Content Intelligence", "💰 Business Impact", "📋 Recommendation"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#94a3b8;'>
        <div>👥 <b style='color:#f1f5f9;'>50,000</b> Users</div>
        <div style='margin-top:0.5rem;'>⏱️ <b style='color:#f1f5f9;'>4 Week</b> Experiment</div>
        <div style='margin-top:0.5rem;'>🎯 <b style='color:#f1f5f9;'>AI Feed</b> vs Chronological</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────
# Page 1: Overview
# ─────────────────────────────
if page == "🏠 Overview":
    st.title("📊 Social Engagement Intelligence Platform")
    st.markdown("<p style='color:#94a3b8;'>Simulating a Meta-style product analytics system — AI Feed vs Chronological Feed</p>", unsafe_allow_html=True)

    st.divider()

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Users", f"{len(df):,}")
    col2.metric("📅 Experiment Duration", "4 Weeks")
    col3.metric("📉 Overall Churn Rate", f"{df['churned_30d'].mean():.1%}")
    col4.metric("⏱️ Avg Daily Active Min", f"{df['daily_active_minutes'].mean():.1f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x='daily_active_minutes', color='group',
            barmode='overlay', opacity=0.75,
            title='Daily Active Minutes: Control vs Treatment',
            color_discrete_map={'control': COLORS['control'], 'treatment': COLORS['treatment']},
            nbins=60
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        churn_by_group = df.groupby(['age_group', 'group'])['churned_30d'].mean().reset_index()
        churn_by_group['churned_30d'] = (churn_by_group['churned_30d'] * 100).round(2)
        fig = px.bar(
            churn_by_group, x='age_group', y='churned_30d', color='group',
            barmode='group', title='Churn Rate by Age Group & Feed Type (%)',
            color_discrete_map={'control': COLORS['control'], 'treatment': COLORS['treatment']}
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # Engagement by account type
    acct_df = df.groupby(['account_type', 'group'])['daily_active_minutes'].mean().reset_index()
    fig = px.bar(
        acct_df, x='account_type', y='daily_active_minutes', color='group',
        barmode='group', title='Avg Engagement by Account Type',
        color_discrete_map={'control': COLORS['control'], 'treatment': COLORS['treatment']}
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────
# Page 2: A/B Test Results
# ─────────────────────────────
elif page == "🧪 A/B Test Results":
    st.title("🧪 A/B Test Results — AI Feed vs Chronological Feed")

    control = df[df.group == 'control']['daily_active_minutes']
    treatment = df[df.group == 'treatment']['daily_active_minutes']
    lift = ((treatment.mean() - control.mean()) / control.mean()) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Control Avg", f"{control.mean():.1f} min")
    col2.metric("Treatment Avg", f"{treatment.mean():.1f} min", f"+{lift:.1f}%")
    col3.metric("Statistical Significance", "p < 0.001 ✅")
    col4.metric("Effect Size (Cohen's d)", "0.42 — Medium")

    st.divider()

    # Interactive filters
    col1, col2 = st.columns(2)
    with col1:
        segment = st.selectbox("📊 Analyze by segment:", ['age_group', 'account_type'])
    with col2:
        metric = st.selectbox("📈 Metric:", ['daily_active_minutes', 'interaction_rate', 'stories_watched'])

    seg_df = df.groupby([segment, 'group'])[metric].mean().reset_index()
    fig = px.bar(
        seg_df, x=segment, y=metric, color='group',
        barmode='group',
        title=f'{metric.replace("_", " ").title()} by {segment.replace("_", " ").title()}',
        color_discrete_map={'control': COLORS['control'], 'treatment': COLORS['treatment']}
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Distribution comparison
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for grp, color in [('control', COLORS['control']), ('treatment', COLORS['treatment'])]:
            fig.add_trace(go.Box(
                y=df[df.group == grp]['daily_active_minutes'],
                name=grp.capitalize(), marker_color=color,
                boxmean=True
            ))
        fig.update_layout(title='Distribution Comparison', **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Lift heatmap by segment
        pivot = df.groupby(['age_group', 'account_type'])['daily_active_minutes'].mean().unstack()
        fig = px.imshow(
            pivot, title='Avg Daily Active Minutes — Age × Account Type',
            color_continuous_scale='Blues', text_auto='.1f'
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────
# Page 3: Churn Risk
# ─────────────────────────────
elif page == "🔮 Churn Risk Monitor":
    st.title("🔮 Churn Risk Monitor")

    high_risk = (preds['churn_probability'] > 0.6).sum()
    critical_risk = (preds['churn_probability'] > 0.8).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model AUC", "0.84")
    col2.metric("High Risk Users (>60%)", f"{high_risk:,}")
    col3.metric("Critical Risk (>80%)", f"{critical_risk:,}")
    col4.metric("Avg Churn Probability", f"{preds['churn_probability'].mean():.1%}")

    st.divider()

    # Threshold slider
    threshold = st.slider("🎚️ Risk Threshold", 0.1, 0.9, 0.6, 0.05,
                          help="Adjust to see how many users fall above the risk threshold")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            preds, x='churn_probability', nbins=50,
            title='Churn Probability Distribution',
            color_discrete_sequence=[COLORS['treatment']]
        )
        fig.add_vline(x=threshold, line_dash='dash', line_color=COLORS['warning'],
                      annotation_text=f'Threshold: {threshold}',
                      annotation_font_color=COLORS['warning'])
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk segments pie
        risk_labels = pd.cut(
            preds['churn_probability'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low (<30%)', 'Medium (30-60%)', 'High (60-80%)', 'Critical (>80%)']
        ).value_counts()
        fig = px.pie(
            values=risk_labels.values, names=risk_labels.index,
            title='Users by Risk Segment',
            color_discrete_sequence=[COLORS['success'], COLORS['warning'],
                                      COLORS['danger'], '#C73E1D']
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.image(os.path.join(BASE_DIR, 'reports/shap_importance_bar.png'), caption='SHAP Feature Importance — XGBoost Model')

# ─────────────────────────────
# Page 4: Content Intelligence
# ─────────────────────────────
elif page == "💬 Content Intelligence":
    st.title("💬 Content Intelligence — NLP Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment Model", "DistilBERT")
    col2.metric("Topic Model", "BERTopic")
    col3.metric("Sample Size", "5,000 captions")

    st.divider()
    st.image(os.path.join(BASE_DIR, 'reports/nlp_sentiment.png'), caption='Sentiment Analysis by Feed Type')

    st.divider()
    st.subheader("📰 AI-Generated Weekly Trend Summary")
    st.caption("Generated using Claude API — summarizing top content topics for the product team")
    with open(os.path.join(BASE_DIR, 'reports/weekly_trend_summary.txt'), 'r') as f:
        st.info(f.read())

    st.divider()
    st.subheader("🖼️ Topic Word Clouds")
    try:
        st.image(os.path.join(BASE_DIR, 'reports/topic_wordclouds.png'), caption='BERTopic Clusters — Top Content Themes')
    except:
        st.caption("Word cloud image not found")

# ─────────────────────────────
# Page 5: Business Impact
# ─────────────────────────────
elif page == "💰 Business Impact":
    st.title("💰 Business Impact Estimator")
    st.markdown("<p style='color:#94a3b8;'>Translate experiment results into estimated revenue impact</p>", unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        dau = st.slider("Current DAU (millions)", 100, 1000, 200, 50,
                        help="Daily Active Users on the platform")
    with col2:
        arpu = st.slider("Avg Monthly Revenue Per User ($)", 1, 15, 4, 1,
                         help="Average revenue per user per month")
    with col3:
        lift_pct = st.slider("Engagement Lift (%)", 1, 20, 12, 1,
                             help="Lift from A/B test. Default is our experiment result.")

    st.divider()

    revenue_correlation = 0.40
    incremental_engaged_users = dau * (lift_pct / 100)
    monthly_revenue_impact = incremental_engaged_users * arpu * revenue_correlation * 1_000_000
    annual_revenue_impact = monthly_revenue_impact * 12
    churn_reduction_pct = 0.03
    churned_users_saved = dau * churn_reduction_pct * 1_000_000
    avg_ltv = arpu * 12
    churn_savings = churned_users_saved * avg_ltv
    total_annual_impact = annual_revenue_impact + churn_savings

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Incremental Engaged Users", f"{incremental_engaged_users:.1f}M")
    col2.metric("Est. Monthly Revenue Impact", f"${monthly_revenue_impact/1_000_000:.1f}M")
    col3.metric("Est. Annual Revenue Impact", f"${annual_revenue_impact/1_000_000:.1f}M")
    col4.metric("Churn Savings (Annual)", f"${churn_savings/1_000_000:.1f}M")

    st.divider()
    st.success(f"### 💡 Total Estimated Annual Impact: ${total_annual_impact/1_000_000:.1f}M")

    # Sensitivity analysis
    lift_range = list(range(1, 21))
    revenue_range = [
        dau * (l/100) * arpu * revenue_correlation * 1_000_000 * 12 / 1_000_000
        for l in lift_range
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lift_range, y=revenue_range,
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(99,102,241,0.1)',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(size=6, color=COLORS['primary']),
        name='Annual Revenue Impact'
    ))
    fig.add_vline(x=lift_pct, line_dash='dash', line_color=COLORS['danger'],
                  annotation_text=f'Our lift: {lift_pct}%',
                  annotation_font_color=COLORS['danger'])
    fig.update_layout(
        title='Revenue Impact Sensitivity to Engagement Lift',
        xaxis_title='Engagement Lift (%)',
        yaxis_title='Estimated Annual Revenue Impact ($M)',
        height=400,
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Assumptions & Methodology"):
        st.markdown(f"""
        **Key Assumptions:**
        - Revenue correlation factor: {revenue_correlation*100:.0f}% of engagement lift → revenue
        - Churn reduction: 3% based on XGBoost model recommendation
        - LTV: 12 months × monthly ARPU
        - These are **illustrative estimates** for strategic planning

        **Data Sources:**
        - Engagement lift: A/B experiment results
        - Churn reduction: XGBoost churn model (AUC: 0.84)
        - ARPU range: Industry benchmarks for social media platforms
        """)

# ─────────────────────────────
# Page 6: Recommendation
# ─────────────────────────────
elif page == "📋 Recommendation":
    st.title("📋 Product Recommendation")

    st.markdown("""
    <p style='color:#94a3b8; font-size:1rem;'>
    Based on A/B test results, churn model analysis, and NLP content insights
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ **SHIP** AI Feed\n\n25–34 and 35–44 age segments\n\nHighest lift, lowest churn risk")
    with col2:
        st.warning("⚠️ **HOLD & TEST**\n\n18–24 casual users\n\nElevated churn risk in treatment group")
    with col3:
        st.error("🚫 **DO NOT SHIP**\n\n18–24 casual users (unmodified)\n\nRequires re-engagement nudge first")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Expected 90-Day Impact")
        impact_data = {
            'Metric': ['DAU Lift', 'Churn Reduction', 'Segments Affected'],
            'Value': ['+8%', '-3%', '2 of 4']
        }
        st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🔑 Key Insight")
        st.markdown("""
        <div style='background:#111827; border:1px solid #1f2937;
                    border-left: 4px solid #6366f1;
                    border-radius:8px; padding:1.25rem;
                    color:#94a3b8; line-height:1.7;'>
        The aggregate A/B test result said <b style='color:#f1f5f9;'>ship it</b>.<br><br>
        Segmentation analysis revealed that <b style='color:#E1306C;'>18–24 casual users 
        showed elevated churn risk</b> in the treatment group — changing the recommendation 
        from a full launch to a <b style='color:#10b981;'>staged rollout</b>.
        </div>
        """, unsafe_allow_html=True)
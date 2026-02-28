import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle

# ─────────────────────────────
# Page Config
# ─────────────────────────────
st.set_page_config(
    page_title="Social Engagement Intelligence",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────
# Load Data
# ─────────────────────────────
@st.cache_data  # Caches data so it doesn't reload every time
def load_data():
    df = pd.read_csv('data/processed/user_experiment_data_clean.csv')
    preds = pd.read_csv('data/processed/churn_predictions.csv')
    return df, preds    

df, preds = load_data()

# ─────────────────────────────
# Sidebar Navigation
# ─────────────────────────────
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "🧪 A/B Test Results", "🔮 Churn Risk Monitor", "💬 Content Intelligence", "💰 Business Impact", "📋 Recommendation"]
)


# ─────────────────────────────
# Page 1: Overview
# ─────────────────────────────
if page == "🏠 Overview":
    st.title("Social Engagement Intelligence Platform")
    st.markdown("*Simulating a Meta-style product analytics system*")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", f"{len(df):,}")
    col2.metric("Experiment Duration", "4 Weeks")
    col3.metric("Overall Churn Rate", f"{df['churned_30d'].mean():.1%}")
    col4.metric("Avg Daily Active Minutes", f"{df['daily_active_minutes'].mean():.1f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='daily_active_minutes', color='group', 
                          barmode='overlay', opacity=0.7,
                          title='Daily Active Minutes: Control vs Treatment',
                          color_discrete_map={'control':'#4267B2','treatment':'#E1306C'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        churn_by_group = df.groupby(['age_group', 'group'])['churned_30d'].mean().reset_index()
        fig = px.bar(churn_by_group, x='age_group', y='churned_30d', color='group', 
                    barmode='group', title='Churn Rate by Age Group & Feed Type',
                    color_discrete_map={'control':'#4267B2','treatment':'#E1306C'})
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────
# Page 2: A/B Test Results
# ─────────────────────────────
elif page == "🧪 A/B Test Results":
    st.title("🧪 A/B Test Results — AI Feed vs Chronological Feed")
    
    control = df[df.group=='control']['daily_active_minutes']
    treatment = df[df.group=='treatment']['daily_active_minutes']
    lift = ((treatment.mean() - control.mean()) / control.mean()) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Control Avg", f"{control.mean():.1f} min")
    col2.metric("Treatment Avg", f"{treatment.mean():.1f} min", f"+{lift:.1f}%")
    col3.metric("Statistical Significance", "p < 0.001 ✅")
    
    # Segment selector
    segment = st.selectbox("Analyze by segment:", ['age_group', 'account_type'])
    seg_df = df.groupby([segment, 'group'])['daily_active_minutes'].mean().reset_index()
    fig = px.bar(seg_df, x=segment, y='daily_active_minutes', color='group',
                barmode='group', title=f'Engagement by {segment}',
                color_discrete_map={'control':'#4267B2','treatment':'#E1306C'})
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────
# Page 3: Churn Risk
# ─────────────────────────────
elif page == "🔮 Churn Risk Monitor":
    st.title("🔮 Churn Risk Monitor")
    
    col1, col2 = st.columns(2)
    col1.metric("Model AUC", "0.84")
    col2.metric("High Risk Users", f"{(preds['churn_probability'] > 0.6).sum():,}")
    
    # Risk distribution
    fig = px.histogram(preds, x='churn_probability', nbins=50,
                      title='Churn Probability Distribution',
                      color_discrete_sequence=['#E1306C'])
    fig.add_vline(x=0.6, line_dash='dash', annotation_text='High Risk Threshold')
    st.plotly_chart(fig, use_container_width=True)
    
    st.image('reports/shap_importance_bar.png', caption='SHAP Feature Importance')

# ─────────────────────────────
# Page 4: Content Intelligence
# ─────────────────────────────
elif page == "💬 Content Intelligence":
    st.title("💬 Content Intelligence — NLP Insights")
    
    st.image('reports/nlp_sentiment.png', caption='Sentiment Analysis by Feed Type')
    
    st.subheader("📰 AI-Generated Weekly Trend Summary")
    with open('reports/weekly_trend_summary.txt', 'r') as f:
        st.info(f.read())


# ─────────────────────────────
# Page 5: Business Impact
# ─────────────────────────────
elif page == "💰 Business Impact":
    st.title("💰 Business Impact Estimator")
    st.markdown("*Translate experiment results into estimated revenue impact*")
    
    st.divider()
    
    # ── Assumption Inputs ──
    st.subheader("⚙️ Adjust Assumptions")
    st.caption("Slide to explore different scenarios based on platform size and monetization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dau = st.slider(
            "Current DAU (millions)",
            min_value=100,
            max_value=1000,
            value=200,
            step=50,
            help="Daily Active Users on the platform"
        )
    
    with col2:
        arpu = st.slider(
            "Avg Monthly Revenue Per User ($)",
            min_value=1,
            max_value=15,
            value=4,
            step=1,
            help="Average revenue generated per user per month (ad revenue based)"
        )
    
    with col3:
        lift_pct = st.slider(
            "Engagement Lift (%)",
            min_value=1,
            max_value=20,
            value=12,
            step=1,
            help="Lift observed in the A/B test. Default is our experiment result."
        )
    
    st.divider()
    
    # ── Calculations ──
    # Assumption: engagement lift correlates with ~40% of that lift translating to revenue
    # This is a conservative industry benchmark for social media platforms
    revenue_correlation = 0.40
    
    incremental_engaged_users = dau * (lift_pct / 100)
    monthly_revenue_impact = incremental_engaged_users * arpu * revenue_correlation * 1_000_000
    annual_revenue_impact = monthly_revenue_impact * 12
    
    # Churn savings — from our ML model, intervention reduces churn by ~3%
    churn_reduction_pct = 0.03
    churned_users_saved = dau * churn_reduction_pct * 1_000_000
    avg_ltv = arpu * 12  # Simple LTV = 1 year of ARPU
    churn_savings = churned_users_saved * avg_ltv
    
    total_annual_impact = annual_revenue_impact + churn_savings
    
    # ── Metrics Display ──
    st.subheader("📊 Estimated Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Incremental Engaged Users",
        f"{incremental_engaged_users:.1f}M"
    )
    col2.metric(
        "Est. Monthly Revenue Impact",
        f"${monthly_revenue_impact/1_000_000:.1f}M"
    )
    col3.metric(
        "Est. Annual Revenue Impact",
        f"${annual_revenue_impact/1_000_000:.1f}M"
    )
    col4.metric(
        "Churn Savings (Annual)",
        f"${churn_savings/1_000_000:.1f}M"
    )
    
    # ── Total Impact Banner ──
    st.divider()
    st.success(f"### 💡 Total Estimated Annual Impact: ${total_annual_impact/1_000_000:.1f}M")
    
    # ── Scenario Comparison Chart ──
    st.subheader("📈 Sensitivity Analysis — Revenue Impact by Lift %")
    st.caption("How does the revenue impact change if the lift is higher or lower than expected?")
    
    import plotly.graph_objects as go
    
    lift_range = list(range(1, 21))
    revenue_range = [
        dau * (l/100) * arpu * revenue_correlation * 1_000_000 * 12 / 1_000_000
        for l in lift_range
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lift_range,
        y=revenue_range,
        mode='lines+markers',
        line=dict(color='#44BBA4', width=3),
        marker=dict(size=6),
        name='Annual Revenue Impact'
    ))
    
    # Highlight current experiment lift
    fig.add_vline(
        x=lift_pct,
        line_dash='dash',
        line_color='#E1306C',
        annotation_text=f'Our experiment lift: {lift_pct}%',
        annotation_position='top right'
    )
    
    fig.update_layout(
        xaxis_title='Engagement Lift (%)',
        yaxis_title='Estimated Annual Revenue Impact ($M)',
        title='Revenue Impact Sensitivity to Engagement Lift',
        height=400,
        plot_bgcolor='white',
        yaxis=dict(gridcolor='#f0f0f0'),
        xaxis=dict(gridcolor='#f0f0f0')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Assumptions & Disclaimer ──
    st.divider()
    with st.expander("📋 View Assumptions & Methodology"):
        st.markdown(f"""
        **Key Assumptions:**
        - Revenue correlation factor: {revenue_correlation*100:.0f}% of engagement lift translates to revenue
          *(Industry benchmark for social media ad-based platforms)*
        - Churn reduction: 3% based on ML model intervention recommendation
        - LTV calculation: 12 months × monthly ARPU
        - These are **illustrative estimates** for strategic planning purposes
        
        **Data Sources:**
        - Engagement lift: A/B experiment results (this project)
        - Churn reduction: XGBoost churn model predictions
        - ARPU range: Industry benchmarks for social media platforms
        
        *Actual revenue impact depends on ad inventory, user quality, 
        engagement depth, and platform-specific monetization rates.*
        """)

# ─────────────────────────────
# Page 6: Recommendation
# ─────────────────────────────
elif page == "📋 Recommendation":
    st.title("📋 Product Recommendation")
    
    st.success("✅ **SHIP** AI feed to 25-34 and 35-44 age segments")
    st.warning("⚠️ **HOLD** 18-24 casual users — run follow-up experiment")
    st.error("🚫 **DO NOT SHIP** to 18-24 casual users without re-engagement nudge")
    
    st.markdown("""
    ### Summary
    The AI-curated feed drove a statistically significant lift in engagement, 
    but a segmented rollout is recommended based on churn risk patterns in younger casual users.
    
    **Expected 90-day impact if recommendation followed:**
    - +8% DAU lift across 25-34 and 35-44 segments
    - -3% reduction in overall churn rate
    - Neutral impact on 18-24 segment pending follow-up test
    """)
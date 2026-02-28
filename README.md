# Social Engagement Intelligence Platform

A Meta-style end-to-end product analytics project simulating a feed algorithm A/B test.

## 🎯 Business Question
Did an AI-curated "For You" feed improve user engagement vs. a chronological feed?

## 📊 What's Inside
- **A/B Testing**: Power analysis, sanity checks, significance testing, segmentation
- **ML Model**: XGBoost churn prediction (AUC: 0.84) with SHAP explainability
- **NLP/GenAI**: Sentiment analysis + BERTopic + Claude-generated trend summaries
- **Dashboard**: Interactive Streamlit app with 5 pages

## 🛠️ Tech Stack
Python | XGBoost | HuggingFace | BERTopic | SHAP | Streamlit | Plotly | Claude API

## 🚀 Run It Locally
bash
git clone https://github.com/YOUR_USERNAME/social-engagement-intelligence
cd social-engagement-intelligence
pip install -r requirements.txt
streamlit run dashboard/app.py

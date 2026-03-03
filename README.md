# 📊 Social Engagement Intelligence Platform
### *A Meta-style end-to-end product analytics case study*

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)

---

## 🔗 Live Dashboard
👉 **[View the deployed dashboard here](https://social-engagement-intelligence.streamlit.app/)**

---

## 🎯 Business Question

> *"Instagram is testing an AI-curated 'For You' feed vs. a chronological feed across 50,000 users. Did it improve engagement — and should we ship it to everyone?"*

---

## 🔍 Key Finding

The AI feed drove a statistically significant **+12% lift** in daily active minutes — but segmentation revealed that **18–24 casual users showed elevated churn risk** in the treatment group. This changed the recommendation from a full launch to a **staged rollout**, saving an estimated millions in potential churn losses.

---

## 🗂️ Project Modules

### 1. 🧪 A/B Testing & Experimentation
- Power analysis to confirm experiment was sufficiently powered
- Randomization sanity checks (pre-experiment metric comparison)
- Two-sample t-test + chi-square on 5 metrics
- **Novelty effect check** — weekly lift analysis to confirm the effect is lasting
- **Multiple testing correction** — Bonferroni + Benjamini-Hochberg
- Segmentation by age group and account type

### 2. 🤖 Churn Prediction (ML)
- Feature engineering — engagement drop, recency score, user tenure
- Class imbalance handled with SMOTE
- **Logistic Regression baseline** vs **XGBoost** — justified model complexity
- **SHAP explainability** — feature importance for PM conversations
- **Calibration curve + Platt Scaling** — verified probability trustworthiness
- AUC-ROC: 0.84

### 3. 💬 NLP & GenAI
- Sentiment analysis using HuggingFace DistilBERT
- BERTopic topic modeling — discovered content clusters unsupervised
- **Word clouds** per topic cluster for visual presentation
- Topic distribution comparison: control vs. treatment feed
- Claude API integration — auto-generated weekly trend summary

### 4. 📊 Streamlit Dashboard (5 pages)
- Overview — key experiment metrics at a glance
- A/B Test Results — interactive segment explorer
- Churn Risk Monitor — model predictions + SHAP plots
- Content Intelligence — sentiment + topic analysis + AI trend summary
- **💰 Business Impact Estimator** — interactive revenue calculator with sensitivity analysis
- Final Product Recommendation

---

## 💡 Final Recommendation

| Segment | Decision | Rationale |
|---|---|---|
| 25–34 age group | ✅ Ship AI feed | Highest engagement lift, low churn risk |
| 35–44 age group | ✅ Ship AI feed | Stable lift, strong retention |
| 18–24 casual users | ⚠️ Hold | Elevated churn risk in treatment group |
| Business accounts | ✅ Ship AI feed | Positive lift, no guardrail violations |

**Estimated 90-day impact (if recommendation followed):**
- +8% DAU lift across eligible segments
- -3% overall churn rate reduction
- Significant estimated annual revenue impact

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Data & Analysis | Python, Pandas, NumPy |
| Statistics | SciPy, Statsmodels |
| Machine Learning | Scikit-learn, XGBoost, SHAP, imbalanced-learn |
| NLP | HuggingFace Transformers, BERTopic, WordCloud |
| GenAI | Anthropic Claude API |
| Visualization | Plotly, Seaborn, Matplotlib |
| Dashboard | Streamlit |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
social-engagement-intelligence/
│
├── data/
│   ├── raw/                        # Original synthetic dataset
│   └── processed/                  # Cleaned data + model predictions
│
├── notebooks/
│   ├── 01_data_generation.ipynb    # 50K user synthetic dataset
│   ├── 02_ab_testing.ipynb         # Full experiment analysis
│   ├── 03_churn_model.ipynb        # ML churn prediction
│   └── 04_nlp_sentiment.ipynb      # NLP + GenAI analysis
│
├── dashboard/
│   └── app.py                      # Streamlit dashboard
│
├── reports/                        # All generated charts and summaries
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/sakshiraade/social-engagement-intelligence.git
cd social-engagement-intelligence

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
cd dashboard
streamlit run app.py
```

---

## 📓 Run the Notebooks

Open VS Code → select **"Social Intelligence Project"** kernel → run notebooks in order:
1. `01_data_generation.ipynb`
2. `02_ab_testing.ipynb`
3. `03_churn_model.ipynb`
4. `04_nlp_sentiment.ipynb`

---

## 👩‍💻 About

Built by **Sakshi Aade** as a portfolio project simulating a Meta-style product analytics workflow.

🔗 [LinkedIn](https://www.linkedin.com/in/sakshi-aade/) | 📧 sakshiaade03@gmail.com

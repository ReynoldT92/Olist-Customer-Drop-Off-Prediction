# Olist Marketplace Integrity Audit

**Predicting First-Time Customer Drop-Off for Brazilian E-Commerce**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

---

## 📊 Project Overview

**Problem:** 95% of Olist's first-time customers never make a second purchase, representing a massive retention opportunity.

**Solution:** Built a calibrated machine learning model to predict customer drop-off risk and identify high-value retention opportunities.

**Business Impact:**
- Identifies customers with **81% retention probability** (16x better than baseline)
- Enables targeted interventions with **758% ROI** for high-retention segments
- Provides actionable recommendations based on feature importance analysis

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Model Performance** | PR AUC: 0.9655 |
| **Calibration** | Perfect (95.0% predicted vs 95.0% actual) |
| **Best Customer Retention** | 81.1% (vs 5% baseline) |
| **Prediction Range** | 18.9% - 96.3% drop-off |
| **Training Data** | 28,020 customers, 33 features |

---

## 🚀 Live Demo

**Try the interactive web app:** [Coming Soon - Streamlit Cloud]

**Features:**
- Real-time drop-off risk prediction
- Personalized retention recommendations
- ROI calculator for intervention decisions
- Risk level classification (Low/Medium/High/Critical)

---

## 📁 Project Structure
```
Olist-Marketplace-Integrity-Audit/
│
├── data/
│   ├── raw/                          # Original Olist datasets (8 tables)
│   └── processed/
│       └── feature_matrix.csv        # Engineered features (28,020 × 33)
│
├── notebooks/
│   ├── 01_eda_problem_framing.ipynb           # Data exploration & business problem
│   ├── 02_feature_engineering.ipynb           # Feature creation & validation
│   ├── 03_predictive_modeling.ipynb           # Model training & calibration
│   └── 04_streamlit_deployment.ipynb          # Web app deployment
│
├── outputs/
│   ├── figures/                      # Visualizations (PR curves, confusion matrices)
│   └── models/
│       ├── logistic_regression_calibrated.pkl  # Production model
│       └── feature_importance.csv              # Top features ranked
│
├── streamlit_app/
│   ├── app.py                        # Interactive web application
│   └── requirements.txt              # Dependencies
│
├── README.md                         # This file
└── requirements.txt                  # Project dependencies
```

---

## 🛠️ Technologies Used

**Data Processing:**
- Python 3.9+
- Pandas, NumPy
- Scikit-learn

**Machine Learning:**
- Logistic Regression (calibrated with Platt scaling)
- Random Forest & XGBoost (baseline comparison)
- Stratified K-Fold cross-validation
- SMOTE for class imbalance handling

**Deployment:**
- Streamlit (interactive web app)
- Pickle (model serialization)

**Visualization:**
- Matplotlib, Seaborn
- Plotly (interactive charts)

---

## 📊 Methodology

### 1. **Data Integration** (Notebook 1)
Merged 8 Olist datasets to create customer-centric view:
- Orders, products, payments, reviews, delivery logistics
- Removed data integrity issues (duplicates, canceled orders)
- Defined target: `dropped_off` (no second purchase within dataset timeframe)

### 2. **Feature Engineering** (Notebook 2)
Created 33 features across 6 categories:
- **Delivery Performance:** Delays, early delivery, total delivery time
- **Economic Factors:** Freight %, price per item, installment usage
- **Product Attributes:** Category repeatability, weight, review sentiment
- **Customer Profile:** Geographic location (Southeast vs other states)
- **Temporal Patterns:** Holiday season, weekend purchases, purchase month
- **Behavioral Clustering:** K-Means segmentation (2 clusters)

**Leakage Prevention:** Removed target-leaking features (e.g., category drop-off rates)

### 3. **Predictive Modeling** (Notebook 3)
Trained and compared three baseline models:

| Model | PR AUC | Minority Recall | Notes |
|-------|--------|-----------------|-------|
| **Logistic Regression** | **0.9654** | **62.6%** | ✅ Selected |
| Random Forest | 0.9500 | 0.0% | Collapsed to majority class |
| XGBoost | 0.9500 | 0.0% | Collapsed to majority class |

**Why Logistic Regression Won:**
- Only model that identified retained customers
- Interpretable coefficients for business recommendations
- Robust to extreme class imbalance (95/5 split)

**Calibration Fix:**
- Initial model miscalibrated (51% mean prediction vs 95% actual)
- Applied **Platt Scaling** using `CalibratedClassifierCV`
- Achieved perfect calibration (95.0% predicted vs 95.0% actual)

### 4. **Deployment** (Notebook 4)
Built interactive Streamlit web application with:
- Customer profile input form (20 features)
- Real-time drop-off risk prediction
- Personalized retention recommendations
- ROI calculator for intervention cost-effectiveness

---

## 🔑 Key Insights

### Top Retention Drivers (Reduce Drop-Off):
1. **Holiday Season Purchases** (-0.61) - 61% more likely to return
2. **Repeatable Categories** (-0.44) - Health/beauty, books, pet supplies
3. **Installment Payments** (-0.26) - 26% retention increase
4. **Early Delivery** (-0.18) - Positive delivery experience
5. **Southeast Location** (-0.12) - SP, RJ, MG, ES states

### Top Drop-Off Drivers (Increase Drop-Off):
1. **High Freight Cost** (+0.13) - Shipping >20% of order value
2. **Weekend Purchases** (+0.10) - Lower engagement
3. **Heavy Products** (+0.08) - Harder to repeat purchase
4. **Late Delivery** (+0.15) - Negative first experience
5. **Non-Southeast Customers** (+0.12) - Geographic barriers

---

## 💡 Business Recommendations

### 1. **Target High-Retention Segments**
- Focus acquisition on holiday season shoppers
- Promote repeatable product categories (health/beauty, books, pet supplies)
- Prioritize Southeast Brazil customers

### 2. **Reduce Friction for All Customers**
- Offer free/subsidized shipping (especially >20% of order value)
- Promote installment payment options at checkout
- Improve delivery reliability (reduce late deliveries)

### 3. **Implement Targeted Interventions**
- **Best-case customers** (81% retention): Premium onboarding, loyalty program
- **Average customers** (5% retention): Standard retention offers
- **Worst-case customers** (3% retention): Do NOT intervene (negative ROI)

### 4. **Strategic Insight**
The model excels at identifying **high-retention customers** rather than distinguishing among drop-off customers. This suggests:
- **Opportunity:** Invest in acquiring the RIGHT customers
- **Reality:** Drop-off is nearly universal (95%+) regardless of negative signals
- **Action:** Shift from "save everyone" to "acquire winners"

---

## 🚦 Getting Started

### Prerequisites
```bash
Python 3.9+
pip (package manager)
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ReynoldT92/Olist-Marketplace-Integrity-Audit.git
cd Olist-Marketplace-Integrity-Audit
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run Jupyter notebooks** (optional - explore analysis):
```bash
jupyter notebook
```

4. **Launch Streamlit app:**
```bash
cd streamlit_app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📈 Model Performance Details

### Confusion Matrix (Test Set)
```
                    Predicted
                 Retained  Dropped
Actual Retained     174      104      (62.6% recall)
       Dropped       54    5,272      (99.0% recall)
```

### Precision-Recall Trade-off
- **At 50% threshold:** 96.6% precision, 56.0% recall
- **At 30% threshold:** 85.2% precision, 75.8% recall
- **At 10% threshold:** 62.1% precision, 92.4% recall

### Calibration Performance
- **Before calibration:** 51.1% mean prediction (43.9pp error)
- **After Platt scaling:** 95.0% mean prediction (0.1pp error)
- **Calibration method:** Sigmoid (Platt scaling) with 5-fold CV

---

## 📚 Dataset

**Source:** [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)

**Description:** Real anonymized data from 100k orders (2016-2018) on Olist marketplace

**Tables Used:**
- `olist_orders_dataset.csv` (99,441 orders)
- `olist_order_items_dataset.csv` (112,650 items)
- `olist_order_payments_dataset.csv` (103,886 payments)
- `olist_order_reviews_dataset.csv` (99,224 reviews)
- `olist_customers_dataset.csv` (99,441 customers)
- `olist_products_dataset.csv` (32,951 products)
- `olist_sellers_dataset.csv` (3,095 sellers)
- `olist_geolocation_dataset.csv` (1,000,163 locations)

---

## 🎓 Skills Demonstrated

- **Data Engineering:** Multi-table joins, data integrity validation, feature engineering
- **Machine Learning:** Classification, calibration, hyperparameter tuning, cross-validation
- **Statistical Analysis:** Imbalanced datasets, precision-recall optimization, ROI modeling
- **Business Analytics:** Customer segmentation, retention economics, intervention prioritization
- **Deployment:** Streamlit app development, model serialization, production-ready code
- **Communication:** Technical documentation, visualization design, stakeholder presentation

---

## 📝 Future Improvements

- [ ] Deploy to Streamlit Community Cloud for public access
- [ ] Add time-series forecasting for seasonal patterns
- [ ] Implement A/B testing framework for interventions
- [ ] Explore deep learning models (LSTM for sequential behavior)
- [ ] Build automated retraining pipeline with MLflow
- [ ] Add customer lifetime value (CLV) prediction
- [ ] Create executive dashboard with real-time metrics

---

## 👤 Author

**Reynold Takura Choruma**

📧 Email: [chorumareynold@gmail.com]  
🔗 LinkedIn: [www.linkedin.com/in/reynold-choruma-3b6a27127]  
🐙 GitHub: [@ReynoldT92](https://github.com/ReynoldT92)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Olist** for providing the dataset
- **Ironhack Munich** Data Analytics Bootcamp
- **Instructor Sabina** for guidance and feedback

---

## 📊 Project Timeline

- **Week 1-2:** Data exploration & problem framing
- **Week 3:** Feature engineering & validation
- **Week 4:** Model training & optimization
- **Week 5:** Calibration & Streamlit deployment
- **Presentation:** April 11, 2026

---

**⭐ If you found this project helpful, please consider giving it a star!**

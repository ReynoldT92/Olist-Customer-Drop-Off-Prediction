
import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Olist Retention Predictor",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stButton>button:hover {background-color: #27ae60;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained Logistic Regression model"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'outputs', 'models', 'logistic_regression_calibrated.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.title("🛒 Olist Customer Retention Predictor")
st.markdown("### Predict first-time customer drop-off risk")

st.markdown("""
**Problem:** 95% of Olist first-time customers never make a second purchase.

**Solution:** This tool predicts drop-off risk based on first order characteristics.

**How it works:**
1. Enter customer's first order details below
2. Click "Predict Drop-off Risk"
3. Get instant risk assessment and recommendations
""")

st.divider()

# ============================================================================
# INPUT FORM
# ============================================================================

st.header("📝 Customer First Order Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚚 Delivery")
    delivery_delay = st.number_input(
        "Delivery Delay (days)",
        min_value=-30,
        max_value=100,
        value=0,
        help="Negative = early, Positive = late"
    )
    
    days_to_delivery = st.number_input(
        "Total Days to Delivery",
        min_value=0,
        max_value=100,
        value=10
    )

with col2:
    st.subheader("💰 Economics")
    freight_pct = st.slider(
        "Freight % of Order Value",
        min_value=0.0,
        max_value=80.0,
        value=15.0,
        help="Shipping cost as % of order value"
    )
    
    num_items = st.number_input(
        "Number of Items",
        min_value=1,
        max_value=20,
        value=1
    )
    
    price_per_item = st.number_input(
        "Price per Item (R$)",
        min_value=1.0,
        max_value=10000.0,
        value=100.0
    )
    
    uses_installments = st.checkbox("Uses Installment Payment", value=False)

with col3:
    st.subheader("📍 Customer & Product")
    is_southeast = st.checkbox(
        "Southeast Brazil Customer",
        value=True,
        help="SP, RJ, MG, ES states"
    )
    
    is_repeatable_category = st.checkbox(
        "Repeatable Category",
        value=False,
        help="Health/beauty, books, pet supplies"
    )
    
    is_heavy_product = st.checkbox("Heavy Product (>5kg)", value=False)
    
    has_comment = st.checkbox("Left Review Comment", value=False)
    
    is_holiday_season = st.checkbox(
        "Holiday Season Purchase",
        value=False,
        help="November or December"
    )
    
    is_weekend = st.checkbox("Weekend Purchase", value=False)

# Advanced options
with st.expander("⚙️ Advanced Options"):
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        purchase_month = st.selectbox(
            "Purchase Month",
            options=list(range(1, 13)),
            index=10  # November
        )
        
        purchase_day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x],
            index=0
        )
    
    with adv_col2:
        cluster = st.selectbox(
            "Customer Segment",
            options=["Unknown", "Budget Shoppers (0)", "High Risk (1)"],
            index=0
        )

st.divider()

# ============================================================================
# PREDICTION
# ============================================================================

if st.button("Predict Drop-off Risk", type="primary"):
    
    # Calculate derived features
    is_late_delivery = int(delivery_delay > 0)
    is_very_late = int(delivery_delay > 10)
    is_early_delivery = int(delivery_delay < 0)
    is_high_freight = int(freight_pct > 20)
    
    # Cluster encoding
    cluster_0 = int(cluster == "Budget Shoppers (0)")
    cluster_1 = int(cluster == "High Risk (1)")
    
    # Create feature vector - ONLY THE 20 FEATURES THE MODEL USES!
    features = pd.DataFrame({
        'delivery_delay': [float(delivery_delay)],
        'is_late_delivery': [int(is_late_delivery)],
        'is_very_late': [int(is_very_late)],
        'is_early_delivery': [int(is_early_delivery)],
        'freight_pct': [float(freight_pct)],
        'is_high_freight': [int(is_high_freight)],
        'num_items': [int(num_items)],
        'price_per_item': [float(price_per_item)],
        'uses_installments': [int(uses_installments)],
        'is_southeast': [int(is_southeast)],
        'is_repeatable_category': [int(is_repeatable_category)],
        'is_heavy_product': [int(is_heavy_product)],
        'has_comment': [int(has_comment)],
        'purchase_month': [int(purchase_month)],
        'purchase_day_of_week': [int(purchase_day_of_week)],
        'is_weekend': [int(is_weekend)],
        'is_holiday_season': [int(is_holiday_season)],
        'days_to_delivery': [int(days_to_delivery)],
        'cluster_0': [int(cluster_0)],
        'cluster_1': [int(cluster_1)]
    })
    
    try:
        # Make prediction
        prediction_proba = model.predict_proba(features)[0]
        drop_off_prob = prediction_proba[1] * 100
        retention_prob = prediction_proba[0] * 100
        
        # Determine risk level
        if drop_off_prob >= 98:
            risk_level = "🔴 CRITICAL RISK"
            risk_color = "red"
        elif drop_off_prob >= 95:
            risk_level = "🟠 HIGH RISK"
            risk_color = "orange"
        elif drop_off_prob >= 90:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "blue"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
        
        # Display results
        st.divider()
        st.header("📊 Prediction Results")
        
        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Drop-off Probability",
                f"{drop_off_prob:.1f}%",
                delta=f"{drop_off_prob - 95:.1f}% vs baseline",
                delta_color="inverse"
            )
        
        with metric_col2:
            st.metric(
                "Retention Probability",
                f"{retention_prob:.1f}%",
                delta=f"{retention_prob - 5:.1f}% vs baseline"
            )
        
        with metric_col3:
            if risk_color == "red":
                st.error(risk_level)
            elif risk_color == "orange":
                st.warning(risk_level)
            elif risk_color == "blue":
                st.info(risk_level)
            else:
                st.success(risk_level)
        
        # Recommendations
        st.subheader("💡 Personalized Recommendations")
        
        recommendations = []
        
        if drop_off_prob >= 95:
            recommendations.append("🚨 **URGENT:** High-risk customer - activate retention protocol immediately")
        
        if is_high_freight:
            recommendations.append("📦 **High shipping cost** - Consider free shipping offer")
        
        if not is_repeatable_category:
            recommendations.append("🔄 **Non-repeatable product** - Cross-sell to recurring categories")
        
        if not uses_installments:
            recommendations.append("💳 **Promote installment payments** - 26% retention increase")
        
        if is_holiday_season:
            recommendations.append("🎄 **Holiday purchase advantage** - 61% more likely to return!")
        else:
            recommendations.append("📅 **Consider seasonal promotion** to re-engage")
        
        if is_late_delivery:
            recommendations.append("⏰ **Late delivery** - Issue apology credit or compensation")
        
        if not is_southeast:
            recommendations.append("🗺️ **Non-Southeast customer** - Extra attention needed")
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # ROI Calculator
        st.subheader("💰 Retention ROI Estimate")
        
        roi_col1, roi_col2 = st.columns(2)
        
        with roi_col1:
            st.markdown("**Intervention Cost:** R$ 15")
            st.markdown("**Success Rate:** 30%")
            st.markdown("**Customer LTV:** R$ 160")
        
        with roi_col2:
            expected_value = (retention_prob / 100) * 160 - 15
            roi = ((expected_value + 15) / 15 - 1) * 100 if expected_value > -15 else -100
            
            st.metric("Expected Value", f"R$ {expected_value:.2f}")
            st.metric("ROI", f"{roi:.1f}%")
            
            if expected_value > 0:
                st.success("Intervention recommended")
            else:
                st.warning("Intervention not cost-effective")
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.error("Please check inputs and try again")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Olist Marketplace Integrity Audit</strong> | Reynold Choruma | March 2026</p>
    <p>Model: Calibrated Logistic Regression | PR AUC: 0.9655 | Perfect Calibration</p>
</div>
""", unsafe_allow_html=True)

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Page Configuration
BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Customer Churn Intelligence Suite",
    page_icon="🛡️",
    layout="wide"
)

# Modern Custom CSS Styling & Component Enhancements
st.markdown(
    """
    <style>
    /* Main Background & Font */
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    /* Card Containers */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-card h3 {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card h2 {
        color: #0f172a;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }

    /* Headers */
    .main-header {
        font-size: 32px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #475569;
        font-size: 16px;
        margin-top: 5px;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Cached Data Loading & Preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df

df = load_data()

# Model Training Pipeline Cached
@st.cache_resource
def get_trained_model(data):
    df_ml = data.copy().drop(["customerID"], axis=1)
    label_encoders = {}
    for col in df_ml.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        label_encoders[col] = le

    X = df_ml.drop("Churn", axis=1)
    y = df_ml["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    return gb, X, X_test, y_test, label_encoders

model, X_features, X_test, y_test, encoders = get_trained_model(df)
y_pred = model.predict(X_test)

# Sidebar Navigation
st.sidebar.markdown("### 🛡️ Churn Intelligence")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Dashboard Overview",
        "Live Churn Predictor",
        "Dataset Explorer",
        "Exploratory Analytics",
        "Model Performance"
    ]
)

# --- 1. DASHBOARD OVERVIEW ---
if page == "Dashboard Overview":
    st.markdown("<div class='main-header'>Customer Churn Executive Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>High-level metrics tracking customer retention and risk indicators.</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    total_customers = len(df)
    churn_customers = len(df[df["Churn"] == "Yes"])
    active_customers = total_customers - churn_customers
    churn_rate = (churn_customers / total_customers) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>Total Customers</h3><h2>{total_customers:,}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3>Active Users</h3><h2>{active_customers:,}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3>Churned Users</h3><h2>{churn_customers:,}</h2></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h3>Churn Rate</h3><h2>{churn_rate:.2f}%</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Tenure vs Churn Tendency")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", palette="crest", ax=ax)
        sns.despine()
        st.pyplot(fig)
        
    with col_b:
        st.subheader("Monthly Charges Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=df, x="MonthlyCharges", hue="Churn", multiple="stack", palette="flare", ax=ax)
        sns.despine()
        st.pyplot(fig)

# --- 2. LIVE CHURN PREDICTOR ---
elif page == "Live Churn Predictor":
    st.markdown("<div class='main-header'>🔮 Interactive Churn Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Configure customer attributes below to evaluate their risk level in real-time.</div>", unsafe_allow_html=True)
    st.markdown("---")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 18.0, 9000.0, 800.0)
            contract = st.selectbox("Contract Type", df["Contract"].unique())
        with c2:
            internet_service = st.selectbox("Internet Service", df["InternetService"].unique())
            online_security = st.selectbox("Online Security", df["OnlineSecurity"].unique())
            tech_support = st.selectbox("Tech Support", df["TechSupport"].unique())
            payment_method = st.selectbox("Payment Method", df["PaymentMethod"].unique())
        with c3:
            paperless = st.selectbox("Paperless Billing", df["PaperlessBilling"].unique())
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", df["Partner"].unique())
            dependents = st.selectbox("Dependents", df["Dependents"].unique())

        submitted = st.form_submit_button("Run Prediction Analysis", use_container_width=True)

    if submitted:
        # Construct input dataframe matching training structure
        input_data = df.iloc[0:1].drop(["customerID", "Churn"], axis=1).copy()
        input_data["tenure"] = tenure
        input_data["MonthlyCharges"] = monthly_charges
        input_data["TotalCharges"] = total_charges
        input_data["Contract"] = contract
        input_data["InternetService"] = internet_service
        input_data["OnlineSecurity"] = online_security
        input_data["TechSupport"] = tech_support
        input_data["PaymentMethod"] = payment_method
        input_data["PaperlessBilling"] = paperless
        input_data["SeniorCitizen"] = senior
        input_data["Partner"] = partner
        input_data["Dependents"] = dependents

        # Encode input variables using trained encoders
        for col in input_data.select_dtypes(include="object").columns:
            if col in encoders:
                le = encoders[col]
                # handle unseen labels safely
                val = input_data[col].values[0]
                if val not in le.classes_:
                    val = le.classes_[0]
                input_data[col] = le.transform([val])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        st.markdown("---")
        if prediction == 1: # Churn encoded position
            st.error(f"⚠️ **High Churn Risk Detected!** (Probability: {prob[1]*100:.1f}%). Recommended action: Offer retention discounts or long-term contract incentives.")
        else:
            st.success(f"✅ **Low Churn Risk / Loyal Customer** (Retention Probability: {prob[0]*100:.1f}%).")

# --- 3. DATASET EXPLORER ---
elif page == "Dataset Explorer":
    st.markdown("<div class='main-header'>📁 Raw Dataset Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Inspect raw records straight from the telecom repository.</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    row_limit = st.slider("Rows to display", min_value=5, max_value=len(df), value=10)
    st.dataframe(df.head(row_limit), use_container_width=True)

# --- 4. EXPLORATORY ANALYTICS ---
elif page == "Visualizations":
    st.markdown("<div class='main-header'>📈 Exploratory Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Deep dive into demographic and account feature distributions.</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    viz_tab1, viz_tab2 = st.tabs(["Categorical Breakdowns", "Correlation Matrix"])
    
    with viz_tab1:
        cat_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "InternetService", "Contract", "PaymentMethod"]
        selected_cat = st.selectbox("Select Feature to Visualize", cat_cols)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x=selected_cat, hue="Churn", palette="Set2", ax=ax)
        for container in ax.containers:
            ax.bar_label(container)
        sns.despine()
        st.pyplot(fig)
        
    with viz_tab2:
        st.subheader("Feature Correlation Heatmap")
        temp = df.copy()
        for col in temp.select_dtypes(include="object"):
            le = LabelEncoder()
            temp[col] = le.fit_transform(temp[col])

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(temp.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# --- 5. MODEL PERFORMANCE ---
elif page == "Model Performance":
    st.markdown("<div class='main-header'>⚙️ Machine Learning Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Evaluation metrics for the Gradient Boosting Classification backend.</div>", unsafe_allow_html=True)
    st.markdown("---")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Precision", f"{precision*100:.2f}%")
    col3.metric("Recall", f"{recall*100:.2f}%")
    col4.metric("F1 Score", f"{f1*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
    with c_right:
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            "Feature": X_features.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(8)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=importance, x="Importance", y="Feature", palette="viridis", ax=ax)
        sns.despine()
        st.pyplot(fig)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #94a3b8; font-size: 13px;'>
    Customer Churn Intelligence Suite &bull; Built with Streamlit & Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)
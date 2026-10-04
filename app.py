from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #17324d;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

df = pd.read_csv(BASE_DIR / "Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset",
        "Visualizations",
        "Model Performance"
    ]
)

if page == "Dashboard":
    st.markdown("<h1 style='text-align: center;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("------")
    total_customers = len(df)
    churn_customers = len(df[df["Churn"] == "Yes"])
    active_customers = total_customers - churn_customers
    churn_rate = (churn_customers / total_customers) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", total_customers)
    col2.metric("Active Customers", active_customers)
    col3.metric("Churn Customers", churn_customers)
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")

if page == "Dataset":
    st.subheader("Dataset Preview")
    x=st.slider("Data",min_value=1,max_value=len(df),value=5)
    st.dataframe(df.head(x))

if page == "Visualizations":
    st.markdown("<h1 style='text-align: center;'>📈 Customer Churn Visualizations</h1>", unsafe_allow_html=True)
    st.markdown("------")
    st.subheader("Customer Churn Distribution")
    st.markdown("----")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        st.subheader("Churn Distribution")
        sns.countplot(data=df,x="Churn",ax=ax,palette="Set2")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    with col2:
        st.subheader("Gender vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(data=df,x="gender",hue="Churn",ax=ax,palette="Set3")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Senior Citizen vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(data=df,x="SeniorCitizen",hue="Churn",ax=ax,palette="Set1")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    with col2:
        st.subheader("Partner vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(data=df,x="Partner",hue="Churn",ax=ax,palette="Set2")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Dependents vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(data=df,x="Dependents",hue="Churn",ax=ax,palette="Set3")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    with col2: 
        st.subheader("Internet Service vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(data=df,x="InternetService",hue="Churn",ax=ax,palette="Set1")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:   
        st.subheader("Online Security vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(data=df,x="OnlineSecurity",hue="Churn",ax=ax,palette="Set2")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    with col2:
        st.subheader("Online Backup vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(data=df,x="OnlineBackup",hue="Churn",ax=ax,palette="Set3")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Device Protection vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(data=df,x="DeviceProtection",hue="Churn",ax=ax,palette="Set1")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    with col2:
        st.subheader("Tech Support vs Churn")
        fig, ax = plt.subplots()
        sns.countplot(data=df,x="TechSupport",hue="Churn",ax=ax,palette="Set2")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

        

    st.subheader("Tenure Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["tenure"],bins=20,kde=True,ax=ax,color="purple")
    st.pyplot(fig)

    st.subheader("Total Charges Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["TotalCharges"],bins=20,kde=True,ax=ax,color="orange")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    temp = df.copy()
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    for col in temp.select_dtypes(include="object"):
        temp[col] = encoder.fit_transform(temp[col])

    fig, ax = plt.subplots(figsize=(15,8))
    sns.heatmap(
            temp.corr(),
            annot=True,
            cmap="Blues",
            ax=ax)
    st.pyplot(fig)

if page == "Model Performance":
    st.markdown("<h1 style='text-align: center;'>Performance Analysis</h1>", unsafe_allow_html=True)
    st.markdown("------")

    df_ml = df.copy()
    df_ml = df_ml.drop(["customerID"], axis=1)

    for col in df_ml.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        df_ml[col] = encoder.fit_transform(df_ml[col])

    X = df_ml.drop("Churn", axis=1)
    y = df_ml["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train,y_train)

    y_pred = gb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Model Accuracy")
    st.metric("Accuracy", f"{accuracy*100:.2f}%")
    st.markdown("---")

    left,space,right = st.columns([2, 0.4, 2])
    with left:
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn","Churn"],
            yticklabels=["No Churn","Churn"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("📈 Model Performance")
        st.metric("Accuracy", f"{accuracy_score(y_test,y_pred)*100:.2f}%")
        st.metric("Precision", f"{precision_score(y_test,y_pred)*100:.2f}%")
        st.metric("Recall", f"{recall_score(y_test,y_pred)*100:.2f}%")
        st.metric("F1 Score", f"{f1_score(y_test,y_pred)*100:.2f}%")

    st.subheader("📄 Classification Report")
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.subheader("⭐ Feature Importance")

    importance = pd.DataFrame({
        "Feature":X.columns,
        "Importance":gb.feature_importances_
    })
    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=importance,x="Importance",y="Feature",ax=ax)
    st.pyplot(fig)
    importance.index = range(1,len(importance)+1)
    st.dataframe(importance)

st.markdown("-----")
st.markdown(
    """
    <center>

    Developed for Analysis and Prediction of Customer Churn in Telecom Industry.    

    </center>
    """,
    unsafe_allow_html=True
)

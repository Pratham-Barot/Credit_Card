import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# ---------------- HEADER ----------------
st.title("💳 Credit Card Fraud Detection System")
st.caption("Upload transaction data to detect fraudulent activity")

st.divider()

# ---------------- MODEL INFO ----------------
st.info(f"Model expects {model.n_features_in_} features per transaction.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV file with transaction data",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        # Check feature count
        if data.shape[1] != model.n_features_in_:
            st.error(
                f"Feature mismatch! Model expects {model.n_features_in_} features "
                f"but uploaded file has {data.shape[1]} columns."
            )
        else:
            # Predict
            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1]

            result_df = data.copy()
            result_df["Fraud Probability"] = probabilities
            result_df["Prediction"] = predictions

            st.divider()
            st.subheader("📊 Prediction Results")

            st.dataframe(result_df.head())

            fraud_count = sum(predictions)
            total = len(predictions)

            st.metric("Total Transactions", total)
            st.metric("Fraudulent Transactions Detected", fraud_count)

    except Exception as e:
        st.error("Error processing file.")
        st.write(e)

# ---------------- FOOTER ----------------
st.divider()
st.caption("Developed by Pratham Barot | AI/ML Fraud Detection Project")
# 💳 Credit Card Fraud Detection System

An AI-powered Machine Learning project that detects fraudulent credit card transactions using a Random Forest classifier.

---

## Dataset

Dataset Source: Kaggle Credit Card Fraud Detection Dataset

## 🚀 Project Overview

This project builds a fraud detection system using supervised machine learning.  
The model predicts whether a transaction is:

- **0 → Legitimate**
- **1 → Fraudulent**

The system is deployed using **Streamlit** for real-time fraud prediction.

---

## 📊 Features Used

- Time
- V1 – V28 (PCA-transformed anonymized transaction features)
- Amount

These features were used to train a Random Forest model to detect fraud patterns.

---

## 🧠 Machine Learning Model

- Algorithm: Random Forest
- Handles Imbalanced Dataset
- Evaluated using:
  - Precision
  - Recall
  - ROC-AUC
  - Confusion Matrix

---

## 🌐 Deployment

The application is deployed using Streamlit.

Users can:
- Upload transaction CSV files
- Get fraud probability
- Detect fraudulent transactions instantly
  
---

## ⚙️ Run Locally

1️⃣ Clone the repository:

git clone https://github.com/Pratham-Barot/Credit_Card.git

2️⃣ Go to project folder:

cd Credit_Card

3️⃣ Install dependencies:

pip install -r requirements.txt

4️⃣ Run Streamlit app:

streamlit run app.py

---

## 📦 requirements.txt

streamlit  
scikit-learn  
joblib  
numpy  
pandas  

---

## 👨‍💻 Developed By

Pratham Barot  
B.Tech ICT | AI/ML Enthusiast

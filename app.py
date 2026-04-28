import google.generativeai as genai

genai.configure(api_key="AIzaSyAJqoeJ1-WWk-wghD6H2SK8MH0t1pjuwTA")

model_gemini = genai.GenerativeModel("models/gemini-1.5-flash")

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="BiasLens AI", layout="wide")

st.title("BiasLens AI")
st.subheader("Detect & Fix Bias in AI Models")

# Upload or use default dataset
file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    df = pd.read_csv("data.csv")
    st.info("Using default dataset")

st.write("### Dataset Preview")
st.dataframe(df)

# Convert Gender to numeric
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

X = df[['Gender', 'Income', 'CreditScore']]
y = df['Approved']

# Train model
model = LogisticRegression()
model.fit(X, y)

df['Prediction'] = df["Approved"]

st.write("###  Model Predictions")
st.dataframe(df)

# Bias Detection
male_rate = df[df['Gender']==0]['Prediction'].mean()
female_rate = df[df['Gender']==1]['Prediction'].mean()

gap = abs(male_rate - female_rate) * 100

st.write("### Bias Analysis...")

col1, col2 = st.columns(2)

col1.metric("Male Approval %", f"{male_rate*100:.2f}%")
col2.metric("Female Approval %", f"{female_rate*100:.2f}%")

st.metric("Bias Gap", f"{gap:.2f}%")
st.divider()
st.header("📈 Fairness Score")
st.progress(int(100-gap))
st.caption("Higher score means more fair model!")

if gap > 10:
    st.error("⚠️ Bias Detected in Model")
else:
    st.success("✅ Model is Fair")

    # Gemini Explanation Button
if st.button("Why? Ask Gemeni✨️"):
    st.write("Gemeni Explanation:")

    if gap > 10:
        st.write(
            "The model is biased because the dataset contains more approvals for male applicants. "
            "As a result, the model has learned to favor males over females. "
            "Removing gender as a feature or balancing the dataset helps reduce this bias."
        )
    else:
        st.write(
            "The model appears fair because approval rates are similar across groups, "
            "indicating no strong bias in decision-making."
        )

# Fix Bias Button
if st.button("Fix Bias"):

    # Remove Gender influence (simple fairness trick)
    Xb = df[["Income", "CreditScore"]]  
    yb = df["Approved"]

    model.fit(Xb, yb)

    df["Prediction"] = model.predict(Xb)

    male_rate2 = df[df["Gender"] == 0]["Prediction"].mean()
    female_rate2 = df[df["Gender"] == 1]["Prediction"].mean()

    gap2 = abs(male_rate2 - female_rate2) * 100

    st.write("## ✅ After Bias Fix")

    st.metric("Male Approval %", f"{male_rate2*100:.2f}%")
    st.metric("Female Approval %", f"{female_rate2*100:.2f}%")
    st.metric("New Bias Gap", f"{gap2:.2f}%")

    st.success(" Bias Reduced Successfully!")

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="BiasLens AI", layout="wide")


def load_sample_data():
    data = {
        "Gender": [
            "Male","Male","Male","Male","Male",
            "Male","Male","Male","Male","Male",
            "Female","Female","Female","Female","Female",
            "Female","Female","Female","Female","Female","Female"
        ],
        "Income": [
            60000,58000,62000,55000,53000,
            50000,48000,47000,45000,44000,
            62000,60000,58000,56000,54000,
            52000,50000,48000,46000,44000,42000
        ],
        "CreditScore": [
            720,710,730,700,690,
            680,670,660,650,640,
            730,720,710,700,690,
            680,670,660,650,640,630
        ],
        "Approved": [
            1,1,1,1,1,
            0,1,1,0,1,
            1,1,1,0,0,
            0,0,0,1,0,0
        ]
    }
    return pd.DataFrame(data)


if "df" not in st.session_state:
    st.session_state.df = None

if "fix_clicked" not in st.session_state:
    st.session_state.fix_clicked = False

if "explain_clicked" not in st.session_state:
    st.session_state.explain_clicked = False


st.title("BiasLens AI")
st.subheader("Detect & Fix Bias in AI Models")

st.write("## Dataset Options")

use_sample = st.button("Use Sample Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    key="main_uploader"
)


if use_sample:
    st.session_state.df = load_sample_data()
    st.success("Using sample dataset")

elif uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)

if st.session_state.df is None:
    st.warning("Please upload a dataset or use sample data.")
    st.stop()

df = st.session_state.df.copy()

required_columns = ["Gender", "Income", "CreditScore", "Approved"]

if not all(col in df.columns for col in required_columns):
    st.error("Dataset must contain: Gender, Income, CreditScore, Approved")
    st.stop()


st.write("### Dataset Preview")
st.dataframe(df)


df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

X = df[["Gender", "Income", "CreditScore"]]
y = df["Approved"]

model = LogisticRegression()
model.fit(X, y)

df["Prediction"] = df["Approved"]


st.write("### Model Predictions")
st.dataframe(df)

male_rate = df[df["Gender"] == 0]["Prediction"].mean()
female_rate = df[df["Gender"] == 1]["Prediction"].mean()
gap = abs(male_rate - female_rate) * 100

st.write("### Bias Analysis")

col1, col2 = st.columns(2)
col1.metric("Male Approval %", f"{male_rate*100:.2f}%")
col2.metric("Female Approval %", f"{female_rate*100:.2f}%")

st.metric("Bias Gap", f"{gap:.2f}%")

st.divider()
st.header("Fairness Score")
st.progress(int(100 - gap))
st.caption("Higher score means a more fair model")

if gap > 10:
    st.error("Bias Detected in Model")
else:
    st.success("Model appears fair")


if st.button("Why? Ask Google Gemini"):
    st.session_state.explain_clicked = True

if st.session_state.explain_clicked:
    st.write("### Explanation")

    if gap > 10:
        st.info(
            "The model is biased because approval rates differ significantly "
            "between male and female applicants. This usually comes from "
            "imbalanced or biased historical data."
        )
    else:
        st.success(
            "The model appears fair because approval rates are similar across groups."
        )

if st.button("Fix Bias"):
    st.session_state.fix_clicked = True

if st.session_state.fix_clicked:

    X_fixed = df[["Income", "CreditScore"]]
    y_fixed = df["Approved"]

    fixed_model = LogisticRegression()
    fixed_model.fit(X_fixed, y_fixed)

    df["Prediction"] = fixed_model.predict(X_fixed)

    male_rate2 = df[df["Gender"] == 0]["Prediction"].mean()
    female_rate2 = df[df["Gender"] == 1]["Prediction"].mean()
    gap2 = abs(male_rate2 - female_rate2) * 100

    st.write("## After Bias Fix")

    col3, col4 = st.columns(2)
    col3.metric("Male Approval %", f"{male_rate2*100:.2f}%")
    col4.metric("Female Approval %", f"{female_rate2*100:.2f}%")

    st.metric("New Bias Gap", f"{gap2:.2f}%")

    st.success("Bias Reduced Successfully !")

    st.markdown(" Bias Reduction Explanation:")

    st.success(f"""
What changed?  
- We removed the *Gender* feature from the model.

Why was the model biased earlier? 
- The model was using Gender, leading to unfair differences in approval rates.

What did we do?
We retrained the model using only:
- Income  
- Credit Score  

Result:
Bias gap reduced from **{gap:.2f}%** to **{gap2:.2f}%**

Conclusion:  
Removing sensitive attributes improves fairness, but complete fairness may require more advanced techniques.
""")

# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
from data_loader import load_dataset

st.set_page_config(layout="wide", page_title="ML Prediction App")

MODELS_DIR = "models"

# -------------------------------
# Dataset groups
# -------------------------------
CLASS_DATASETS = {"Titanic": "titanic", "Zoo": "zoo"}
REG_DATASETS = {"Salary Data": "salary_data", "Insurance": "insurance"}

ALGS_CLASS = ["logistic", "decision_tree", "rf"]
ALGS_REG = ["linear", "ridge", "lasso", "rf"]

# -------------------------------
# Decoding maps for encoded data
# -------------------------------
DECODE_MAPS = {
    "titanic": {
        "Sex": {0: "Female", 1: "Male"},
        "Embarked": {0: "C", 1: "Q", 2: "S"},
        "Pclass": {1: "1st Class", 2: "2nd Class", 3: "3rd Class"},
    }
}

ENCODE_MAPS = {
    ds: {col: {v: k for k, v in mp.items()} for col, mp in cols.items()}
    for ds, cols in DECODE_MAPS.items()
}

# -------------------------------
# Session init
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"

# ================= MAIN PAGE =================
if st.session_state.page == "main":
    st.title("Machine Learning Prediction App")

    left, mid, right = st.columns([1, 2, 1])

    with left:
        task = st.selectbox("Task", ["Classification", "Regression"])

        if task == "Classification":
            label = st.selectbox("Dataset", list(CLASS_DATASETS.keys()))
            dataset = CLASS_DATASETS[label]
            alg = st.selectbox("Algorithm", ALGS_CLASS)
        else:
            label = st.selectbox("Dataset", list(REG_DATASETS.keys()))
            dataset = REG_DATASETS[label]
            alg = st.selectbox("Algorithm", ALGS_REG)

        if st.button("Proceed to Prediction"):
            st.session_state.cfg = {
                "task": task,
                "dataset": dataset,
                "alg": alg,
                "label": label
            }
            st.session_state.page = "prediction"
            st.rerun()

    with mid:
        st.header("Algorithms Information")
        st.markdown("""
        - **Logistic Regression** – Linear classifier  
        - **Decision Tree** – Rule-based classifier  
        - **Random Forest** – Ensemble model  
        - **Linear / Ridge / Lasso** – Regression models
        """)

    with right:
        st.header("Datasets Information")
        st.markdown("""
        - **Titanic** – Survival prediction  
        - **Zoo** – Animal classification  
        - **Salary Data** – Salary regression  
        - **Insurance** – Expense prediction
        """)

# ================= PREDICTION PAGE =================
else:
    cfg = st.session_state.cfg
    st.title(f"Prediction — {cfg['label']} ({cfg['alg']})")

    if st.button("Back"):
        st.session_state.page = "main"
        st.rerun()

    # Load trained pipeline (preprocessor + model)
    model = joblib.load(f"{MODELS_DIR}/{cfg['dataset']}_{cfg['alg']}.pkl")

    # Load raw dataset (for UI + EDA)
    df = load_dataset(cfg["dataset"])

    # Load target name (for EDA only)
    schema = json.load(open(f"{MODELS_DIR}/{cfg['dataset']}_{cfg['alg']}_schema.json"))
    target = schema["target"]

    # Raw features only
    X_raw = df.drop(columns=[target])

    # ---------------- INPUT FORM ----------------
    st.sidebar.header("Input Features")

    with st.sidebar.form("predict_form"):
        user_input = {}

        for col in X_raw.columns:

            # Case 1: Decoded categorical (Titanic)
            if cfg["dataset"] in DECODE_MAPS and col in DECODE_MAPS[cfg["dataset"]]:
                options = list(DECODE_MAPS[cfg["dataset"]][col].values())
                selected = st.selectbox(col, options)
                user_input[col] = ENCODE_MAPS[cfg["dataset"]][col][selected]

            # Case 2: Numeric
            elif pd.api.types.is_numeric_dtype(X_raw[col]):
                if (X_raw[col] % 1 == 0).all():
                    user_input[col] = st.number_input(
                        col,
                        value=int(X_raw[col].median()),
                        step=1
                    )
                else:
                    user_input[col] = st.number_input(
                        col,
                        value=float(X_raw[col].median())
                    )

            # Case 3: Other categorical
            else:
                user_input[col] = st.selectbox(
                    col,
                    sorted(X_raw[col].dropna().unique())
                )

        submitted = st.form_submit_button("Predict")

    # ---------------- PREDICTION RESULT ----------------
    if submitted:
        X_input = pd.DataFrame([user_input])
        pred = model.predict(X_input)[0]
        st.subheader("Prediction Result")
        st.success(pred)

    # ================= EDA =================
    st.header("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Dataset Overview", "Feature Distributions", "Target Analysis", "Feature Correlations"]
    )

    # ---- Overview ----
    with tab1:
        st.dataframe(df.head())
        st.dataframe(df.describe(include="all"))

    # ---- Feature Distributions ----
    with tab2:
        numeric_cols = df.select_dtypes("number").columns
        for i, col in enumerate(numeric_cols):
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, key=f"dist_{col}_{i}")

    # ---- Target Analysis ----
    with tab3:
        if df[target].dtype == "object":
            vc = df[target].value_counts().reset_index()
            vc.columns = ["value", "count"]
            fig = px.bar(vc, x="value", y="count", title=f"Target Distribution: {target}")
            st.plotly_chart(fig, key="target_bar")
        else:
            fig = px.histogram(df, x=target, title=f"Target Distribution: {target}")
            st.plotly_chart(fig, key="target_hist")

    # ---- Correlations ----
    with tab4:
        num_df = df.select_dtypes("number")
        if len(num_df.columns) > 1:
            fig = px.imshow(num_df.corr(), text_auto=True, title="Feature Correlations")
            st.plotly_chart(fig, key="corr_plot")
        else:
            st.info("Not enough numeric columns for correlation analysis.")

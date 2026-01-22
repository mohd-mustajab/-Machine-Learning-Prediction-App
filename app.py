# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
from data_loader import load_dataset

st.set_page_config(layout="wide", page_title="ML Prediction App")

MODELS_DIR = "models"

CLASS_DATASETS = {"Titanic": "titanic", "Zoo": "zoo"}
REG_DATASETS = {"Salary Data": "salary_data", "Insurance": "insurance"}

ALGS_CLASS = ["logistic", "decision_tree", "rf"]
ALGS_REG = ["linear", "ridge", "lasso", "rf"]

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

if "page" not in st.session_state:
    st.session_state.page = "main"

# ================= MAIN PAGE =================
if st.session_state.page == "main":
    st.title("Machine Learning Prediction App")

    left, mid, right = st.columns([1, 2, 1])

    with left:
        task = st.selectbox("Task", ["Classification", "Regression"])

        if task == "Classification":
            label = st.selectbox("Dataset", CLASS_DATASETS.keys())
            dataset = CLASS_DATASETS[label]
            alg = st.selectbox("Algorithm", ALGS_CLASS)
        else:
            label = st.selectbox("Dataset", REG_DATASETS.keys())
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
        st.header("Algorithms Info")
        st.write("Logistic, Decision Tree, Random Forest, Linear, Ridge, Lasso")

    with right:
        st.header("Datasets Info")
        st.write("Titanic, Zoo, Salary Data, Insurance")

# ================= PREDICTION PAGE =================
else:
    cfg = st.session_state.cfg
    st.title(f"Prediction — {cfg['label']} ({cfg['alg']})")

    if st.button("Back"):
        st.session_state.page = "main"
        st.rerun()

    model = joblib.load(f"{MODELS_DIR}/{cfg['dataset']}_{cfg['alg']}.pkl")
    df = load_dataset(cfg["dataset"])

    schema = json.load(open(f"{MODELS_DIR}/{cfg['dataset']}_{cfg['alg']}_schema.json"))
    target = schema["target"]

    X_raw = df.drop(columns=[target])

    st.sidebar.header("Input Features")

    with st.sidebar.form("predict_form"):
        user_input = {}

        for col in X_raw.columns:
            if cfg["dataset"] in DECODE_MAPS and col in DECODE_MAPS[cfg["dataset"]]:
                options = list(DECODE_MAPS[cfg["dataset"]][col].values())
                selected = st.selectbox(col, options)
                user_input[col] = ENCODE_MAPS[cfg["dataset"]][col][selected]

            elif pd.api.types.is_numeric_dtype(X_raw[col]):
                if (X_raw[col] % 1 == 0).all():
                    user_input[col] = st.number_input(col, int(X_raw[col].median()), step=1)
                else:
                    user_input[col] = st.number_input(col, float(X_raw[col].median()))
            else:
                user_input[col] = st.selectbox(col, sorted(X_raw[col].dropna().unique()))

        submitted = st.form_submit_button("Predict")

    if submitted:
        X_input = pd.DataFrame([user_input])

        # ALIGN COLUMNS
        expected_cols = model.named_steps["preprocessor"].feature_names_in_
        for c in expected_cols:
            if c not in X_input:
                X_input[c] = np.nan
        X_input = X_input[expected_cols]

        # FIX DTYPES (CRITICAL)
        pre = model.named_steps["preprocessor"]
        num_cols = pre.transformers_[0][2]
        cat_cols = pre.transformers_[1][2]

        for c in num_cols:
            if c in X_input:
                X_input[c] = pd.to_numeric(X_input[c], errors="coerce")
        for c in cat_cols:
            if c in X_input:
                X_input[c] = X_input[c].astype(object)

        pred = model.predict(X_input)[0]
        st.subheader("Prediction Result")
        st.success(pred)

    # ================= EDA =================
    st.header("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Feature Distributions", "Target Analysis", "Feature Correlations"]
    )

    with tab1:
        st.dataframe(df.head())
        st.dataframe(df.describe(include="all"))

    with tab2:
        for i, col in enumerate(df.select_dtypes("number").columns):
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, key=f"dist_{col}_{i}")

    with tab3:
        if df[target].dtype == "object":
            vc = df[target].value_counts().reset_index()
            vc.columns = ["value", "count"]
            st.plotly_chart(px.bar(vc, x="value", y="count"), key="target_bar")
        else:
            st.plotly_chart(px.histogram(df, x=target), key="target_hist")

    with tab4:
        num_df = df.select_dtypes("number")
        if len(num_df.columns) > 1:
            st.plotly_chart(px.imshow(num_df.corr(), text_auto=True), key="corr_plot")

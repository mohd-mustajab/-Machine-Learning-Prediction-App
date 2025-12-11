# app.py (with Model Comparison)
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
from glob import glob
from data_loader import load_dataset, list_expected_files
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

st.set_page_config(layout="wide", page_title="ML Prediction App")

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

# Dataset groups
CLASS_DATASETS = {"Titanic": "titanic", "Zoo": "zoo"}
REG_DATASETS = {"Salary Data": "salary_data", "Insurance": "insurance"}

ALGS_CLASS = ["logistic", "decision_tree", "rf"]
ALGS_REG = ["linear", "rf", "ridge", "lasso"]

# helpers --------------------------------------------------------------------
def find_saved_models_for_dataset(dataset_key):
    """Return list of alg names for which models exist (models/{dataset}_{alg}.pkl)."""
    pattern = os.path.join(MODELS_DIR, f"{dataset_key}_*.pkl")
    files = glob(pattern)
    algs = []
    for f in files:
        base = os.path.basename(f).replace(".pkl", "")
        try:
            _, alg = base.rsplit("_", 1)
            algs.append(alg)
        except ValueError:
            continue
    return sorted(algs)

def load_metrics(dataset_key, alg):
    p = os.path.join(OUTPUTS_DIR, f"metrics_{dataset_key}_{alg}.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_predictions_df(dataset_key, alg):
    p = os.path.join(OUTPUTS_DIR, f"predictions_{dataset_key}_{alg}.csv")
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def load_schema(dataset_key, alg):
    p = os.path.join(MODELS_DIR, f"{dataset_key}_{alg}_schema.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def pretty(val):
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

# session init ---------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "main"
if "config" not in st.session_state:
    st.session_state.config = None

# MAIN PAGE -----------------------------------------------------------------
if st.session_state.page == "main":
    st.title("Machine Learning Prediction App")

    left, mid, right = st.columns([1,2,1])

    # LEFT: configuration
    with left:
        st.header("Configuration")
        task = st.selectbox("Task type", ["Classification", "Regression"])

        dataset_map = CLASS_DATASETS if task == "Classification" else REG_DATASETS
        dataset_label = st.selectbox("Dataset", list(dataset_map.keys()))
        dataset_key = dataset_map[dataset_label]

        algs_allowed = ALGS_CLASS if task == "Classification" else ALGS_REG
        alg_choice = st.selectbox("Algorithm", algs_allowed)

        if st.button("Proceed to Prediction"):
            st.session_state.config = {"task": task, "dataset_label": dataset_label, "dataset_key": dataset_key, "algorithm": alg_choice}
            st.session_state.page = "prediction"
            st.rerun()

        st.markdown("---")
        st.subheader("Model Comparison")
        if st.button("Compare saved models for this dataset"):
            st.session_state.compare_request = {"dataset_key": dataset_key, "task": task}
            st.rerun()

    # MIDDLE: algorithm info
    with mid:
        st.header("Algorithms (info)")
        with st.expander("Logistic Regression"):
            st.write("Linear classifier typically used for binary classification.")
        with st.expander("Decision Tree"):
            st.write("Interpretable tree-based classifier.")
        with st.expander("Random Forest"):
            st.write("Ensemble of trees; works for both tasks.")
        with st.expander("Linear / Ridge / Lasso"):
            st.write("Linear regression models; Ridge=L2, Lasso=L1 regularization.")

# MODEL COMPARISON VIEW (triggered from main or prediction) ------------------
if st.session_state.get("compare_request"):
    req = st.session_state.pop("compare_request")
    dkey = req["dataset_key"]
    task = req["task"]
    st.header(f"Model comparison — dataset: {dkey} ({task})")

    saved_algs = find_saved_models_for_dataset(dkey)
    if not saved_algs:
        st.info("No trained models found for this dataset. Run training first.")
    else:
        # Build metrics table
        rows = []
        for alg in saved_algs:
            metrics = load_metrics(dkey, alg) or {}
            row = {"algorithm": alg}
            if task == "Classification":
                row["accuracy"] = metrics.get("accuracy", None)
            else:
                row["r2"] = metrics.get("r2", None)
                row["mse"] = metrics.get("mse", None)
            rows.append(row)
        df_metrics = pd.DataFrame(rows).set_index("algorithm")
        st.subheader("Summary metrics")
        st.dataframe(df_metrics)

        # choose primary metric and plot
        if task == "Classification":
            metric_col = "accuracy"
            st.subheader("Accuracy comparison")
            if df_metrics[metric_col].notna().any():
                fig = px.bar(df_metrics.reset_index(), x="algorithm", y=metric_col, text=metric_col, title="Accuracy by algorithm")
                st.plotly_chart(fig)
        else:
            metric_col = "r2"
            st.subheader("R² comparison")
            if df_metrics[metric_col].notna().any():
                fig = px.bar(df_metrics.reset_index(), x="algorithm", y=metric_col, text=metric_col, title="R² by algorithm")
                st.plotly_chart(fig)
    st.markdown("---")
    if st.button("Back to main"):
        st.rerun()

# PREDICTION PAGE -----------------------------------------------------------
if st.session_state.page == "prediction":
    cfg = st.session_state.get("config")
    if not cfg:
        st.error("No configuration found.")
        st.stop()

    task = cfg["task"]; dataset_label = cfg["dataset_label"]; dataset_key = cfg["dataset_key"]; alg = cfg["algorithm"]
    st.title(f"Prediction — {dataset_label} ({alg})")

    # Back button
    if st.button("Back to Main"):
        st.session_state.page = "main"
        st.rerun()

    # load model + schema
    model_path = os.path.join(MODELS_DIR, f"{dataset_key}_{alg}.pkl")
    schema_path = os.path.join(MODELS_DIR, f"{dataset_key}_{alg}_schema.json")
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)
    schema = load_schema(dataset_key, alg)

    # load raw dataset for EDA and range defaults
    try:
        df_raw = load_dataset(dataset_key)
    except Exception as e:
        df_raw = None
        st.warning(f"Could not load raw dataset for EDA: {e}")

    # Left sidebar: Inputs form
    with st.sidebar.form("predict_form"):
        st.header("Inputs")
        input_vals = {}
        if schema:
            for feat in schema.get("features", []):
                name = feat["name"]; ftype = feat.get("type", "numeric")
                default = 0
                tooltip = ""
                if df_raw is not None and name in df_raw.columns:
                    col = df_raw[name]
                    try:
                        mn = float(col.min()); mx = float(col.max()); md = float(col.median()) if pd.api.types.is_numeric_dtype(col) else None
                        tooltip = f"Range: {mn} → {mx}"
                        if md is not None:
                            default = md
                    except Exception:
                        tooltip = ""
                label = f"{name} — {tooltip}" if tooltip else name
                if ftype == "integer":
                    input_vals[name] = st.number_input(label, value=int(default), step=1, format="%d")
                elif ftype == "numeric":
                    input_vals[name] = st.number_input(label, value=float(default))
                else:
                    if df_raw is not None and name in df_raw.columns:
                        uniqs = list(df_raw[name].dropna().unique())
                        if 1 < len(uniqs) <= 200:
                            input_vals[name] = st.selectbox(name, options=uniqs)
                        else:
                            input_vals[name] = st.text_input(name)
                    else:
                        input_vals[name] = st.text_input(name)
        else:
            st.write("No schema found. Provide raw inputs as name=value pairs.")
            raw = st.text_area("Raw inputs (name=value comma-separated)")
            input_vals = {"__raw__": raw}

        submitted = st.form_submit_button("Predict")

    # Main content: prediction & metrics & EDA
    if submitted:
        try:
            if "__raw__" in input_vals:
                raw = input_vals["__raw__"]
                row = {}
                for it in [x.strip() for x in raw.split(",") if x.strip()]:
                    if "=" in it:
                        k,v = it.split("=",1); row[k.strip()] = v.strip()
                X = pd.DataFrame([row])
            else:
                X = pd.DataFrame([input_vals], columns=[f["name"] for f in schema.get("features", [])])
                # coerce numeric types
                for f in schema.get("features", []):
                    n=f["name"]; t=f.get("type","numeric")
                    if t=="integer": X[n] = pd.to_numeric(X[n], errors="coerce").astype("Int64")
                    if t=="numeric": X[n] = pd.to_numeric(X[n], errors="coerce").astype(float)
            pred = model.predict(X)
            st.subheader("Prediction Result")
            if task == "Classification":
                st.success(f"Predicted class: {pred[0]}")
            else:
                st.success(f"Predicted value: {pred[0]:.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Model performance display
    st.subheader("Model performance")
    metrics = load_metrics(dataset_key, alg)
    if metrics:
        if task == "Classification":
            if "accuracy" in metrics:
                st.metric("Accuracy", pretty(metrics["accuracy"]))
            st.json(metrics)
        else:
            if "r2" in metrics:
                st.metric("R²", pretty(metrics["r2"]))
            if "mse" in metrics:
                st.write(f"MSE: {pretty(metrics['mse'])}")
            st.json(metrics)
    else:
        st.info("No saved metrics available for this model.")

    # EDA tabs
    st.header("Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Overview","Feature Distributions","Target Analysis","Correlations"])

    with tab1:
        if df_raw is not None:
            st.write("Shape:", df_raw.shape)
            st.dataframe(df_raw.head(10))
            st.dataframe(df_raw.describe(include="all").transpose())
        else:
            st.write("No raw dataset available for overview.")

    with tab2:
        if df_raw is not None:
            numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
            for c in numeric_cols[:8]:
                st.plotly_chart(px.histogram(df_raw, x=c, nbins=30, title=f"Distribution: {c}"))
        else:
            st.write("No raw dataset available.")

    with tab3:
        if df_raw is not None:
            target = schema.get("target") if schema else None
            if not target:
                for cand in ("Survived","survived","Salary","expenses","charges","animal_name"):
                    if cand in df_raw.columns:
                        target = cand; break
            if target and target in df_raw.columns:
                if pd.api.types.is_numeric_dtype(df_raw[target]):
                    st.plotly_chart(px.histogram(df_raw, x=target, nbins=40, title=f"Target: {target}"))
                else:
                    vc = df_raw[target].value_counts().reset_index(); vc.columns=["value","count"]
                    st.plotly_chart(px.bar(vc, x="value", y="count", title=f"Target counts: {target}"))
            else:
                st.write("Target not found for analysis.")
        else:
            st.write("No raw dataset available.")

    with tab4:
        if df_raw is not None:
            numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
            if len(numeric_cols) >= 2:
                st.plotly_chart(px.imshow(df_raw[numeric_cols].corr(), text_auto=True, aspect="auto", title="Numeric feature correlations"))
            else:
                st.write("Not enough numeric columns for correlation matrix.")

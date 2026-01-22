# train.py
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from models import (
    get_classification_model,
    get_regression_model,
    save_model
)

# -------------------------------
# Output directories
# -------------------------------
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# -------------------------------
# SAFE schema inference
# -------------------------------
def infer_schema(df: pd.DataFrame, target: str):
    """
    Infer feature schema for frontend input generation.
    Safe for Python 3.13 and mixed dtypes.
    """
    features = []

    for col in df.columns:
        if col == target:
            continue

        s = df[col].dropna()

        if pd.api.types.is_numeric_dtype(s):
            # integer-like numeric check
            if len(s) > 0 and (s % 1 == 0).all():
                ftype = "integer"
            else:
                ftype = "numeric"
        else:
            ftype = "categorical"

        features.append({
            "name": col,
            "type": ftype
        })

    return {
        "features": features,
        "target": target
    }

# -------------------------------
# Preprocessor builder
# -------------------------------
def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

# -------------------------------
# Training function
# -------------------------------
def train_and_save(csv_path, dataset, target, alg, task):
    print(f"\n🔹 Training {dataset} | Algorithm: {alg} | Task: {task}")

    # Load dataset
    df = pd.read_csv(csv_path)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")

    # Drop rows with missing target
    df = df.dropna(subset=[target]).reset_index(drop=True)

    X = df.drop(columns=[target])
    y = df[target]

    # Build preprocessing
    preprocessor = build_preprocessor(X)

    # Select model
    if task == "classification":
        model = get_classification_model(alg)
    else:
        model = get_regression_model(alg)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # -------------------------------
    # SAFE train-test split
    # -------------------------------
    stratify_arg = None
    if task == "classification":
        class_counts = y.value_counts()
        if class_counts.min() >= 2:
            stratify_arg = y
        else:
            print("⚠ Stratification disabled (some classes have only 1 sample)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg
    )

    # Train
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    # -------------------------------
    # Save model
    # -------------------------------
    model_path = MODELS_DIR / f"{dataset}_{alg}.pkl"
    save_model(pipeline, model_path)

    # -------------------------------
    # Save schema (RAW features)
    # -------------------------------
    schema = infer_schema(df, target)
    with open(MODELS_DIR / f"{dataset}_{alg}_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # -------------------------------
    # Metrics
    # -------------------------------
    metrics = {}
    if task == "classification":
        metrics["accuracy"] = float(accuracy_score(y_test, preds))
    else:
        metrics["r2"] = float(r2_score(y_test, preds))
        metrics["mse"] = float(mean_squared_error(y_test, preds))

    with open(OUTPUTS_DIR / f"metrics_{dataset}_{alg}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # -------------------------------
    # Save predictions
    # -------------------------------
    out = X_test.copy()
    out["_true"] = y_test.values
    out["_pred"] = preds
    out.to_csv(OUTPUTS_DIR / f"predictions_{dataset}_{alg}.csv", index=False)

    print(f"✔ Training complete: {dataset}_{alg}")

# -------------------------------
# CLI entry point
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset key (titanic, zoo, salary_data, insurance)")
    parser.add_argument("--preprocessed", required=True, help="path to CSV file")
    parser.add_argument("--target", required=True, help="target column")
    parser.add_argument("--alg", required=True, help="algorithm name")
    parser.add_argument("--task", required=True, choices=["classification", "regression"])
    args = parser.parse_args()

    train_and_save(
        csv_path=args.preprocessed,
        dataset=args.dataset,
        target=args.target,
        alg=args.alg,
        task=args.task
    )

# train.py
"""
Train a model on a preprocessed CSV and save model + schema + test predictions + metrics.

Usage examples:
python train.py --dataset titanic --preprocessed data/titanic_cleaned.csv --target Survived --alg logistic --task classification
python train.py --dataset insurance --preprocessed data/insurance_cleaned.csv --target expenses --alg rf --task regression
"""
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from models import get_classification_pipeline, get_regression_pipeline, save_model

OUT_MODELS = Path("models")
OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def infer_schema(df: pd.DataFrame, target_col: str) -> dict:
    features = []
    for c in df.columns:
        if c == target_col:
            continue
        s = df[c]
        if pd.api.types.is_integer_dtype(s):
            t = "integer"
        elif pd.api.types.is_float_dtype(s) or pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            if not non_null.empty and np.all(np.equal(np.mod(non_null.values, 1), 0)):
                t = "integer"
            else:
                t = "numeric"
        else:
            t = "categorical"
        features.append({"name": c, "type": t})
    return {"features": features, "target": target_col}

def train_and_save(preprocessed_csv: str, target_col: str, dataset_key: str, alg: str,
                   task: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(preprocessed_csv)
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found in CSV columns: {df.columns.tolist()}")

    # drop rows with missing target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # validate task vs target dtype
    is_numeric_target = pd.api.types.is_numeric_dtype(y)
    n_unique = y.dropna().nunique()
    maybe_discrete_numeric = is_numeric_target and n_unique <= 20

    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")

    if task == "classification":
        if is_numeric_target and not maybe_discrete_numeric:
            raise ValueError(
                "Target appears continuous (many unique numeric values) but task is classification.\n"
                "Either set --task regression or convert the target to discrete classes."
            )
    else: # regression
        if not is_numeric_target:
            raise ValueError(
                "Target appears non-numeric (categorical) but task is regression.\n"
                "Either set --task classification or convert the target to numeric."
            )

    # save schema
    schema = infer_schema(df, target_col)
    schema_path = OUT_MODELS / f"{dataset_key}_{alg}_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"Saved schema -> {schema_path}")

    # select pipeline by explicit task
    if task == "classification":
        pipe = get_classification_pipeline(alg)
    else:
        pipe = get_regression_pipeline(alg)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("Training model...")
    pipe.fit(X_train, y_train)
    print("Training complete.")

    model_path = OUT_MODELS / f"{dataset_key}_{alg}.pkl"
    save_model(pipe, model_path)
    print(f"Saved model -> {model_path}")

    preds = pipe.predict(X_test)
    out = X_test.copy()
    out["_true"] = y_test.values
    out["_pred"] = preds
    preds_path = OUT_DIR / f"predictions_{dataset_key}_{alg}.csv"
    out.to_csv(preds_path, index=False)
    print(f"Saved test predictions -> {preds_path}")

    metrics = {}
    if task == "classification":
        try:
            metrics["accuracy"] = float(accuracy_score(y_test, preds))
        except Exception:
            pass
    else:
        metrics["r2"] = float(r2_score(y_test, preds))
        metrics["mse"] = float(mean_squared_error(y_test, preds))

    metrics_path = OUT_DIR / f"metrics_{dataset_key}_{alg}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")

    return {
        "model_path": str(model_path),
        "schema_path": str(schema_path),
        "preds_path": str(preds_path),
        "metrics_path": str(metrics_path),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="logical dataset key (e.g., titanic)")
    parser.add_argument("--preprocessed", required=True, help="path to preprocessed CSV")
    parser.add_argument("--target", required=True, help="target column name")
    parser.add_argument("--alg", required=True, help="algorithm, see README for allowed values")
    parser.add_argument("--task", required=True, choices=["classification","regression"], help="task: classification or regression")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    res = train_and_save(preprocessed_csv=args.preprocessed,
                         target_col=args.target,
                         dataset_key=args.dataset,
                         alg=args.alg,
                         task=args.task,
                         test_size=args.test_size,
                         random_state=args.random_state)
    print("Artifacts:", res)

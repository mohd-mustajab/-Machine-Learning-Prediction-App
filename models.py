# models.py
"""
Model-only pipelines (no preprocessing). Supported algorithms:

Classification:
  - logistic        -> LogisticRegression
  - decision_tree   -> DecisionTreeClassifier
  - rf              -> RandomForestClassifier

Regression:
  - linear          -> LinearRegression
  - rf              -> RandomForestRegressor
  - ridge           -> Ridge
  - lasso           -> Lasso

Provides:
  - get_classification_pipeline(alg)
  - get_regression_pipeline(alg)
  - save_model(pipe, path)
  - load_model(path)
"""
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def get_classification_pipeline(algorithm: str = "logistic"):
    a = algorithm.lower()
    if a == "logistic":
        clf = LogisticRegression(max_iter=2000)
    elif a == "decision_tree":
        clf = DecisionTreeClassifier(random_state=42)
    elif a == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unsupported classification algorithm. Choose: 'logistic', 'decision_tree', 'rf'.")
    return Pipeline([("clf", clf)])

def get_regression_pipeline(algorithm: str = "linear"):
    a = algorithm.lower()
    if a == "linear":
        reg = LinearRegression()
    elif a == "rf":
        reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    elif a == "ridge":
        reg = Ridge()
    elif a == "lasso":
        reg = Lasso()
    else:
        raise ValueError("Unsupported regression algorithm. Choose: 'linear', 'rf', 'ridge', 'lasso'.")
    return Pipeline([("reg", reg)])

def save_model(pipe, path: str):
    joblib.dump(pipe, path)

def load_model(path: str):
    return joblib.load(path)

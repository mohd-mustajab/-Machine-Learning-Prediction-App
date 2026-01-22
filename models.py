# models.py
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# -------- Classification Models --------
def get_classification_model(algorithm: str):
    a = algorithm.lower()
    if a == "logistic":
        return LogisticRegression(max_iter=2000)
    elif a == "decision_tree":
        return DecisionTreeClassifier(random_state=42)
    elif a == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unsupported classification algorithm")


# -------- Regression Models --------
def get_regression_model(algorithm: str):
    a = algorithm.lower()
    if a == "linear":
        return LinearRegression()
    elif a == "ridge":
        return Ridge()
    elif a == "lasso":
        return Lasso()
    elif a == "rf":
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unsupported regression algorithm")


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)

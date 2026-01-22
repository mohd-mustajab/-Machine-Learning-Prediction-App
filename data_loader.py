# data_loader.py
from pathlib import Path
import pandas as pd

DATASET_FILES = {
    "titanic": "titanic_cleaned.csv",
    "zoo": "zoo_data-classification.csv",
    "salary_data": "Salary_Data.csv",
    "insurance": "insurance_cleaned.csv",
}

SEARCH_DIRS = [Path("data"), Path("."), Path("/mnt/data")]


def load_dataset(name: str) -> pd.DataFrame:
    name = name.lower()
    if name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset {name}")

    fname = DATASET_FILES[name]
    for d in SEARCH_DIRS:
        p = d / fname
        if p.exists():
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            return df

    raise FileNotFoundError(f"{fname} not found in data/")

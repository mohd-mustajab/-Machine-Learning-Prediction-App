# data_loader.py
from pathlib import Path
import pandas as pd
from typing import Optional, List, Tuple, Dict

# EXACT DATASET → FILENAME MAP (ONLY THESE FILES WILL EVER BE USED)
DATASET_FILES: Dict[str, str] = {
    "titanic": "titanic_cleaned.csv",
    "zoo": "zoo_data-classification.csv",
    "salary_data": "Salary_Data.csv",
    "insurance": "insurance_cleaned.csv",
}

# Directories to search
SEARCH_DIRS = [
    Path("data"),
    Path("/mnt/data"),
    Path("."),
]

def _candidate_paths(filename: str, explicit_path: Optional[str] = None) -> List[Path]:
    candidates = []

    # If explicit path is file, try it first
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return [p]
        if p.is_dir():
            candidates.append(p / filename)

    # search in known directories
    for d in SEARCH_DIRS:
        candidates.append(d / filename)

    # direct filename
    candidates.append(Path(filename))

    # remove duplicates
    seen = set()
    uniq = []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq

def _read_csv(path: Path) -> pd.DataFrame:
    """Try reading CSV with common encodings and separators."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    seps = [",", ";", "\t"]
    encs = ["utf-8", "utf-8-sig", "latin1"]

    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                if df.shape[1] >= 1:
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            except Exception:
                pass

    # final fallback
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_local_csv(filename: str, explicit_path: Optional[str] = None) -> pd.DataFrame:
    """Try reading the file from multiple directories."""
    candidates = _candidate_paths(filename, explicit_path)
    last_err = None

    for c in candidates:
        try:
            return _read_csv(c)
        except FileNotFoundError:
            continue
        except Exception as e:
            last_err = e
            if c.exists():
                raise RuntimeError(f"Found file but could not read it: {c}\nError: {e}") from e
            continue

    raise FileNotFoundError(
        f"Could not find '{filename}'. Tried:\n" +
        "\n".join(str(c) for c in candidates)
    )

# ================================
# Dataset-specific loader wrappers
# ================================

def load_titanic(local_path: Optional[str] = None) -> pd.DataFrame:
    return load_local_csv(DATASET_FILES["titanic"], local_path)

def load_zoo(local_path: Optional[str] = None) -> pd.DataFrame:
    return load_local_csv(DATASET_FILES["zoo"], local_path)

def load_salary_data(local_path: Optional[str] = None) -> pd.DataFrame:
    return load_local_csv(DATASET_FILES["salary_data"], local_path)

def load_insurance(local_path: Optional[str] = None) -> pd.DataFrame:
    return load_local_csv(DATASET_FILES["insurance"], local_path)

# ================================
# Generic loader
# ================================
def load_dataset(name: str, local_path: Optional[str] = None) -> pd.DataFrame:
    name = name.lower()
    if name == "titanic":
        return load_titanic(local_path)
    if name in ("zoo", "zoo_data"):
        return load_zoo(local_path)
    if name in ("salary_data", "salary"):
        return load_salary_data(local_path)
    if name in ("insurance", "medical_insurance"):
        return load_insurance(local_path)

    raise ValueError(f"Unknown dataset '{name}'. Allowed: {list(DATASET_FILES.keys())}")

def list_expected_files() -> dict:
    out = {}
    for key, fname in DATASET_FILES.items():
        paths = []
        for d in SEARCH_DIRS:
            paths.append(str(d / fname))
        paths.append(fname)
        out[key] = paths
    return out

if __name__ == "__main__":
    print("Expected search locations:")
    for k, v in list_expected_files().items():
        print(f"{k}:")
        for p in v:
            print("  ", p)

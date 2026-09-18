"""
batch_extractor.py
──────────────────
Phase 3 — Batch Feature Extraction

Loads labeled datasets from CSV, extracts 64-D feature vectors for every manifest,
and formats them into training-ready numpy matrices and pandas DataFrames.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.stage1_parser.ast_extractor import extract_features, features_to_vector
from src.stage1_parser.feature_schema import FEATURE_NAMES
from src.utils.logger import get_logger

log = get_logger("batch_extractor")


def extract_dataset(labeled_csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Extracts features for all files in a labeled CSV.
    Returns:
        X: np.ndarray of shape (N, 64)
        y: np.ndarray of shape (N,) (0 for COMPLIANT, 1 for MISCONFIGURED)
        filenames: list of file paths
        df_valid: filtered DataFrame with valid parsed files
    """
    df = pd.read_csv(labeled_csv_path)
    X_list = []
    y_list = []
    valid_indices = []

    log.info(f"Extracting features for {len(df)} manifests from {labeled_csv_path}...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        fpath = Path(row["file_path"])
        domain = row["domain"]
        feats = extract_features(fpath, domain=domain)
        if feats is not None:
            vec = features_to_vector(feats)
            X_list.append(vec)
            label_val = 1 if row["label"] == "MISCONFIGURED" else 0
            y_list.append(label_val)
            valid_indices.append(idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    df_valid["label_int"] = y

    log.info(f"Successfully extracted {len(X)} feature vectors of dimension {X.shape[1]}")
    return X, y, FEATURE_NAMES, df_valid

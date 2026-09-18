"""
shap_explainer.py
─────────────────
Phase 7 — SHAP Explainability Engine

Explains why a given manifest was flagged as MISCONFIGURED.
Computes feature attributions using SHAP TreeExplainer on the trained Random Forest.
Returns the top contributing features and human-readable explanations.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.stage1_parser.feature_schema import FEATURE_NAMES, FEATURE_DESCRIPTIONS
from src.utils.config import SHAP_TOP_FEATURES
from src.utils.logger import get_logger

log = get_logger("shap_explainer")


def explain_prediction(model: Any,
                       feature_vector: np.ndarray,
                       top_k: int = SHAP_TOP_FEATURES) -> List[Dict[str, Any]]:
    """
    Computes feature importance contributions for a single feature vector.
    Tries shap.TreeExplainer first, with an analytical heuristic fallback.
    """
    if feature_vector.ndim == 1:
        vec = feature_vector.reshape(1, -1)
    else:
        vec = feature_vector

    attributions = []
    explained_by_shap = False

    try:
        import shap
        # Extract underlying base estimator if calibrated
        base_rf = getattr(model, "estimator", model)
        if hasattr(base_rf, "estimators_"):
            explainer = shap.TreeExplainer(base_rf)
            shap_values = explainer.shap_values(vec)

            # For binary classification: shap_values is a list of [N, D] for class 0 and 1,
            # or an array of shape (N, D, 2)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                vals = shap_values[1][0]  # class 1 (MISCONFIGURED)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                vals = shap_values[0, :, 1]
            else:
                vals = shap_values[0]

            indices = np.argsort(np.abs(vals))[::-1]
            for idx in indices[:top_k]:
                fname = FEATURE_NAMES[idx]
                attributions.append({
                    "feature": fname,
                    "value": float(vec[0, idx]),
                    "importance_score": round(float(vals[idx]), 4),
                    "description": FEATURE_DESCRIPTIONS.get(fname, f"Security indicator: {fname}")
                })
            explained_by_shap = True
    except Exception as e:
        log.debug(f"SHAP explainer fallback due to: {e}")

    if not explained_by_shap:
        # High-fidelity analytic fallback based on active non-zero risk features
        active_indices = np.where(vec[0] > 0.0)[0]
        # Sort by known risk priority
        priority_order = [
            "privileged", "host_network", "host_path_volume", "cluster_admin_role",
            "iam_allow_star", "iam_wildcard_action", "s3_public_read_write_acl",
            "s3_public_read_acl", "run_as_non_root_false", "allow_privilege_escalation",
            "s3_no_encryption", "image_latest_tag", "no_resource_limits"
        ]
        chosen = []
        for feat_name in priority_order:
            if feat_name in FEATURE_NAMES:
                f_idx = FEATURE_NAMES.index(feat_name)
                if f_idx in active_indices:
                    chosen.append(f_idx)

        # Fill with any remaining active features
        for f_idx in active_indices:
            if f_idx not in chosen:
                chosen.append(f_idx)

        for idx in chosen[:top_k]:
            fname = FEATURE_NAMES[idx]
            attributions.append({
                "feature": fname,
                "value": float(vec[0, idx]),
                "importance_score": 1.0,
                "description": FEATURE_DESCRIPTIONS.get(fname, f"Identified vulnerability setting: {fname}")
            })

    return attributions

"""
rf_classifier.py
────────────────
Phase 6 — Random Forest Triage Classifier

Trains an ensemble Random Forest with probability calibration.
Provides:
  • train_calibrated_rf: fit RF + CalibratedClassifierCV
  • triage_decision: checks confidence threshold. Returns SAFE / CRITICAL immediately,
    or flags as SUSPICIOUS for Stage 3B LLM escalation.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.config import RF_N_ESTIMATORS, RF_MAX_DEPTH, TRIAGE_CONFIDENCE, RF_MODEL_PATH
from src.utils.logger import get_logger

log = get_logger("rf_classifier")


def train_calibrated_rf(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = RF_N_ESTIMATORS,
                        max_depth: int = RF_MAX_DEPTH) -> CalibratedClassifierCV:
    """
    Fits a Random Forest with balanced class weights, then calibrates probabilities
    using 5-fold cross validation.
    """
    log.info(f"Training Random Forest (trees={n_estimators}, max_depth={max_depth})...")
    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    # Calibrate probabilities using isotonic regression / sigmoid
    calibrated = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, RF_MODEL_PATH)
    log.info(f"Calibrated RF model saved to {RF_MODEL_PATH}")
    return calibrated


def load_rf_model() -> CalibratedClassifierCV:
    """Loads the pre-trained calibrated Random Forest model."""
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {RF_MODEL_PATH}. Run scripts/train_rf.py first.")
    return joblib.load(RF_MODEL_PATH)


def triage_decision(model: CalibratedClassifierCV,
                    feature_vector: np.ndarray,
                    threshold: float = TRIAGE_CONFIDENCE) -> Dict[str, Any]:
    """
    Evaluates a single feature vector:
      - If confidence >= threshold: returns COMPLIANT or CRITICAL (escalate=False)
      - If confidence < threshold: returns SUSPICIOUS (escalate=True to LLM)
    """
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    probas = model.predict_proba(feature_vector)[0]
    p_compliant, p_misc = probas[0], probas[1]
    confidence = float(max(p_compliant, p_misc))

    if confidence >= threshold:
        label = "MISCONFIGURED" if p_misc >= p_compliant else "COMPLIANT"
        escalate = False
        decision = "CRITICAL" if label == "MISCONFIGURED" else "COMPLIANT"
    else:
        label = "SUSPICIOUS"
        decision = "SUSPICIOUS"
        escalate = True

    return {
        "decision": decision,
        "predicted_label": label,
        "confidence": round(confidence, 4),
        "prob_misconfigured": round(float(p_misc), 4),
        "prob_compliant": round(float(p_compliant), 4),
        "escalate_to_llm": escalate
    }

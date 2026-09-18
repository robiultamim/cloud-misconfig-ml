"""
train_rf.py
───────────
Phase 6 Training Script

Loads the combined benchmark CSV, extracts AST feature vectors,
splits into 80/20 train/test sets, fits the calibrated Random Forest,
and outputs classification metrics.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import LABELED_DIR
from src.stage1_parser.batch_extractor import extract_dataset
from src.stage3_triage.rf_classifier import train_calibrated_rf, triage_decision
from src.utils.logger import get_logger

log = get_logger("train_rf")


def main():
    combined_csv = LABELED_DIR / "combined_benchmark.csv"
    if not combined_csv.exists():
        log.error(f"Dataset not found at {combined_csv}. Run scripts/run_labelling.py first.")
        sys.exit(1)

    # 1. Extract feature matrix
    X, y, feature_names, df_valid = extract_dataset(combined_csv)

    # Save extracted feature matrix for fast loading by GNN and other stages
    np.savez_compressed(LABELED_DIR / "features_dataset.npz", X=X, y=y)
    df_valid.to_csv(LABELED_DIR / "features_metadata.csv", index=False)
    log.info(f"Saved pre-extracted features to {LABELED_DIR / 'features_dataset.npz'}")

    # 2. Train / Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df_valid, test_size=0.20, random_state=42, stratify=y
    )
    log.info(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Train Calibrated Random Forest
    model = train_calibrated_rf(X_train, y_train)

    # 4. Evaluate on Test Set
    probas = model.predict_proba(X_test)[:, 1]
    y_pred = (probas >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, target_names=["COMPLIANT", "MISCONFIGURED"], digits=4)
    auc = roc_auc_score(y_test, probas)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "="*55)
    print("RANDOM FOREST TEST EVALUATION")
    print("="*55)
    print(report)
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("="*55)

    # 5. Triage Escalation Analysis
    escalation_count = 0
    for i in range(len(X_test)):
        res = triage_decision(model, X_test[i], threshold=0.90)
        if res["escalate_to_llm"]:
            escalation_count += 1

    esc_pct = (escalation_count / len(X_test)) * 100
    print(f"Triage Escalation Rate (Threshold=0.90): {escalation_count}/{len(X_test)} ({esc_pct:.1f}%)")
    print(f"Directly classified without LLM: {len(X_test) - escalation_count}/{len(X_test)} ({100 - esc_pct:.1f}%)")
    print(f"API Cost Reduction achieved: {100 - esc_pct:.1f}%")
    print("="*55)


if __name__ == "__main__":
    main()

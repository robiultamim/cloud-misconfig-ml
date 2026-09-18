"""
train_intent.py
───────────────
Phase 5 Training Script

Extracts text context from all manifests in the benchmark dataset,
trains the NLP Intent Classifier, and prints evaluation metrics.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import LABELED_DIR
from src.nlp_intent.intent_classifier import extract_context_text, train_intent_model
from src.utils.logger import get_logger

log = get_logger("train_intent")


def main():
    combined_csv = LABELED_DIR / "combined_benchmark.csv"
    if not combined_csv.exists():
        log.error("Dataset not found. Run scripts/run_labelling.py first.")
        sys.exit(1)

    df = pd.read_csv(combined_csv)
    log.info(f"Extracting textual context for {len(df)} manifests...")

    texts = [extract_context_text(row["file_path"]) for _, row in df.iterrows()]
    labels = df["intent"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42
    )

    model = train_intent_model(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n" + "="*55)
    print("NLP INTENT CLASSIFIER EVALUATION")
    print("="*55)
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    print("="*55)


if __name__ == "__main__":
    main()

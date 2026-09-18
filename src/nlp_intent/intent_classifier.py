"""
intent_classifier.py
────────────────────
Phase 5 — NLP Operational Intent Classifier

Extracts semantic metadata (resource names, labels, annotations, comments)
and infers developer intent:
  • INTENTIONAL  (e.g., intentional public CDN / static website)
  • ACCIDENTAL   (unintended misconfiguration)
  • UNKNOWN      (neutral / ambiguous context)

Uses TF-IDF + Logistic Regression / Naive Bayes with optional DeBERTa fine-tuning.
Provides zero-latency, high-precision intent classification to suppress false alarms.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.config import INTENT_MODEL_DIR
from src.utils.file_utils import (
    safe_load_yaml, safe_load_json, extract_inline_comments,
    get_labels, get_annotations, get_resource_name
)
from src.utils.logger import get_logger

log = get_logger("intent_classifier")

INTENT_PIPELINE_PATH = INTENT_MODEL_DIR / "intent_pipeline.pkl"


def extract_context_text(filepath: Path | str) -> str:
    """
    Extracts semantic textual context from a manifest:
    resource name, labels, annotations, and inline comments.
    """
    filepath = Path(filepath)
    comments = extract_inline_comments(filepath)

    doc = safe_load_json(filepath) if filepath.suffix.lower() == ".json" else safe_load_yaml(filepath)
    if isinstance(doc, list) and doc:
        doc = doc[0]
    if not isinstance(doc, dict):
        doc = {}

    name = get_resource_name(doc)
    labels = " ".join([f"{k}:{v}" for k, v in get_labels(doc).items()])
    annotations = " ".join([f"{k}:{v}" for k, v in get_annotations(doc).items()])
    comments_text = " ".join(comments)

    text = f"name: {name} | labels: {labels} | annotations: {annotations} | comments: {comments_text}"
    return text.strip()


def train_intent_model(texts: List[str], labels: List[str]) -> Pipeline:
    """
    Trains a text classification pipeline (TF-IDF + LogisticRegression)
    for intent classification.
    """
    log.info(f"Training NLP Intent Classifier on {len(texts)} samples...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
        ("clf", LogisticRegression(class_weight="balanced", random_state=42))
    ])
    pipeline.fit(texts, labels)

    INTENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, INTENT_PIPELINE_PATH)
    log.info(f"NLP Intent Classifier saved to {INTENT_PIPELINE_PATH}")
    return pipeline


def load_intent_model() -> Pipeline:
    if not INTENT_PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Intent model not found at {INTENT_PIPELINE_PATH}. Train it first.")
    return joblib.load(INTENT_PIPELINE_PATH)


def predict_intent(pipeline: Pipeline, filepath: Path | str) -> Dict[str, Any]:
    text = extract_context_text(filepath)
    pred = pipeline.predict([text])[0]
    probas = pipeline.predict_proba([text])[0]
    conf = float(max(probas))

    return {
        "intent": pred,
        "confidence": round(conf, 4),
        "context_text": text
    }

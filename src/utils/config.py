"""
config.py — Central configuration for all pipeline stages.
Edit paths and model parameters here; everything else imports from this file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Load .env ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

# ─── API Keys ─────────────────────────────────────────────────────────────────
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
GOOGLE_KEY    = os.getenv("GOOGLE_API_KEY", "")

# ─── Data Paths ───────────────────────────────────────────────────────────────
DATA_DIR        = BASE_DIR / "data"
RAW_K8S_DIR     = DATA_DIR / "raw" / "kubernetes"
RAW_IAM_DIR     = DATA_DIR / "raw" / "iam"
RAW_SAM_DIR     = DATA_DIR / "raw" / "serverless"
LABELED_DIR     = DATA_DIR / "labeled"
GRAPHS_DIR      = DATA_DIR / "graphs"

LABEL_K8S_CSV   = LABELED_DIR / "kubernetes.csv"
LABEL_IAM_CSV   = LABELED_DIR / "iam.csv"
LABEL_SAM_CSV   = LABELED_DIR / "serverless.csv"

# ─── Model Paths ──────────────────────────────────────────────────────────────
MODELS_DIR      = BASE_DIR / "models"
RF_MODEL_PATH   = MODELS_DIR / "rf_calibrated.pkl"
GNN_MODEL_PATH  = MODELS_DIR / "gnn_security.pt"
INTENT_MODEL_DIR= MODELS_DIR / "intent_classifier"

# ─── Stage 1: Feature Extraction ──────────────────────────────────────────────
FEATURE_VECTOR_DIM = 64

# ─── Stage 2: GNN ─────────────────────────────────────────────────────────────
GNN_HIDDEN_DIM   = 128
GNN_NUM_EPOCHS   = 200
GNN_LEARNING_RATE= 0.001
GNN_WEIGHT_DECAY = 5e-4

# ─── Stage 3: Triage ──────────────────────────────────────────────────────────
RF_N_ESTIMATORS     = 500
RF_MAX_DEPTH        = 20
TRIAGE_CONFIDENCE   = 0.90   # RF confidence threshold before escalating to LLM
LLM_PROVIDER        = "gemini"  # "gemini" or "openai"
LLM_MODEL_GEMINI    = "gemini-1.5-flash"
LLM_MODEL_OPENAI    = "gpt-4o"
LLM_TEMPERATURE     = 0.1

# ─── Stage 4: Remediation ─────────────────────────────────────────────────────
COMPILER_MAX_RETRIES = 3
SHAP_TOP_FEATURES    = 5

# ─── Domains ──────────────────────────────────────────────────────────────────
DOMAINS = ["kubernetes", "iam", "serverless"]
SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
INTENT_LABELS   = ["INTENTIONAL", "ACCIDENTAL", "UNKNOWN"]
TRIAGE_LABELS   = ["COMPLIANT", "SUSPICIOUS", "CRITICAL"]

# ─── Ensure all directories exist ─────────────────────────────────────────────
for _d in [RAW_K8S_DIR, RAW_IAM_DIR, RAW_SAM_DIR,
           LABELED_DIR, GRAPHS_DIR, MODELS_DIR, INTENT_MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

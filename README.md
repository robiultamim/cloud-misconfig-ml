# Cloud Misconfiguration Detection Using Machine Learning
An intelligent, multi-domain framework for detecting, explaining, and remediating cloud security misconfigurations across **Kubernetes Workloads**, **IAM Identity Policies**, and **Cloud Storage (S3)**.

---

## 🚀 Quick Start (For Evaluators & Teachers)

You can run an audit on sample manifests immediately without setting up API keys or training models from scratch (pre-trained model weights are included!).

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone [https://github.com/robiul-tamim/cloud-misconfig-ml.git](https://github.com/robiultamim/cloud-misconfig-ml.git)
cd cloud-misconfig-ml

# Create virtual environment (Python 3.10+ recommended)
python -m venv venv

# Activate virtual environment:
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Audit (1-Command Demo)

Run the audit on a misconfigured Kubernetes manifest to see **Classification, GNN Threat Evaluation, SHAP Attribution, and Compiler-Validated YAML Patch Generation**:
```bash
python scripts/run_pipeline.py --input data/raw/kubernetes/k8s_MISCONFIGURED_0000.yaml
```

Run an audit on a batch of files:
```bash
python scripts/run_pipeline.py --input data/raw/kubernetes/ --limit 5
```

---

## 📊 Interactive Jupyter Notebook
To see all experimental details, data visualizations, and step-by-step pipeline evaluations:
```bash
jupyter notebook notebooks/cloud_misconfig_experiments.ipynb
```

---

## 🏗️ Re-Training Models (Optional)
All pre-trained weights are located in `models/`. If you wish to re-train the models from scratch:

```bash
# 1. Re-run Consensus Labelling on raw manifests
python scripts/run_labelling.py

# 2. Train the Calibrated Random Forest Triage Model
python scripts/train_rf.py

# 3. Train the Heterogeneous Graph Neural Network (GNN)
python scripts/train_gnn.py

# 4. Train the NLP Intent Classifier
python scripts/train_intent.py
```

---

## 📂 Repository Structure

```
cloud-misconfig-ml/
├── data/
│   ├── raw/                 # Sample raw YAML/JSON manifests (K8s, IAM, SAM)
│   └── labeled/             # Labeled benchmark datasets & pre-extracted features (.npz, .csv)
├── models/                  # Pre-trained model weights
│   ├── rf_calibrated.pkl    # Calibrated Random Forest triage model (~5 MB)
│   ├── gnn_security.pt      # PyTorch Heterogeneous GNN weights (~600 KB)
│   └── intent_classifier/   # NLP Intent classifier model (~73 KB)
├── notebooks/
│   └── cloud_misconfig_experiments.ipynb  # Interactive experiments notebook
├── scripts/
│   ├── run_pipeline.py      # Main end-to-end inference entry point
│   ├── run_labelling.py     # Consensus multi-scanner labeling runner
│   ├── train_rf.py          # Random Forest training script
│   ├── train_gnn.py         # GNN training script
│   └── train_intent.py      # NLP Intent training script
├── src/                     # Core source code modules
│   ├── stage1_parser/       # AST 64-D feature extraction
│   ├── stage2_graph/        # Heterogeneous security graph & GNN architecture
│   ├── stage3_triage/       # Random Forest triage & LLM client fallback
│   ├── stage4_remediation/  # SHAP XAI & Compiler-in-the-loop validation
│   ├── nlp_intent/          # Semantic intent understanding
│   └── utils/               # File parsers, logging, and configuration
├── requirements.txt         # Required Python packages
├── .gitignore               # Ignored cache/environment files
└── README.md                # Project documentation
```

---

## 👥 Authors
* **Israt Jahan Rani** (ID: 011201376)
* **Robiul Islam Tamim** (ID: 011212039)

"""
run_labelling.py
────────────────
Phase 2 — Run Consensus Labelling Pipeline

Audits all raw manifests in:
  - data/raw/kubernetes/
  - data/raw/iam/
  - data/raw/serverless/

Produces labeled CSVs:
  - data/labeled/kubernetes.csv
  - data/labeled/iam.csv
  - data/labeled/serverless.csv
  - data/labeled/combined_benchmark.csv
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import (
    RAW_K8S_DIR, RAW_IAM_DIR, RAW_SAM_DIR,
    LABEL_K8S_CSV, LABEL_IAM_CSV, LABEL_SAM_CSV, LABELED_DIR
)
from src.utils.file_utils import discover_manifests
from src.labelling.consensus_labeller import audit_manifest
from src.utils.logger import get_logger

log = get_logger("run_labelling")


def label_domain(raw_dir: Path, out_csv: Path, domain: str) -> pd.DataFrame:
    files = discover_manifests(raw_dir)
    log.info(f"Auditing {len(files)} {domain} manifests...")

    records = []
    for f in tqdm(files, desc=f"Labeling {domain}"):
        res = audit_manifest(f, domain=domain)
        records.append(res)

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    log.info(f"Saved {len(df)} records to {out_csv}")
    return df


def main():
    dfs = []

    # 1. Kubernetes
    if RAW_K8S_DIR.exists():
        df_k8s = label_domain(RAW_K8S_DIR, LABEL_K8S_CSV, "kubernetes")
        dfs.append(df_k8s)

    # 2. IAM / CloudFormation
    if RAW_IAM_DIR.exists():
        df_iam = label_domain(RAW_IAM_DIR, LABEL_IAM_CSV, "iam")
        dfs.append(df_iam)

    # 3. Serverless (SAM)
    if RAW_SAM_DIR.exists():
        df_sam = label_domain(RAW_SAM_DIR, LABEL_SAM_CSV, "serverless")
        dfs.append(df_sam)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined_path = LABELED_DIR / "combined_benchmark.csv"
        combined.to_csv(combined_path, index=False)
        log.info(f"Combined benchmark saved to {combined_path}")

        print("\n" + "="*50)
        print("LABELLING SUMMARY")
        print("="*50)
        print(combined.groupby(["domain", "label"]).size())
        print("="*50)
        print(f"Total labeled files: {len(combined)}")
        print("="*50)


if __name__ == "__main__":
    main()

"""
run_pipeline.py
───────────────
Full End-to-End Inference Pipeline

Orchestrates all 4 stages on any single manifest or directory:
  Stage 1: Multi-format AST Feature Extraction (64-D)
  Stage 2: Heterogeneous GNN Security Evaluation
  Stage 3: Tiered Random Forest Triage + NLP Intent + LLM Escalation
  Stage 4: SHAP Attribution + Compiler-Validated Patch Generation

Usage:
    python scripts/run_pipeline.py --input data/raw/kubernetes/k8s_MISCONFIGURED_0000.yaml
    python scripts/run_pipeline.py --input data/raw/kubernetes/ --limit 5
"""

import sys
import argparse
from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stage1_parser.ast_extractor import extract_features, features_to_vector
from src.stage3_triage.rf_classifier import load_rf_model, triage_decision
from src.stage2_graph.gnn_model import SecurityHeteroGNN
from src.nlp_intent.intent_classifier import load_intent_model, predict_intent
from src.stage3_triage.llm_client import audit_with_llm
from src.stage4_remediation.shap_explainer import explain_prediction
from src.stage4_remediation.patch_generator import generate_hardened_patch
from src.stage4_remediation.compiler_loop import validate_remediation_patch
from src.utils.file_utils import discover_manifests
from src.utils.config import GNN_MODEL_PATH
from src.utils.logger import get_logger

log = get_logger("pipeline")


def audit_single_file(filepath: Path, rf_model, intent_model, gnn_model=None) -> dict:
    log.info(f"\n{'='*60}\nAuditing: {filepath.name}\n{'='*60}")
    content = filepath.read_text(encoding="utf-8", errors="ignore")

    # ── Stage 1: AST Extraction ──────────────────────────────────────────
    feats = extract_features(filepath)
    if feats is None:
        log.error(f"Failed to parse manifest: {filepath}")
        return {"file": str(filepath), "status": "ERROR"}

    vec = features_to_vector(feats)

    # ── Stage 2: GNN Security Evaluation ─────────────────────────────────
    gnn_verdict = "N/A"
    if gnn_model is not None:
        with torch.no_grad():
            node_tensor = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
            proj = gnn_model.node_projections["Pod"](node_tensor)
            logits = gnn_model.classifier_pod(proj)
            gnn_pred = int(logits.argmax(dim=-1).item())
            gnn_verdict = "MISCONFIGURED" if gnn_pred == 1 else "COMPLIANT"

    # ── Stage 3: Random Forest Triage + Intent Classification ─────────────
    triage = triage_decision(rf_model, vec)
    intent_res = predict_intent(intent_model, filepath)

    verdict = triage["predicted_label"]
    confidence = triage["confidence"]
    escalated = triage["escalate_to_llm"]

    # If flagged as suspicious or intentional, escalate / adjust
    if intent_res["intent"] == "INTENTIONAL" and verdict == "MISCONFIGURED":
        log.info("NLP Intent detected INTENTIONAL public usage — evaluating suppression.")
        escalated = True

    llm_findings = None
    if escalated:
        log.info("Escalating to Stage 3B LLM Reasoning Engine...")
        attributions = explain_prediction(rf_model, vec)
        domain = "kubernetes" if "k8s" in filepath.name.lower() else "cloud"
        llm_findings = audit_with_llm(content, domain=domain, top_features=attributions)
        verdict = llm_findings["final_verdict"]
        confidence = llm_findings["confidence"]

    # ── Stage 4: SHAP Attribution & Compiler Remediation ─────────────────
    shap_report = []
    patch_result = None
    if verdict == "MISCONFIGURED":
        shap_report = explain_prediction(rf_model, vec)
        raw_patch = generate_hardened_patch(filepath, domain="kubernetes", triggering_features=shap_report)
        patch_result = validate_remediation_patch(raw_patch, domain="kubernetes", extension=filepath.suffix)

    # ── Summary Report ───────────────────────────────────────────────────
    report = {
        "file": str(filepath.name),
        "verdict": verdict,
        "confidence": confidence,
        "gnn_graph_score": gnn_verdict,
        "intent_detected": intent_res["intent"],
        "escalated_to_llm": escalated,
        "shap_attributions": shap_report[:3],
        "compiler_validated_patch": patch_result["valid"] if patch_result else None
    }

    print(f"VERDICT         : {verdict} (Confidence: {confidence:.2%})")
    print(f"GNN Graph Eval  : {gnn_verdict}")
    print(f"Developer Intent: {intent_res['intent']} (Conf: {intent_res['confidence']:.2%})")
    if shap_report:
        print("\nTOP CONTRIBUTING RISKS (SHAP):")
        for i, a in enumerate(shap_report[:3], 1):
            print(f"  {i}. [{a['feature']}]: {a['description']}")
    if patch_result and patch_result["valid"]:
        print(f"\nREMEDIATION     : [SUCCESS] Compiler validated fix generated ({len(patch_result['validated_content'])} bytes).")

    return report


def main():
    parser = argparse.ArgumentParser(description="End-to-End Cloud Misconfiguration Pipeline")
    parser.add_argument("--input", type=str, required=True, help="File or directory to audit")
    parser.add_argument("--limit", type=int, default=5, help="Max files if scanning a directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    rf_model = load_rf_model()
    intent_model = load_intent_model()

    gnn_model = None
    if GNN_MODEL_PATH.exists():
        gnn_model = SecurityHeteroGNN()
        gnn_model.load_state_dict(torch.load(GNN_MODEL_PATH, weights_only=True))
        gnn_model.eval()

    if input_path.is_file():
        files = [input_path]
    else:
        files = discover_manifests(input_path)[:args.limit]

    results = []
    for f in files:
        res = audit_single_file(f, rf_model, intent_model, gnn_model)
        results.append(res)

    print(f"\n{'='*60}\nAUDIT SUMMARY ({len(results)} manifests evaluated)\n{'='*60}")
    for r in results:
        print(f"  {r['file']:<35} -> {r['verdict']} (Intent: {r['intent_detected']})")


if __name__ == "__main__":
    main()

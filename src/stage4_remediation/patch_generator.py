"""
patch_generator.py
──────────────────
Phase 8 — Automated Remediation Patch Generator

Takes the original manifest, domain, and SHAP vulnerability attributions,
and generates a hardened, compliant YAML/JSON patch.
Preserves non-security business logic while neutralizing detected risks.
"""

import sys
import copy
from pathlib import Path
from typing import Dict, Any
import yaml
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.file_utils import safe_load_yaml, safe_load_json
from src.utils.logger import get_logger

log = get_logger("patch_generator")


def generate_hardened_patch(filepath: Path | str, domain: str, triggering_features: list[dict]) -> str:
    """
    Generates a syntactically valid hardened manifest.
    Applies precise rule-based transformations informed by SHAP attributions.
    """
    filepath = Path(filepath)
    is_json = filepath.suffix.lower() == ".json"

    doc = safe_load_json(filepath) if is_json else safe_load_yaml(filepath)
    if isinstance(doc, list) and doc:
        doc = doc[0]
    if not isinstance(doc, dict):
        return filepath.read_text(encoding="utf-8")

    hardened = copy.deepcopy(doc)
    feat_names = [f.get("feature", "") for f in triggering_features]

    # 1. Kubernetes Transformations
    if domain == "kubernetes":
        kind = hardened.get("kind", "")

        # Pod or Deployment spec
        if kind in ("Deployment", "DaemonSet", "StatefulSet", "ReplicaSet"):
            pod_spec = hardened.setdefault("spec", {}).setdefault("template", {}).setdefault("spec", {})
        elif kind == "Pod":
            pod_spec = hardened.setdefault("spec", {})
        else:
            pod_spec = {}

        if pod_spec:
            # Neutralize host networking/PID
            if "host_network" in feat_names:
                pod_spec["hostNetwork"] = False
            if "host_pid" in feat_names:
                pod_spec["hostPID"] = False

            # Neutralize container security context
            containers = pod_spec.get("containers", [])
            for c in containers:
                sc = c.setdefault("securityContext", {})
                sc["privileged"] = False
                sc["allowPrivilegeEscalation"] = False
                sc["runAsNonRoot"] = True
                sc["runAsUser"] = 10001
                sc["readOnlyRootFilesystem"] = True
                sc["capabilities"] = {"drop": ["ALL"]}
                sc["seccompProfile"] = {"type": "RuntimeDefault"}

                # Set resource limits
                c.setdefault("resources", {
                    "limits": {"cpu": "500m", "memory": "256Mi"},
                    "requests": {"cpu": "100m", "memory": "64Mi"}
                })

                # Fix latest tag
                img = c.get("image", "")
                if img.endswith(":latest"):
                    c["image"] = img[:-7] + ":1.0.0"

        # RBAC ClusterRole
        if kind == "ClusterRole" and ("cluster_admin_role" in feat_names or "wildcard_verb" in feat_names):
            hardened["rules"] = [
                {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]},
                {"apiGroups": ["apps"], "resources": ["deployments"], "verbs": ["get", "list"]}
            ]

    # 2. IAM / CloudFormation Transformations
    elif domain == "iam":
        resources = hardened.get("Resources", {})
        for r_name, r_def in resources.items():
            r_type = r_def.get("Type", "")
            props = r_def.setdefault("Properties", {})

            # S3 Bucket hardening
            if "S3::Bucket" in r_type:
                props["AccessControl"] = "Private"
                props["BucketEncryption"] = {
                    "ServerSideEncryptionConfiguration": [{
                        "ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}
                    }]
                }
                props["VersioningConfiguration"] = {"Status": "Enabled"}
                props["PublicAccessBlockConfiguration"] = {
                    "BlockPublicAcls": True,
                    "BlockPublicPolicy": True,
                    "IgnorePublicAcls": True,
                    "RestrictPublicBuckets": True
                }

            # IAM Role hardening
            if "IAM::Role" in r_type or "IAM::Policy" in r_type:
                policies = props.get("Policies", [])
                for p in policies:
                    doc_stmts = p.get("PolicyDocument", {}).get("Statement", [])
                    for s in doc_stmts:
                        if s.get("Action") == "*":
                            s["Action"] = ["s3:GetObject", "s3:PutObject"]
                        if s.get("Resource") == "*":
                            s["Resource"] = "arn:aws:s3:::production-app-storage/*"

    # 3. Serverless SAM Transformations
    elif domain == "serverless":
        resources = hardened.get("Resources", {})
        for r_name, r_def in resources.items():
            props = r_def.setdefault("Properties", {})
            env = props.get("Environment", {}).get("Variables", {})
            for secret_key in list(env.keys()):
                if any(k in secret_key.upper() for k in ["PASSWORD", "SECRET", "KEY"]):
                    env[secret_key] = "{{resolve:secretsmanager:app/secret}}"

            events = props.get("Events", {})
            for ev in events.values():
                if ev.get("Type") == "Api":
                    ev.setdefault("Properties", {})["Auth"] = {"Authorizer": "CognitoAuthorizer"}

    # Format output
    if is_json:
        return json.dumps(hardened, indent=2)
    else:
        return yaml.dump(hardened, default_flow_style=False, sort_keys=False)

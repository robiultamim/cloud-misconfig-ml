"""
ast_extractor.py
────────────────
Stage 1 — Core AST Feature Extractor

Converts any cloud configuration manifest (YAML/JSON) into a 64-dimensional
numerical feature vector based on the schema defined in feature_schema.py.

Handles three domains:
  • kubernetes  — Pods, Deployments, Services, RBAC manifests
  • iam         — CloudFormation IAM::Role / IAM::Policy / AWS::S3::Bucket
  • serverless  — AWS SAM templates (AWSTemplateFormatVersion + Transform)
"""

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.stage1_parser.feature_schema import FEATURE_NAMES
from src.utils.file_utils import (
    safe_load_yaml, safe_load_json, extract_inline_comments,
    max_nesting_depth, count_keys, get_resource_kind, get_labels, get_annotations
)
from src.utils.logger import get_logger

log = get_logger("ast_extractor")


def extract_features(filepath: str | Path, domain: str = "auto") -> Optional[dict]:
    """
    Main entry point — extract a feature dict from a config file.
    Returns None if the file cannot be parsed.

    Args:
        filepath: path to .yaml / .yml / .json file
        domain:   "kubernetes", "iam", "serverless", or "auto" (auto-detect)

    Returns:
        dict with keys = FEATURE_NAMES, values = float
    """
    filepath = Path(filepath)
    features = {name: 0.0 for name in FEATURE_NAMES}

    # ── Load document ───────────────────────────────────────────────────────
    if filepath.suffix.lower() == ".json":
        doc = safe_load_json(filepath)
    else:
        doc = safe_load_yaml(filepath)

    if doc is None or not isinstance(doc, (dict, list)):
        return None

    # Flatten multi-doc YAML: take first document
    if isinstance(doc, list):
        doc = doc[0] if doc else None
    if not isinstance(doc, dict):
        return None

    # ── Auto-detect domain ──────────────────────────────────────────────────
    if domain == "auto":
        domain = _detect_domain(doc, filepath)

    # ── Structural features (all domains) ──────────────────────────────────
    comments = extract_inline_comments(filepath)
    features["has_inline_comment"]  = float(len(comments) > 0)
    features["has_labels"]          = float(bool(get_labels(doc)))
    features["has_annotations"]     = float(bool(get_annotations(doc)))
    features["nesting_depth_norm"]  = min(max_nesting_depth(doc) / 10.0, 1.0)
    features["total_keys_norm"]     = min(count_keys(doc) / 100.0, 1.0)

    # ── Domain-specific extraction ──────────────────────────────────────────
    if domain == "kubernetes":
        _extract_k8s(doc, features)
    elif domain == "iam":
        _extract_iam(doc, features)
    elif domain == "serverless":
        _extract_sam(doc, features)

    return features


def features_to_vector(features: dict) -> np.ndarray:
    """Convert feature dict to a numpy array in FEATURE_NAMES order."""
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES],
                    dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def _detect_domain(doc: dict, filepath: Path) -> str:
    kind = doc.get("kind", "")
    api  = doc.get("apiVersion", "")
    transform = doc.get("Transform", "")
    resources = doc.get("Resources", {})

    if kind or "kubernetes.io" in api:
        return "kubernetes"
    if "AWS::Serverless" in str(transform) or "AWS::Serverless" in str(resources):
        return "serverless"
    if "AWSTemplateFormatVersion" in doc or resources:
        return "iam"
    # Fallback: check filename
    name = filepath.name.lower()
    if any(k in name for k in ["deployment", "pod", "service", "rbac", "role", "k8s"]):
        return "kubernetes"
    if "sam" in name or "serverless" in name:
        return "serverless"
    return "iam"


# ──────────────────────────────────────────────────────────────────────────────
# KUBERNETES EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

def _extract_k8s(doc: dict, features: dict):
    kind = get_resource_kind(doc)

    # Resource type flags
    if kind in ("Pod", "Deployment", "DaemonSet", "StatefulSet", "ReplicaSet", "Job", "CronJob"):
        features["resource_type_k8s_pod"] = 1.0
        _extract_pod_spec(doc, features, kind)
    elif kind in ("Role", "ClusterRole", "RoleBinding", "ClusterRoleBinding"):
        features["resource_type_rbac"] = 1.0
        _extract_rbac(doc, features, kind)

    # Namespace check
    ns = doc.get("metadata", {}).get("namespace", "default") or "default"
    features["namespace_default"] = float(ns == "default")


def _get_pod_spec(doc: dict, kind: str) -> dict:
    """Navigate to the Pod spec regardless of the resource kind."""
    if kind == "Pod":
        return doc.get("spec", {})
    # Deployment, DaemonSet, etc. — spec.template.spec
    return doc.get("spec", {}).get("template", {}).get("spec", {})


def _extract_pod_spec(doc: dict, features: dict, kind: str):
    spec = _get_pod_spec(doc, kind)
    if not isinstance(spec, dict):
        return

    # Pod-level flags
    features["host_network"]      = float(spec.get("hostNetwork", False))
    features["host_pid"]          = float(spec.get("hostPID", False))
    features["host_ipc"]          = float(spec.get("hostIPC", False))
    features["automount_sa_token"]= float(spec.get("automountServiceAccountToken", True))

    sa = spec.get("serviceAccountName", "default") or "default"
    features["service_account_default"] = float(sa.lower() in ("default", ""))

    # Volumes — check for hostPath
    volumes = spec.get("volumes", []) or []
    features["host_path_volume"] = float(
        any(isinstance(v, dict) and "hostPath" in v for v in volumes)
    )

    # Containers
    containers = spec.get("containers", []) or []
    init_containers = spec.get("initContainers", []) or []
    all_containers = containers + init_containers

    features["multi_container"] = float(len(containers) > 1)

    # Initialise as "good" (will flip to 1.0 if any container is bad)
    for container in all_containers:
        if not isinstance(container, dict):
            continue
        _extract_container(container, features)


def _extract_container(c: dict, features: dict):
    sc = c.get("securityContext", {}) or {}

    # runAsNonRoot
    if sc.get("runAsNonRoot") is False or sc.get("runAsNonRoot") is None:
        features["run_as_non_root_false"] = 1.0

    # runAsUser / runAsGroup == 0
    if sc.get("runAsUser") == 0:
        features["run_as_root_uid"] = 1.0
    if sc.get("runAsGroup") == 0:
        features["run_as_root_gid"] = 1.0

    if sc.get("privileged"):
        features["privileged"] = 1.0

    if sc.get("allowPrivilegeEscalation") is not False:  # absent = True by default
        features["allow_privilege_escalation"] = 1.0

    if not sc.get("readOnlyRootFilesystem"):
        features["read_only_root_fs_false"] = 1.0

    # Capabilities
    caps = sc.get("capabilities", {}) or {}
    adds = [str(a).upper() for a in (caps.get("add", []) or [])]
    drops = [str(d).upper() for d in (caps.get("drop", []) or [])]
    if any(a in ("ALL", "SYS_ADMIN", "NET_ADMIN") for a in adds):
        features["capabilities_add_all"] = 1.0
    if "ALL" not in drops:
        features["capabilities_drop_missing"] = 1.0

    # Seccomp (container-level annotation check done at pod level too)
    if "seccompProfile" not in sc:
        features["seccomp_missing"] = 1.0

    # Image tag
    image = c.get("image", "") or ""
    if image.endswith(":latest") or (":" not in image and "@" not in image):
        features["image_latest_tag"] = 1.0
    if "@sha256:" not in image:
        features["image_no_digest"] = 1.0

    # Resources
    res = c.get("resources", {}) or {}
    limits   = res.get("limits", {})
    requests = res.get("requests", {})
    if not limits:
        features["no_resource_limits"] = 1.0
    if not requests:
        features["no_resource_requests"] = 1.0

    # Probes
    if not c.get("livenessProbe"):
        features["no_liveness_probe"] = 1.0
    if not c.get("readinessProbe"):
        features["no_readiness_probe"] = 1.0


def _extract_rbac(doc: dict, features: dict, kind: str):
    if kind in ("ClusterRole", "Role"):
        rules = doc.get("rules", []) or []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            verbs     = [str(v) for v in (rule.get("verbs", []) or [])]
            resources = [str(r) for r in (rule.get("resources", []) or [])]
            api_groups= [str(a) for a in (rule.get("apiGroups", []) or [])]

            if "*" in verbs:
                features["wildcard_verb"] = 1.0
            if "*" in resources:
                features["wildcard_resource"] = 1.0
            if "*" in api_groups:
                features["wildcard_api_group"] = 1.0
            if "secrets" in resources:
                features["secrets_access"] = 1.0
            if "exec" in verbs or "create" in verbs:
                features["exec_access"] = 1.0

        # Check if this is the cluster-admin role
        name = doc.get("metadata", {}).get("name", "")
        if name == "cluster-admin" or doc.get("aggregationRule"):
            features["cluster_admin_role"] = 1.0

    elif kind == "ClusterRoleBinding":
        features["all_namespaces_binding"] = 1.0
        # Check if it binds to cluster-admin
        role_ref = doc.get("roleRef", {}) or {}
        if role_ref.get("name") == "cluster-admin":
            features["cluster_admin_role"] = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# IAM / CLOUDFORMATION EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

def _extract_iam(doc: dict, features: dict):
    resources = doc.get("Resources", {}) or {}

    for r_name, r_def in resources.items():
        if not isinstance(r_def, dict):
            continue
        r_type = r_def.get("Type", "")
        props  = r_def.get("Properties", {}) or {}

        if "IAM::Role" in r_type or "IAM::Policy" in r_type or "IAM::ManagedPolicy" in r_type:
            features["resource_type_iam"] = 1.0
            _extract_iam_policy_doc(props, features)

        if "S3::Bucket" in r_type:
            features["resource_type_s3"] = 1.0
            _extract_s3_resource(props, features)

        if "Lambda::Function" in r_type:
            features["resource_type_lambda"] = 1.0
            _extract_lambda_resource(props, features)


def _extract_iam_policy_doc(props: dict, features: dict):
    # Inline policies
    policy_doc = props.get("PolicyDocument", {}) or {}
    assume_doc  = props.get("AssumeRolePolicyDocument", {}) or {}
    managed     = props.get("ManagedPolicyArns", []) or []

    # Check for AdministratorAccess
    if any("AdministratorAccess" in str(m) for m in managed):
        features["iam_admin_policy"] = 1.0

    for doc_key in [policy_doc, assume_doc]:
        stmts = doc_key.get("Statement", []) or []
        if isinstance(stmts, dict):
            stmts = [stmts]

        for stmt in stmts:
            if not isinstance(stmt, dict):
                continue
            effect    = stmt.get("Effect", "")
            actions   = _to_list(stmt.get("Action", []))
            resources = _to_list(stmt.get("Resource", []))
            principal = stmt.get("Principal", {})
            condition = stmt.get("Condition", {})

            if effect == "Allow":
                if "*" in actions or "iam:*" in actions:
                    features["iam_wildcard_action"] = 1.0
                if "*" in resources:
                    features["iam_wildcard_resource"] = 1.0
                if "*" in actions and "*" in resources:
                    features["iam_allow_star"] = 1.0
                    features["iam_admin_policy"] = 1.0
                if not condition:
                    features["iam_no_condition"] = 1.0
                if "iam:PassRole" in actions or "*" in actions:
                    features["iam_passrole"] = 1.0
                if "sts:AssumeRole" in actions and "*" in resources:
                    features["iam_assume_role_star"] = 1.0

            # Cross-account trust check
            if isinstance(principal, dict):
                aws_principal = _to_list(principal.get("AWS", []))
                for p in aws_principal:
                    if re.match(r"arn:aws:iam::\d{12}:root", str(p)):
                        features["iam_cross_account_trust"] = 1.0
                    if str(p) == "*":
                        features["iam_cross_account_trust"] = 1.0


def _extract_s3_resource(props: dict, features: dict):
    acl = props.get("AccessControl", "")
    if acl == "PublicRead":
        features["s3_public_read_acl"] = 1.0
    if acl == "PublicReadWrite":
        features["s3_public_read_write_acl"] = 1.0

    # Encryption
    enc = props.get("BucketEncryption", {})
    if not enc:
        features["s3_no_encryption"] = 1.0

    # Versioning
    ver = props.get("VersioningConfiguration", {}) or {}
    if ver.get("Status", "") != "Enabled":
        features["s3_versioning_off"] = 1.0

    # Logging
    if not props.get("LoggingConfiguration"):
        features["s3_no_logging"] = 1.0

    # Block public access
    bpa = props.get("PublicAccessBlockConfiguration", {}) or {}
    if not bpa or not all(bpa.get(k, False) for k in
                          ["BlockPublicAcls", "BlockPublicPolicy",
                           "IgnorePublicAcls", "RestrictPublicBuckets"]):
        features["s3_block_public_off"] = 1.0


def _extract_lambda_resource(props: dict, features: dict):
    if not props.get("VpcConfig"):
        features["lambda_no_vpc"] = 1.0

    if not props.get("DeadLetterConfig"):
        features["lambda_no_dlq"] = 1.0

    # Check env vars for secrets
    env = props.get("Environment", {}).get("Variables", {}) or {}
    secret_keys = ["PASSWORD", "SECRET", "KEY", "TOKEN", "CREDENTIAL", "PWD"]
    if any(any(s in k.upper() for s in secret_keys) for k in env):
        features["lambda_env_secrets"] = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# SAM / SERVERLESS EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

def _extract_sam(doc: dict, features: dict):
    resources = doc.get("Resources", {}) or {}
    for r_name, r_def in resources.items():
        if not isinstance(r_def, dict):
            continue
        r_type = r_def.get("Type", "")
        props  = r_def.get("Properties", {}) or {}

        if "Serverless::Function" in r_type or "Lambda::Function" in r_type:
            features["resource_type_lambda"] = 1.0
            _extract_lambda_resource(props, features)

            # SAM-specific: check API events for auth
            events = props.get("Events", {}) or {}
            for ev_name, ev in events.items():
                if not isinstance(ev, dict):
                    continue
                if ev.get("Type") == "Api":
                    ev_props = ev.get("Properties", {}) or {}
                    if not ev_props.get("Auth"):
                        features["sam_no_auth"] = 1.0

        if "S3::Bucket" in r_type:
            features["resource_type_s3"] = 1.0
            _extract_s3_resource(props, features)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _to_list(val) -> list:
    """Ensure a value is always a list (handles str, list, None)."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]

"""
generate_synthetic_dataset.py
─────────────────────────────
Generates a realistic synthetic dataset of cloud config manifests with
ground-truth labels for training.

Produces:
  • ~600 Kubernetes YAML manifests  (Pods, Deployments, RBAC)
  • ~300 IAM/CloudFormation JSON templates
  • ~200 SAM YAML serverless templates
  Total: ~1,100 labeled synthetic manifests

Each file is:
  - Structurally valid YAML/JSON
  - Varied across severity levels (COMPLIANT / MISCONFIGURED)
  - Named with metadata embedded in the filename

Run:
    python scripts/generate_synthetic_dataset.py
"""

import sys
import json
import random
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import RAW_K8S_DIR, RAW_IAM_DIR, RAW_SAM_DIR
from src.utils.logger import get_logger

log = get_logger("synthetic_gen")
random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

APP_NAMES = ["frontend", "backend", "worker", "api", "auth", "cache",
             "metrics", "logging", "gateway", "scheduler", "processor"]
NAMESPACES = ["default", "production", "staging", "kube-system", "monitoring", "dev"]
IMAGES = {
    "safe":    ["nginx:1.25.3", "python:3.11-slim", "node:20-alpine@sha256:abc123"],
    "unsafe":  ["nginx:latest", "ubuntu", "python", "myapp:latest"],
}

def rand_name():
    return f"{random.choice(APP_NAMES)}-{random.randint(1000, 9999)}"

def rand_image(safe: bool):
    pool = IMAGES["safe"] if safe else IMAGES["unsafe"]
    return random.choice(pool)

def rand_namespace(safe: bool):
    if safe:
        return random.choice(["production", "staging", "monitoring"])
    return random.choice(NAMESPACES)

def dump_yaml(obj: dict, dest: Path, label: str, idx: int, domain: str):
    filename = f"{domain}_{label}_{idx:04d}.yaml"
    dest_file = dest / filename
    with open(dest_file, "w", encoding="utf-8") as f:
        yaml.dump(obj, f, default_flow_style=False, allow_unicode=True)
    return dest_file


def dump_json(obj: dict, dest: Path, label: str, idx: int, domain: str):
    filename = f"{domain}_{label}_{idx:04d}.json"
    dest_file = dest / filename
    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return dest_file


# ──────────────────────────────────────────────────────────────────────────────
# KUBERNETES GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def gen_k8s_deployment(misconfigured: bool) -> dict:
    name = rand_name()
    ns   = rand_namespace(not misconfigured)

    # Security context settings
    if misconfigured:
        sc = {
            "privileged": random.choice([True, None]),
            "allowPrivilegeEscalation": True,
            "runAsNonRoot": False,
            "readOnlyRootFilesystem": False,
            "capabilities": {"add": random.choice([["ALL"], ["SYS_ADMIN"], ["NET_ADMIN", "SYS_PTRACE"]])},
        }
        host_net = random.choice([True, False])
        host_pid = random.choice([True, False])
        image    = rand_image(safe=False)
        resources = {}
        sa_name   = "default"
        probes    = {}
    else:
        sc = {
            "runAsNonRoot": True,
            "runAsUser": random.randint(1000, 9999),
            "allowPrivilegeEscalation": False,
            "readOnlyRootFilesystem": True,
            "capabilities": {"drop": ["ALL"]},
            "seccompProfile": {"type": "RuntimeDefault"},
        }
        host_net = False
        host_pid = False
        image    = rand_image(safe=True)
        resources = {
            "limits":   {"cpu": "500m", "memory": "256Mi"},
            "requests": {"cpu": "100m", "memory": "64Mi"},
        }
        sa_name = f"sa-{name}"
        probes = {
            "livenessProbe":  {"httpGet": {"path": "/health", "port": 8080}, "initialDelaySeconds": 10},
            "readinessProbe": {"httpGet": {"path": "/ready",  "port": 8080}, "initialDelaySeconds": 5},
        }

    container = {
        "name":            name,
        "image":           image,
        "securityContext": sc,
        "resources":       resources,
        **probes,
    }

    return {
        "apiVersion": "apps/v1",
        "kind":       "Deployment",
        "metadata": {
            "name":      name,
            "namespace": ns,
            "labels":    {"app": name, "environment": "production" if not misconfigured else "dev"},
        },
        "spec": {
            "replicas": random.randint(1, 5),
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": {"app": name}},
                "spec": {
                    "hostNetwork":                  host_net,
                    "hostPID":                      host_pid,
                    "serviceAccountName":           sa_name,
                    "automountServiceAccountToken": misconfigured,
                    "containers": [container],
                },
            },
        },
    }


def gen_k8s_cluster_role(misconfigured: bool) -> dict:
    name = f"role-{rand_name()}"

    if misconfigured:
        rules = [{"apiGroups": ["*"], "resources": ["*"], "verbs": ["*"]}]
    else:
        rules = [
            {"apiGroups": ["apps"], "resources": ["deployments"], "verbs": ["get", "list", "watch"]},
            {"apiGroups": [""], "resources": ["pods", "services"], "verbs": ["get", "list"]},
        ]

    return {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind":       "ClusterRole",
        "metadata":   {"name": name},
        "rules":      rules,
    }


def gen_k8s_pod(misconfigured: bool) -> dict:
    name  = rand_name()
    ns    = rand_namespace(not misconfigured)
    image = rand_image(safe=not misconfigured)

    sc = {}
    if misconfigured:
        sc = {"privileged": True, "runAsUser": 0, "allowPrivilegeEscalation": True}
    else:
        sc = {"runAsNonRoot": True, "runAsUser": 1001, "allowPrivilegeEscalation": False,
              "readOnlyRootFilesystem": True, "capabilities": {"drop": ["ALL"]}}

    volumes = []
    if misconfigured and random.random() > 0.5:
        volumes = [{"name": "host-vol", "hostPath": {"path": "/", "type": "Directory"}}]

    return {
        "apiVersion": "v1",
        "kind":       "Pod",
        "metadata":   {"name": name, "namespace": ns},
        "spec": {
            "hostNetwork":   misconfigured and random.random() > 0.6,
            "hostPID":       misconfigured and random.random() > 0.7,
            "containers": [{
                "name":            name,
                "image":           image,
                "securityContext": sc,
                "resources": {} if misconfigured else {
                    "limits":   {"cpu": "200m", "memory": "128Mi"},
                    "requests": {"cpu": "50m",  "memory": "32Mi"},
                },
            }],
            "volumes": volumes,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# IAM / CLOUDFORMATION GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def gen_iam_role_cfn(misconfigured: bool) -> dict:
    name = rand_name().replace("-", "")

    if misconfigured:
        policy_doc = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect":   "Allow",
                "Action":   "*",
                "Resource": "*",
            }]
        }
        trust_principal = {"AWS": f"arn:aws:iam::999999999999:root"}
    else:
        policy_doc = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect":    "Allow",
                "Action":    ["s3:GetObject", "s3:PutObject"],
                "Resource":  f"arn:aws:s3:::my-bucket-{name}/*",
                "Condition": {"StringEquals": {"s3:prefix": "uploads/"}},
            }]
        }
        trust_principal = {"Service": "lambda.amazonaws.com"}

    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": f"IAM Role for {name}",
        "Resources": {
            f"Role{name}": {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "RoleName": f"role-{name}",
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [{
                            "Effect":    "Allow",
                            "Principal": trust_principal,
                            "Action":    "sts:AssumeRole",
                        }]
                    },
                    "Policies": [{
                        "PolicyName":     f"policy-{name}",
                        "PolicyDocument": policy_doc,
                    }]
                }
            }
        }
    }


def gen_s3_bucket_cfn(misconfigured: bool) -> dict:
    name = f"bucket-{rand_name()}"

    if misconfigured:
        bucket_props = {
            "BucketName":    name,
            "AccessControl": random.choice(["PublicRead", "PublicReadWrite"]),
        }
    else:
        bucket_props = {
            "BucketName": name,
            "AccessControl": "Private",
            "BucketEncryption": {
                "ServerSideEncryptionConfiguration": [{
                    "ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}
                }]
            },
            "VersioningConfiguration": {"Status": "Enabled"},
            "LoggingConfiguration":    {"DestinationBucketName": f"{name}-logs"},
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls":       True,
                "BlockPublicPolicy":     True,
                "IgnorePublicAcls":      True,
                "RestrictPublicBuckets": True,
            },
        }

    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Resources": {
            name.replace("-", ""): {
                "Type":       "AWS::S3::Bucket",
                "Properties": bucket_props,
            }
        }
    }


# ──────────────────────────────────────────────────────────────────────────────
# SAM / SERVERLESS GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def gen_sam_function(misconfigured: bool) -> dict:
    name = rand_name().replace("-", "")

    if misconfigured:
        env = {"Variables": {"DB_PASSWORD": "admin123", "SECRET_KEY": "hardcoded"}}
        vpc  = None
        dlq  = None
        role  = "arn:aws:iam::*:role/AdministratorAccess"
        events = {
            "ApiEvent": {
                "Type": "Api",
                "Properties": {"Path": "/data", "Method": "GET"},
            }
        }
    else:
        env  = {"Variables": {"LOG_LEVEL": "INFO", "REGION": "us-east-1"}}
        vpc  = {"SecurityGroupIds": ["sg-12345"], "SubnetIds": ["subnet-abc"]}
        dlq  = {"Type": "SQS", "TargetArn": {"Fn::GetAtt": ["DLQ", "Arn"]}}
        role  = {"Fn::GetAtt": [f"Role{name}", "Arn"]}
        events = {
            "ApiEvent": {
                "Type": "Api",
                "Properties": {
                    "Path":   "/data",
                    "Method": "GET",
                    "Auth":   {"Authorizer": "CognitoAuthorizer"},
                },
            }
        }

    function = {
        "Type": "AWS::Serverless::Function",
        "Properties": {
            "FunctionName": f"fn-{name}",
            "Runtime":      "python3.11",
            "Handler":      "index.handler",
            "Environment":  env,
            "Events":       events,
            "Role":         role,
        }
    }
    if vpc:
        function["Properties"]["VpcConfig"] = vpc
    if dlq:
        function["Properties"]["DeadLetterQueue"] = dlq

    return {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Transform": "AWS::Serverless-2016-10-31",
        "Resources": {f"Fn{name}": function},
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def generate_kubernetes(n: int = 600):
    log.info(f"Generating {n} Kubernetes manifests...")
    count = 0
    # 55% misconfigured, 45% compliant  (realistic class imbalance)
    for i in range(n):
        misconfig = (i % 100) < 55
        label = "MISCONFIGURED" if misconfig else "COMPLIANT"

        gen_fn = random.choice([gen_k8s_deployment, gen_k8s_pod, gen_k8s_cluster_role])
        obj    = gen_fn(misconfig)
        dump_yaml(obj, RAW_K8S_DIR, label, i, "k8s")
        count += 1
    log.info(f"  ✓ Generated {count} Kubernetes manifests → {RAW_K8S_DIR}")
    return count


def generate_iam(n: int = 300):
    log.info(f"Generating {n} IAM/CloudFormation templates...")
    count = 0
    for i in range(n):
        misconfig = (i % 100) < 55
        label = "MISCONFIGURED" if misconfig else "COMPLIANT"
        gen_fn = random.choice([gen_iam_role_cfn, gen_s3_bucket_cfn])
        obj    = gen_fn(misconfig)
        dump_json(obj, RAW_IAM_DIR, label, i, "iam")
        count += 1
    log.info(f"  ✓ Generated {count} IAM templates → {RAW_IAM_DIR}")
    return count


def generate_serverless(n: int = 200):
    log.info(f"Generating {n} SAM templates...")
    count = 0
    for i in range(n):
        misconfig = (i % 100) < 55
        label = "MISCONFIGURED" if misconfig else "COMPLIANT"
        obj   = gen_sam_function(misconfig)
        dump_yaml(obj, RAW_SAM_DIR, label, i, "sam")
        count += 1
    log.info(f"  ✓ Generated {count} SAM templates → {RAW_SAM_DIR}")
    return count


if __name__ == "__main__":
    k8s_count = generate_kubernetes(600)
    iam_count = generate_iam(300)
    sam_count = generate_serverless(200)

    total = k8s_count + iam_count + sam_count
    print(f"\n{'='*50}")
    print(f"SYNTHETIC DATASET GENERATED")
    print(f"{'='*50}")
    print(f"  Kubernetes  : {k8s_count:>5} manifests")
    print(f"  IAM/CFN     : {iam_count:>5} templates")
    print(f"  Serverless  : {sam_count:>5} templates")
    print(f"  {'TOTAL':>10} : {total:>5} files")
    print(f"{'='*50}")

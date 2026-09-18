"""
download_dataset.py
───────────────────
Phase 1 — Data Collection

Downloads real-world cloud configuration files from GitHub and known public
sources, then saves them to data/raw/{domain}/.

Usage:
    python scripts/download_dataset.py --domain kubernetes --max 500
    python scripts/download_dataset.py --domain all --max 300

Sources:
  • Kubernetes : GitHub API code search (topic:kubernetes, topic:helm)
  • IAM        : aws-cloudformation-templates + PMapper templates
  • Serverless  : AWS Serverless Application Repo + synthetic variants
"""

import os
import sys
import json
import time
import argparse
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm

# ── Make sure the project root is on sys.path ──────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import GITHUB_TOKEN, RAW_K8S_DIR, RAW_IAM_DIR, RAW_SAM_DIR
from src.utils.logger import get_logger

log = get_logger("download")

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN and GITHUB_TOKEN != "your_github_token_here":
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    log.info("GitHub token loaded — authenticated requests (5000 req/hr).")
else:
    log.warning("No GitHub token — using unauthenticated requests (60 req/hr). "
                "Add GITHUB_TOKEN to .env for better rate limits.")

# ──────────────────────────────────────────────────────────────────────────────
# KUBERNETES SOURCES
# ──────────────────────────────────────────────────────────────────────────────

K8S_REPOS = [
    # Official test suites with known-good and known-bad manifests
    ("stackrox/kube-linter",         "test/testdata"),
    ("bridgecrewio/checkov",          "tests/resources"),
    ("GoogleCloudPlatform/microservices-demo", "kubernetes-manifests"),
    ("kubernetes/examples",           ""),
    ("argoproj/argo-cd",              "test/testdata/app"),
    ("cert-manager/cert-manager",     "deploy/charts/cert-manager/templates"),
    ("grafana/grafana",               "packaging/docker"),
    ("prometheus/prometheus",         "documentation/examples"),
    ("elastic/cloud-on-k8s",         "config/samples"),
    ("bitnami/charts",                "bitnami/nginx/templates"),
]

IAM_REPOS = [
    ("awslabs/aws-cloudformation-templates", ""),
    ("trufflesecurity/trufflehog",           ""),
    ("nccgroup/PMapper",                     "test"),
]

SAM_REPOS = [
    ("aws-samples/serverless-patterns",  ""),
    ("awslabs/serverless-application-model", "examples"),
]


def gh_raw_url(repo: str, path: str, branch: str = "main") -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"


def search_github_code(query: str, max_files: int = 100) -> list[dict]:
    """Use GitHub Code Search API to find YAML files matching a query."""
    results = []
    per_page = min(100, max_files)
    for page in range(1, (max_files // per_page) + 2):
        url = "https://api.github.com/search/code"
        params = {"q": query, "per_page": per_page, "page": page}
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 403:
                log.warning("GitHub rate limit hit — sleeping 60s")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                log.debug(f"Search returned {resp.status_code} for query: {query}")
                break
            items = resp.json().get("items", [])
            if not items:
                break
            results.extend(items)
            if len(results) >= max_files:
                break
            time.sleep(1.5)   # Respect secondary rate limits
        except Exception as e:
            log.warning(f"Search error: {e}")
            break
    return results[:max_files]


def download_file(url: str, dest_dir: Path, filename: str = None) -> bool:
    """Download a raw file from a URL and save it to dest_dir."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return False
        content = resp.text
        if len(content.strip()) < 20:   # Skip empty/tiny files
            return False

        # Use SHA1 of URL as unique filename if none provided
        if not filename:
            filename = hashlib.sha1(url.encode()).hexdigest()[:12] + ".yaml"

        # Sanitise filename
        filename = filename.replace("/", "_").replace("\\", "_")
        dest = dest_dir / filename
        dest.write_text(content, encoding="utf-8", errors="ignore")
        return True
    except Exception as e:
        log.debug(f"Download failed for {url}: {e}")
        return False


def download_repo_tree(repo: str, subpath: str, dest_dir: Path,
                       max_files: int = 100, extension: str = ".yaml") -> int:
    """
    Download files from a specific GitHub repo subdirectory using the Trees API.
    Returns the count of successfully downloaded files.
    """
    for branch in ["main", "master"]:
        url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                break
        except Exception:
            continue
    else:
        log.debug(f"Could not reach repo tree: {repo}")
        return 0

    tree = resp.json().get("tree", [])
    files = [
        item for item in tree
        if item["type"] == "blob"
        and item["path"].endswith(extension)
        and (subpath == "" or item["path"].startswith(subpath))
        and item.get("size", 0) < 100_000
    ][:max_files]

    count = 0
    for item in files:
        raw = gh_raw_url(repo, item["path"], branch)
        fname = item["path"].replace("/", "__")
        if download_file(raw, dest_dir, fname):
            count += 1
        time.sleep(0.3)
    return count


def download_kubernetes(max_files: int = 500):
    log.info(f"=== Downloading Kubernetes manifests (target: {max_files}) ===")
    total = 0

    # 1. From curated repos
    per_repo = max(20, max_files // len(K8S_REPOS))
    for repo, subpath in tqdm(K8S_REPOS, desc="Repo crawl"):
        got = download_repo_tree(repo, subpath, RAW_K8S_DIR,
                                  max_files=per_repo, extension=".yaml")
        total += got
        log.info(f"  {repo}: downloaded {got} files (total so far: {total})")
        if total >= max_files:
            break

    # 2. GitHub Code Search for more variety
    if total < max_files:
        remaining = max_files - total
        log.info(f"Code-searching GitHub for {remaining} more K8s manifests...")
        queries = [
            "kind:Deployment apiVersion:apps/v1 extension:yaml",
            "kind:Pod securityContext extension:yaml NOT test",
            "kind:ClusterRole rules extension:yaml",
            "kind:ServiceAccount automountServiceAccountToken extension:yaml",
        ]
        for q in queries:
            items = search_github_code(q, max_files=remaining // len(queries) + 10)
            for item in tqdm(items, desc=f"Search: {q[:30]}"):
                raw_url = (item.get("html_url", "")
                           .replace("github.com", "raw.githubusercontent.com")
                           .replace("/blob/", "/"))
                if download_file(raw_url, RAW_K8S_DIR):
                    total += 1
                if total >= max_files:
                    break
            if total >= max_files:
                break

    log.info(f"Kubernetes: downloaded {total} files → {RAW_K8S_DIR}")
    return total


def download_iam(max_files: int = 300):
    log.info(f"=== Downloading IAM/CloudFormation templates (target: {max_files}) ===")
    total = 0
    per_repo = max(30, max_files // len(IAM_REPOS))

    for repo, subpath in tqdm(IAM_REPOS, desc="IAM repo crawl"):
        for ext in [".yaml", ".yml", ".json"]:
            got = download_repo_tree(repo, subpath, RAW_IAM_DIR,
                                      max_files=per_repo, extension=ext)
            total += got
            if total >= max_files:
                break
        if total >= max_files:
            break

    log.info(f"IAM: downloaded {total} files → {RAW_IAM_DIR}")
    return total


def download_serverless(max_files: int = 200):
    log.info(f"=== Downloading SAM/Serverless templates (target: {max_files}) ===")
    total = 0
    per_repo = max(30, max_files // len(SAM_REPOS))

    for repo, subpath in tqdm(SAM_REPOS, desc="SAM repo crawl"):
        for ext in [".yaml", ".yml"]:
            got = download_repo_tree(repo, subpath, RAW_SAM_DIR,
                                      max_files=per_repo, extension=ext)
            total += got
            if total >= max_files:
                break
        if total >= max_files:
            break

    log.info(f"Serverless: downloaded {total} files → {RAW_SAM_DIR}")
    return total


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download cloud config datasets")
    parser.add_argument("--domain", choices=["kubernetes", "iam", "serverless", "all"],
                        default="all", help="Which domain to download")
    parser.add_argument("--max", type=int, default=300,
                        help="Maximum files to download per domain")
    args = parser.parse_args()

    summary = {}
    if args.domain in ("kubernetes", "all"):
        summary["kubernetes"] = download_kubernetes(args.max)
    if args.domain in ("iam", "all"):
        summary["iam"] = download_iam(args.max)
    if args.domain in ("serverless", "all"):
        summary["serverless"] = download_serverless(args.max)

    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    for domain, count in summary.items():
        print(f"  {domain:>12}: {count:>4} files")
    print(f"  {'TOTAL':>12}: {sum(summary.values()):>4} files")
    print("="*50)

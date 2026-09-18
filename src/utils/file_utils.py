"""
file_utils.py — File discovery, safe YAML/JSON loading, and manifest helpers.
"""
import yaml
import json
import os
from pathlib import Path
from typing import Generator, Optional
from src.utils.logger import get_logger

log = get_logger("file_utils")


def discover_manifests(directory: str | Path,
                       extensions: tuple = (".yaml", ".yml", ".json"),
                       recursive: bool = True) -> list[Path]:
    """
    Walk a directory and return all files matching the given extensions.
    Filters out files larger than 500KB (likely not individual manifests).
    """
    directory = Path(directory)
    results = []
    pattern = "**/*" if recursive else "*"
    for ext in extensions:
        for f in directory.glob(f"{pattern}{ext}"):
            if f.is_file() and f.stat().st_size < 500_000:
                results.append(f)
    log.info(f"Discovered {len(results)} manifests in {directory}")
    return sorted(results)


def safe_load_yaml(filepath: str | Path) -> Optional[dict | list]:
    """
    Load a YAML file safely. Returns None on parse errors.
    Handles multi-document YAML files (returns first valid document).
    """
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            docs = list(yaml.safe_load_all(f))
        # Filter out None/empty documents
        docs = [d for d in docs if d is not None]
        if not docs:
            return None
        return docs[0] if len(docs) == 1 else docs
    except yaml.YAMLError as e:
        log.debug(f"YAML parse error in {filepath}: {e}")
        return None
    except Exception as e:
        log.debug(f"File read error {filepath}: {e}")
        return None


def safe_load_json(filepath: str | Path) -> Optional[dict]:
    """Load a JSON file safely. Returns None on errors."""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        log.debug(f"JSON parse error in {filepath}: {e}")
        return None


def extract_inline_comments(filepath: str | Path) -> list[str]:
    """
    Extract all inline comments from a YAML file.
    Returns list of comment strings (without the # prefix).
    """
    comments = []
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#"):
                    comments.append(stripped[1:].strip())
                elif "#" in stripped:
                    # Inline comment after a value
                    comment_part = stripped.split("#", 1)[1].strip()
                    if comment_part:
                        comments.append(comment_part)
    except Exception:
        pass
    return comments


def get_resource_kind(doc: dict) -> str:
    """Extract the Kubernetes resource kind from a parsed manifest."""
    if not isinstance(doc, dict):
        return "Unknown"
    return doc.get("kind", "Unknown")


def get_resource_name(doc: dict) -> str:
    """Extract the resource name from metadata."""
    if not isinstance(doc, dict):
        return "unnamed"
    return doc.get("metadata", {}).get("name", "unnamed")


def get_resource_namespace(doc: dict) -> str:
    """Extract the namespace, defaulting to 'default'."""
    if not isinstance(doc, dict):
        return "default"
    return doc.get("metadata", {}).get("namespace", "default")


def get_labels(doc: dict) -> dict:
    """Extract metadata.labels as a dict."""
    if not isinstance(doc, dict):
        return {}
    return doc.get("metadata", {}).get("labels", {}) or {}


def get_annotations(doc: dict) -> dict:
    """Extract metadata.annotations as a dict."""
    if not isinstance(doc, dict):
        return {}
    return doc.get("metadata", {}).get("annotations", {}) or {}


def max_nesting_depth(obj, depth: int = 0) -> int:
    """Recursively compute maximum nesting depth of a dict/list structure."""
    if isinstance(obj, dict):
        return max((max_nesting_depth(v, depth + 1) for v in obj.values()),
                   default=depth)
    elif isinstance(obj, list):
        return max((max_nesting_depth(i, depth) for i in obj),
                   default=depth)
    return depth


def count_keys(obj) -> int:
    """Count all keys in a nested dict/list structure."""
    if isinstance(obj, dict):
        return len(obj) + sum(count_keys(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_keys(i) for i in obj)
    return 0

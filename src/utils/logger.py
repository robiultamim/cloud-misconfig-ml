"""
logger.py — Structured, coloured console + file logging for all pipeline stages.
"""
import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parents[2] / "pipeline.log"

def get_logger(name: str = "cloud-misconfig") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # Already configured

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s :: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

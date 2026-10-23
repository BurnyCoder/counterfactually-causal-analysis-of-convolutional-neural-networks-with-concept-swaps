"""Timestamped logging to the terminal and to `logs/<timestamp>.log` at the same time.

Local: one `setup_logging()` call attaches a console handler and a file handler to the package logger.
Global context: every phase logs through `get_logger(__name__)`, so a full run leaves one complete,
timestamped log file next to its results (no LLMs are used in this project, so there are no prompts to log).
"""

import logging  # stdlib logging: https://docs.python.org/3/library/logging.html
import sys  # stdout stream for the console handler
from datetime import datetime  # timestamp for the log file name
from pathlib import Path

FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"  # time | level | module | message


def setup_logging(log_dir: Path, level: int = logging.INFO) -> Path:
    """Attach console + file handlers to the `concept_swap` logger and return the log-file path.

    Local: idempotent (old handlers are removed first) so repeated calls in tests do not duplicate lines.
    Global context: called once by `main.py` before any phase runs.
    """
    log_dir.mkdir(parents=True, exist_ok=True)  # create logs/ on first run
    path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}.log"  # e.g. logs/20261023_161500.log
    root = logging.getLogger("concept_swap")  # package-level logger; children inherit its handlers
    root.setLevel(level)
    for h in list(root.handlers):  # remove handlers from a previous setup call
        root.removeHandler(h)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(logging.Formatter(FMT))  # same format on screen and in the file
        root.addHandler(h)
    return path  # callers log only path.name, so logs carry no absolute (user-identifying) paths


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger (e.g. `concept_swap.phases.e1_replication`)."""
    return logging.getLogger(name if name.startswith("concept_swap") else f"concept_swap.{name}")

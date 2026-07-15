"""Shared library for the vision-LSI pilot.

Quieten the noisy channels that flood stage logs, mirroring the parent MMEG
repo: huggingface_hub logs every Hub request via httpx at INFO, and transformers
emits weight-loading chatter. Real problems still surface at WARNING+.
"""
from __future__ import annotations

import logging
import os
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", module=r"transformers\..*")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

"""Config loading for the vision-LSI pilot.

Plain YAML (not Hydra) is used here so the sub-repo stays self-contained and the
``python -m src.<stage> --config ...`` command-line interface from the plan works
without composition machinery. ``${VAR}`` tokens are expanded from the process
environment at load time; a missing variable expands to an empty string so that
downstream code can decide whether the resulting path is usable.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    """Recursively expand ``${VAR}`` env references in strings."""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class Config(dict):
    """A dict with attribute access and nested-key convenience.

    ``cfg.data.min_occurrences`` and ``cfg["data"]["min_occurrences"]`` are
    equivalent. Nested dicts are wrapped lazily.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value


def load_config(path: str | Path) -> Config:
    """Load a YAML config, expand env vars, and return a Config."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(_expand(raw))


@dataclass
class ImagenetStatus:
    """Result of probing the configured ImageNet root."""

    root: str
    train_dir: Path | None
    available: bool
    n_classes: int
    reason: str = ""


def probe_imagenet(root: str | os.PathLike | None) -> ImagenetStatus:
    """Check whether an ImageNet train tree with WNID class dirs is present.

    Falls back from ``<root>/train`` to ``<root>`` itself so a directory that
    already contains ``n########`` class folders works without a ``train/``
    level. Returns a status object rather than raising, so the caller can decide
    whether the image side is runnable.
    """
    if not root:
        return ImagenetStatus("", None, False, 0, "IMAGENET_ROOT is not set")

    root_path = Path(root)
    if not root_path.exists():
        return ImagenetStatus(str(root), None, False, 0, f"{root_path} does not exist")

    candidates = [root_path / "train", root_path]
    wnid_re = re.compile(r"^n\d{8}$")
    for cand in candidates:
        if not cand.is_dir():
            continue
        wnids = [p for p in cand.iterdir() if p.is_dir() and wnid_re.match(p.name)]
        if wnids:
            return ImagenetStatus(str(root), cand, True, len(wnids))
    return ImagenetStatus(
        str(root), None, False, 0,
        f"no n######## class directories under {root_path} or {root_path}/train",
    )

"""Record the box state and probe the ImageNet tree.

Adapted from the plan's shell audit into a single stage that also writes a JSON
summary the rest of the pipeline can read (e.g. whether the image side is
runnable). Emits a human-readable box_audit.txt alongside the JSON.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from src.pilotlib.config import probe_imagenet


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - best-effort probe
        return f"<unavailable: {exc!r}>"


def _torch_info() -> dict:
    info: dict = {"available": False}
    try:
        import torch

        info.update(
            available=True,
            version=torch.__version__,
            cuda_available=bool(torch.cuda.is_available()),
            cuda_version=torch.version.cuda,
        )
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["gpu0"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        info["error"] = repr(exc)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit the box and probe ImageNet.")
    ap.add_argument("--imagenet-root", default="", help="Path to the ImageNet root")
    ap.add_argument("--output", default="box_audit.json")
    args = ap.parse_args()

    nvidia = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"]
    )
    total, _, free = shutil.disk_usage(".")
    audit = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "nvidia_smi": nvidia,
        "disk_free_gb": round(free / 1e9, 1),
        "torch": _torch_info(),
        "imagenet": asdict(probe_imagenet(args.imagenet_root or None)),
    }
    # asdict leaves the Path as a PosixPath; make it JSON-serialisable.
    if audit["imagenet"]["train_dir"] is not None:
        audit["imagenet"]["train_dir"] = str(audit["imagenet"]["train_dir"])

    Path(args.output).write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        f"host:      {audit['hostname']}",
        f"platform:  {audit['platform']}",
        f"python:    {audit['python']}",
        f"gpu:       {nvidia.splitlines()[0] if nvidia else '<none>'}",
        f"disk free: {audit['disk_free_gb']} GB",
        f"torch:     {audit['torch']}",
        f"imagenet:  available={audit['imagenet']['available']} "
        f"classes={audit['imagenet']['n_classes']} "
        f"{audit['imagenet'].get('reason', '')}".rstrip(),
    ]
    Path("box_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not audit["imagenet"]["available"]:
        print(
            "\n[warn] ImageNet is not available: image-dependent systems "
            "(bert+image, image-profile-only) will be skipped. The text-only "
            "baselines and the bert+label control still run.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

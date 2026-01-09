# Phase-4/loaders/bundle_loader.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional


class BundleNotFoundError(FileNotFoundError):
    pass


def find_latest_bundle(outputs_dir: Path, pattern: str = "phase3_bundle_*.json") -> Path:
    outputs_dir = outputs_dir.resolve()
    if not outputs_dir.exists():
        raise BundleNotFoundError(f"Outputs directory not found: {outputs_dir}")

    candidates = sorted(outputs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise BundleNotFoundError(f"No bundle files matching {pattern} in {outputs_dir}")

    return candidates[0]


def load_bundle(bundle_path: Path) -> Dict[str, Any]:
    with bundle_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_latest_bundle(
    outputs_dir: str = "../Phase-3/outputs",
    pattern: str = "phase3_bundle_*.json"
) -> Tuple[Dict[str, Any], str]:
    outputs_path = Path(outputs_dir)
    latest_path = find_latest_bundle(outputs_path, pattern=pattern)
    bundle = load_bundle(latest_path)
    return bundle, str(latest_path)

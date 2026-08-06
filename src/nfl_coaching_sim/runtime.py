"""Resolve immutable assets in source checkouts and frozen application bundles."""

from __future__ import annotations

import sys
from pathlib import Path


def application_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is not None:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[2]


def bundled_asset(*parts: str) -> Path:
    return application_root().joinpath(*parts)

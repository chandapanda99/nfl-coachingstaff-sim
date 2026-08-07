"""Resolve immutable assets in source checkouts and frozen application bundles."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APPLICATION_DIRECTORY_NAME = "nfl-coachingstaff-sim"
LEGACY_APPLICATION_DIRECTORY_NAMES = (
    "com.aayushchanda.nfl-coachingstaff-sim",
    "NFL Virtual Coaching Staff",
)


def application_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root is not None:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[2]


def bundled_asset(*parts: str) -> Path:
    return application_root().joinpath(*parts)


def user_data_root() -> Path:
    """Return the writable, per-user application directory for this platform."""

    if sys.platform == "win32" and (local_app_data := os.environ.get("LOCALAPPDATA")):
        return Path(local_app_data) / APPLICATION_DIRECTORY_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APPLICATION_DIRECTORY_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APPLICATION_DIRECTORY_NAME


def user_config_path() -> Path:
    return user_data_root() / "config" / ".env"


def user_scenarios_path() -> Path:
    return user_data_root() / "data" / "custom-scenarios.jsonl"


def user_log_path() -> Path:
    return user_data_root() / "logs" / "sidecar.log"


def _available_destination(destination: Path, legacy_name: str) -> Path:
    if not destination.exists():
        return destination
    candidate = destination.with_name(f"{destination.stem}.legacy-{legacy_name}{destination.suffix}")
    index = 2
    while candidate.exists():
        candidate = destination.with_name(f"{destination.stem}.legacy-{legacy_name}-{index}{destination.suffix}")
        index += 1
    return candidate


def migrate_legacy_user_data() -> None:
    """Move legacy Windows data folders into the canonical application directory."""

    if sys.platform != "win32" or not os.environ.get("LOCALAPPDATA"):
        return

    canonical_root = user_data_root()
    local_root = canonical_root.parent

    # Preserve the complete packaged-app directory, including WebView state, when
    # upgrading from the former reverse-domain directory name.
    identifier_root = local_root / "com.aayushchanda.nfl-coachingstaff-sim"
    if identifier_root.is_dir() and not canonical_root.exists():
        shutil.move(str(identifier_root), str(canonical_root))

    known_files = {
        identifier_root / "config" / ".env": user_config_path(),
        identifier_root / "data" / "custom-scenarios.jsonl": user_scenarios_path(),
        identifier_root / "logs" / "sidecar.log": user_log_path(),
        local_root / "NFL Virtual Coaching Staff" / "sidecar.log": user_log_path(),
    }
    for source, destination in known_files.items():
        if not source.is_file():
            continue
        resolved_destination = _available_destination(destination, source.parent.name.replace(" ", "-"))
        resolved_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(resolved_destination))

    for legacy_name in LEGACY_APPLICATION_DIRECTORY_NAMES:
        legacy_root = local_root / legacy_name
        if not legacy_root.is_dir():
            continue
        if any(legacy_root.iterdir()):
            archive = _available_destination(canonical_root / "legacy" / legacy_name, legacy_name.replace(" ", "-"))
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(legacy_root), str(archive))
        else:
            legacy_root.rmdir()

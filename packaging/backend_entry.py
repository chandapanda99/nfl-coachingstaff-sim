"""Minimal PyInstaller entry point for the Tauri Python sidecar."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nfl_coaching_sim.runtime import migrate_legacy_user_data, user_log_path
from nfl_coaching_sim.server import run_server


def _sidecar_log_path() -> Path:
    configured = os.getenv("NFL_COACH_LOG_FILE")
    if configured:
        return Path(configured).expanduser()
    return user_log_path()


def _configure_windowless_streams() -> None:
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_path = _sidecar_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def main() -> None:
    migrate_legacy_user_data()
    _configure_windowless_streams()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--simulator-path", type=Path)
    parser.add_argument("--custom-scenarios", type=Path)
    args = parser.parse_args()
    run_server(args.scenarios, args.simulator_path, args.host, args.port, args.custom_scenarios)


if __name__ == "__main__":
    main()

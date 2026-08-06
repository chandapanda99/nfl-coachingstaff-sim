"""Build and name the Python sidecar for the current Tauri target."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARIES = ROOT / "frontend" / "src-tauri" / "binaries"


def target_triple() -> str:
    machine = platform.machine().lower()
    architecture = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
    if sys.platform == "win32":
        return f"{architecture}-pc-windows-msvc"
    if sys.platform == "darwin":
        return f"{architecture}-apple-darwin"
    return f"{architecture}-unknown-linux-gnu"


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", str(ROOT / "packaging" / "nfl-coach-backend.spec")],
        cwd=ROOT,
        check=True,
    )
    suffix = ".exe" if sys.platform == "win32" else ""
    source = ROOT / "dist" / f"nfl-coach-backend{suffix}"
    BINARIES.mkdir(parents=True, exist_ok=True)
    destination = BINARIES / f"nfl-coach-backend-{target_triple()}{suffix}"
    shutil.copy2(source, destination)
    print(destination)


if __name__ == "__main__":
    main()

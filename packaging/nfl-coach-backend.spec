from pathlib import Path

project_root = Path(SPEC).parent.parent

datas = [
    (str(project_root / "data" / "scenarios"), "data/scenarios"),
    (str(project_root / "artifacts" / "simulator-v1.joblib"), "artifacts"),
]
hiddenimports = [
    "nfl_coaching_sim.providers.azure_foundry",
    "nfl_coaching_sim.providers.ollama",
]

analysis = Analysis(
    [str(project_root / "packaging" / "backend_entry.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["nflreadpy", "pandas", "polars", "pytest"],
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="nfl-coach-backend",
    console=False,
)

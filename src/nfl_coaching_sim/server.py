"""Minimal local server runtime shared by the CLI and packaged desktop sidecar."""

from __future__ import annotations

from pathlib import Path

from nfl_coaching_sim.runtime import bundled_asset
from nfl_coaching_sim.settings import get_application_settings


def load_startup_assets(scenarios_path: Path | None = None, simulator_path: Path | None = None):
    from nfl_coaching_sim.data import demo_scenarios, read_jsonl
    from nfl_coaching_sim.simulator import DeterministicSimulator

    default_scenarios = bundled_asset("data", "scenarios", "benchmark-v1.jsonl")
    default_simulator = bundled_asset("artifacts", "simulator-v1.joblib")
    if scenarios_path is not None:
        scenarios = read_jsonl(scenarios_path)
    elif default_scenarios.exists():
        scenarios = read_jsonl(default_scenarios)
    else:
        scenarios = demo_scenarios()

    resolved_simulator = simulator_path or (default_simulator if default_simulator.exists() else None)
    evaluator = DeterministicSimulator.deferred(resolved_simulator) if resolved_simulator else DeterministicSimulator()
    return scenarios, evaluator


def run_server(
    scenarios_path: Path | None = None,
    simulator_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    custom_scenarios_path: Path | None = None,
) -> None:
    import uvicorn

    from nfl_coaching_sim.api import create_api
    from nfl_coaching_sim.scenario_library import default_custom_scenarios_path

    scenarios, evaluator = load_startup_assets(scenarios_path, simulator_path)
    api = create_api(
        scenarios,
        evaluator,
        custom_scenarios_path=custom_scenarios_path or default_custom_scenarios_path(),
        frontend_dist=bundled_asset("frontend", "dist"),
    )
    uvicorn.run(api, host=host, port=port, log_level=get_application_settings().log_level.lower())

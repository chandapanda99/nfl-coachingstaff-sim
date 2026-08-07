"""Persistent user-created situation library."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock

from nfl_coaching_sim.models import Scenario
from nfl_coaching_sim.runtime import user_scenarios_path

CUSTOM_SCENARIO_SOURCE = "user-created custom scenario"
_library_lock = Lock()


def default_custom_scenarios_path() -> Path:
    """Return the per-user JSONL call-sheet path, with an environment override."""

    override = os.environ.get("NFL_COACH_CUSTOM_SCENARIOS")
    if override:
        return Path(override).expanduser()
    return user_scenarios_path()


def is_custom_scenario(scenario: Scenario) -> bool:
    return scenario.source == CUSTOM_SCENARIO_SOURCE or scenario.scenario_id.startswith("custom-")


def load_custom_scenarios(path: Path) -> list[Scenario]:
    """Load valid saved situations while preserving their call-sheet order."""

    if not path.exists():
        return []
    scenarios: list[Scenario] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            scenario = Scenario.model_validate_json(line)
        except Exception as error:
            raise ValueError(f"invalid saved custom situation on line {line_number}: {error}") from error
        if not is_custom_scenario(scenario):
            raise ValueError(f"line {line_number} is not a custom situation")
        scenarios.append(scenario)
    return scenarios


def _write_custom_scenarios(path: Path, scenarios: list[Scenario]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n" for item in scenarios),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_custom_scenario(path: Path, scenario: Scenario, replacing_scenario_id: str | None = None) -> list[Scenario]:
    """Create or edit a custom situation and atomically rewrite the JSONL library."""

    if not is_custom_scenario(scenario):
        raise ValueError("only user-created situations can be saved in My Situations")
    with _library_lock:
        replaced_ids = {scenario.scenario_id}
        if replacing_scenario_id:
            replaced_ids.add(replacing_scenario_id)
        saved = [item for item in load_custom_scenarios(path) if item.scenario_id not in replaced_ids]
        saved.append(scenario)
        _write_custom_scenarios(path, saved)
    return saved


def delete_custom_scenario(path: Path, scenario_id: str) -> list[Scenario]:
    """Delete one saved custom situation and return the remaining library."""

    with _library_lock:
        saved = load_custom_scenarios(path)
        if not any(item.scenario_id == scenario_id for item in saved):
            raise ValueError("the selected custom situation is no longer in My Situations")
        remaining = [item for item in saved if item.scenario_id != scenario_id]
        _write_custom_scenarios(path, remaining)
    return remaining

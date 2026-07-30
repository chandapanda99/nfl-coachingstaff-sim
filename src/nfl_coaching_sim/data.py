"""nflverse ingestion and reproducible scenario generation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nfl_coaching_sim.models import Action, GameState, Scenario

TRAINING_SEASONS = tuple(range(2016, 2024))
EVALUATION_SEASONS = (2024, 2025)
QUICKSTART_SCENARIO_COUNT = 40

PBP_COLUMNS = (
    "game_id",
    "play_id",
    "season",
    "week",
    "qtr",
    "game_seconds_remaining",
    "down",
    "ydstogo",
    "yardline_100",
    "posteam",
    "defteam",
    "posteam_score",
    "defteam_score",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "wp",
    "ep",
    "epa",
    "wpa",
    "play_type",
    "no_play",
    "qb_kneel",
    "qb_spike",
)


def load_pbp(seasons: Sequence[int]) -> Any:
    """Download/cache nflverse play-by-play with the supported Python loader."""

    import nflreadpy as nfl

    frame = nfl.load_pbp(list(seasons))
    available = [name for name in PBP_COLUMNS if name in frame.columns]
    return frame.select(available)


def rows_from_frame(frame: Any) -> list[dict[str, Any]]:
    if hasattr(frame, "to_dicts"):
        return frame.to_dicts()
    if hasattr(frame, "to_dict"):
        converted = frame.to_dict(orient="records")
        return list(converted)
    return [dict(row) for row in frame]


def is_late_game_candidate(row: Mapping[str, Any]) -> bool:
    required = (
        "game_id",
        "play_id",
        "season",
        "week",
        "qtr",
        "game_seconds_remaining",
        "down",
        "ydstogo",
        "yardline_100",
        "posteam",
        "defteam",
        "posteam_score",
        "defteam_score",
        "wp",
        "ep",
    )
    if any(row.get(key) is None for key in required):
        return False
    if int(row["qtr"]) not in (3, 4):
        return False
    if int(row["down"]) not in (1, 2, 3, 4):
        return False
    if not (0.10 <= float(row["wp"]) <= 0.90):
        return False
    if abs(int(row["posteam_score"]) - int(row["defteam_score"])) > 16:
        return False
    if row.get("no_play") in (1, True) or row.get("qb_kneel") in (1, True):
        return False
    if row.get("qb_spike") in (1, True):
        return False
    return float(row["ydstogo"]) > 0


def observed_action(row: Mapping[str, Any]) -> tuple[Action, Action | None] | None:
    play_type = str(row.get("play_type") or "").lower()
    down = int(row.get("down") or 0)
    if down == 4 and play_type in ("run", "pass"):
        subtype = Action.RUN if play_type == "run" else Action.PASS
        return Action.GO_FOR_IT, subtype
    mapping = {
        "run": Action.RUN,
        "pass": Action.PASS,
        "punt": Action.PUNT,
        "field_goal": Action.FIELD_GOAL,
    }
    action = mapping.get(play_type)
    return (action, None) if action is not None else None


def _bucket(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    differential = int(row["posteam_score"]) - int(row["defteam_score"])
    return (
        int(row["down"]),
        min(int(float(row["ydstogo"]) // 3), 5),
        min(int(float(row["yardline_100"]) // 20), 4),
        min(int(float(row["game_seconds_remaining"]) // 300), 5),
        -1 if differential < 0 else (1 if differential > 0 else 0),
    )


class ExpectedPointsBaseline:
    """Auditable bucketed EPA policy with hierarchical fallbacks."""

    def __init__(self) -> None:
        self.by_bucket: dict[tuple[int, int, int, int, int], dict[Action, float]] = {}
        self.by_down: dict[int, dict[Action, float]] = {}
        self.global_values: dict[Action, float] = {}

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> ExpectedPointsBaseline:
        bucket_values: dict[tuple[int, int, int, int, int], dict[Action, list[float]]] = defaultdict(lambda: defaultdict(list))
        down_values: dict[int, dict[Action, list[float]]] = defaultdict(lambda: defaultdict(list))
        global_values: dict[Action, list[float]] = defaultdict(list)
        for row in rows:
            if not is_late_game_candidate(row):
                continue
            result = observed_action(row)
            if result is None or row.get("epa") is None:
                continue
            action, _ = result
            value = float(row["epa"])
            bucket_values[_bucket(row)][action].append(value)
            down_values[int(row["down"])][action].append(value)
            global_values[action].append(value)
        self.by_bucket = {
            key: {action: sum(values) / len(values) for action, values in actions.items()} for key, actions in bucket_values.items()
        }
        self.by_down = {
            down: {action: sum(values) / len(values) for action, values in actions.items()} for down, actions in down_values.items()
        }
        self.global_values = {action: sum(values) / len(values) for action, values in global_values.items()}
        return self

    def values_for(self, row: Mapping[str, Any], state: GameState) -> dict[Action, float]:
        exact = self.by_bucket.get(_bucket(row), {})
        down = self.by_down.get(state.down, {})
        defaults = {
            Action.RUN: -0.02,
            Action.PASS: 0.03,
            Action.PUNT: -0.15,
            Action.FIELD_GOAL: 0.05,
            Action.GO_FOR_IT: 0.02,
        }
        return {
            action: exact.get(action, down.get(action, self.global_values.get(action, defaults[action]))) for action in state.legal_actions
        }


def state_from_row(row: Mapping[str, Any]) -> GameState:
    return GameState(
        game_id=str(row["game_id"]),
        play_id=int(row["play_id"]),
        season=int(row["season"]),
        week=int(row["week"]),
        quarter=int(row["qtr"]),
        game_seconds_remaining=int(float(row["game_seconds_remaining"])),
        down=int(row["down"]),
        yards_to_go=float(row["ydstogo"]),
        yardline_100=float(row["yardline_100"]),
        possession_team=str(row["posteam"]),
        defensive_team=str(row["defteam"]),
        possession_score=int(row["posteam_score"]),
        defensive_score=int(row["defteam_score"]),
        possession_timeouts=int(row.get("posteam_timeouts_remaining") or 0),
        defensive_timeouts=int(row.get("defteam_timeouts_remaining") or 0),
        win_probability=float(row["wp"]),
        expected_points=float(row["ep"]),
    )


def scenario_from_row(row: Mapping[str, Any], baseline: ExpectedPointsBaseline) -> Scenario:
    state = state_from_row(row)
    raw_id = f"{state.game_id}:{state.play_id}"
    scenario_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
    return Scenario(
        scenario_id=scenario_id,
        state=state,
        ep_baseline=baseline.values_for(row, state),
    )


def _stratum(scenario: Scenario) -> tuple[int, int, int, int, int]:
    state = scenario.state
    return (
        state.down,
        min(state.game_seconds_remaining // 300, 5),
        min(int(state.win_probability * 5), 4),
        min(int(state.yardline_100 // 20), 4),
        -1 if state.score_differential < 0 else (1 if state.score_differential > 0 else 0),
    )


def stratified_sample(scenarios: Sequence[Scenario], limit: int) -> list[Scenario]:
    groups: dict[tuple[int, int, int, int, int], list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        groups[_stratum(scenario)].append(scenario)
    for values in groups.values():
        values.sort(key=lambda item: item.scenario_id)

    # Build one balanced stream per down before interleaving the downs. Iterating
    # every sorted stratum directly can exhaust the limit on first and second down.
    streams: dict[int, deque[Scenario]] = {}
    for down in sorted({key[0] for key in groups}):
        keys = [key for key in sorted(groups) if key[0] == down]
        stream: list[Scenario] = []
        while keys:
            next_keys = []
            for key in keys:
                if groups[key]:
                    stream.append(groups[key].pop(0))
                if groups[key]:
                    next_keys.append(key)
            keys = next_keys
        streams[down] = deque(stream)

    selected: list[Scenario] = []
    while len(selected) < limit and any(streams.values()):
        for down in sorted(streams):
            if streams[down] and len(selected) < limit:
                selected.append(streams[down].popleft())
    return selected


def quickstart_sample(
    scenarios: Sequence[Scenario],
    limit: int = QUICKSTART_SCENARIO_COUNT,
) -> list[Scenario]:
    """Build a deterministic, coach-facing pack weighted toward pressure downs."""

    selected: list[Scenario] = []
    selected_ids: set[str] = set()

    def take(count: int, predicate: Callable[[Scenario], bool]) -> None:
        remaining = min(count, limit - len(selected))
        if remaining <= 0:
            return
        available = [scenario for scenario in scenarios if scenario.scenario_id not in selected_ids and predicate(scenario)]
        for scenario in stratified_sample(available, min(remaining, len(available))):
            selected.append(scenario)
            selected_ids.add(scenario.scenario_id)

    one_score = lambda scenario: abs(scenario.state.score_differential) <= 8
    take(
        6,
        lambda scenario: scenario.state.down == 4
        and scenario.state.quarter == 4
        and scenario.state.game_seconds_remaining <= 300
        and one_score(scenario),
    )
    take(
        4,
        lambda scenario: scenario.state.down == 4 and scenario.state.yards_to_go <= 3,
    )
    take(
        4,
        lambda scenario: scenario.state.down == 4 and scenario.state.yardline_100 <= 53,
    )
    take(
        8,
        lambda scenario: scenario.state.quarter == 4 and scenario.state.game_seconds_remaining <= 120 and one_score(scenario),
    )
    take(
        6,
        lambda scenario: scenario.state.down != 4
        and scenario.state.quarter == 4
        and scenario.state.game_seconds_remaining <= 300
        and one_score(scenario),
    )
    take(4, lambda scenario: scenario.state.yardline_100 <= 10)
    take(4, lambda scenario: scenario.state.down == 3)
    take(limit - len(selected), lambda scenario: True)
    return selected[:limit]


def build_scenarios(
    training_rows: Iterable[Mapping[str, Any]],
    evaluation_rows: Iterable[Mapping[str, Any]],
    limit: int = 250,
) -> list[Scenario]:
    train = [dict(row) for row in training_rows]
    if any(int(row.get("season", 0)) in EVALUATION_SEASONS for row in train):
        raise ValueError("evaluation seasons must not be used to fit the EP baseline")
    baseline = ExpectedPointsBaseline().fit(train)
    candidates = [
        scenario_from_row(row, baseline)
        for row in evaluation_rows
        if int(row.get("season", 0)) in EVALUATION_SEASONS and is_late_game_candidate(row)
    ]
    return stratified_sample(candidates, limit)


def write_jsonl(scenarios: Sequence[Scenario], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(s.model_dump_json() for s in scenarios) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def read_jsonl(path: Path) -> list[Scenario]:
    return [Scenario.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_manifest(
    path: Path,
    files: Mapping[str, str],
    scenario_count: int,
    artifact_counts: Mapping[str, int] | None = None,
) -> None:
    manifest = {
        "schema_version": "1.0",
        "source": "nflverse play-by-play",
        "source_license": "CC-BY-4.0",
        "training_seasons": list(TRAINING_SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "scenario_count": scenario_count,
        "sha256": dict(files),
    }
    if artifact_counts is not None:
        manifest["artifact_counts"] = dict(artifact_counts)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def demo_scenarios() -> list[Scenario]:
    """Deterministic synthetic pack for offline UI and smoke testing."""

    scenarios: list[Scenario] = []
    teams = (
        ("BUF", "KC"),
        ("GB", "DET"),
        ("PHI", "DAL"),
        ("BAL", "CIN"),
        ("SF", "LAR"),
    )
    for index in range(40):
        offense, defense = teams[index % len(teams)]
        down = index % 4 + 1
        offense_score = 17 + index % 11
        defense_score = offense_score + ((index % 7) - 3)
        state = GameState(
            game_id=f"DEMO-{index // 5 + 1}",
            play_id=100 + index,
            season=2025,
            week=index % 18 + 1,
            quarter=3 if index % 3 == 0 else 4,
            game_seconds_remaining=120 + (index * 67) % 1500,
            down=down,
            yards_to_go=float(1 + (index * 3) % 12),
            yardline_100=float(15 + (index * 11) % 75),
            possession_team=offense,
            defensive_team=defense,
            possession_score=offense_score,
            defensive_score=defense_score,
            possession_timeouts=index % 4,
            defensive_timeouts=(index + 2) % 4,
            win_probability=0.2 + (index % 14) * 0.045,
            expected_points=-1.0 + (index % 10) * 0.35,
        )
        defaults = {
            Action.RUN: -0.05 + (index % 4) * 0.03,
            Action.PASS: 0.01 + (index % 5) * 0.025,
            Action.PUNT: -0.12 + (index % 3) * 0.02,
            Action.FIELD_GOAL: 0.02 + (index % 6) * 0.035,
            Action.GO_FOR_IT: -0.01 + (index % 7) * 0.03,
        }
        scenarios.append(
            Scenario(
                scenario_id=f"demo-{index + 1:03d}",
                state=state,
                ep_baseline={action: defaults[action] for action in state.legal_actions},
                source="synthetic quick-start fixture",
                source_license="Apache-2.0",
            )
        )
    return scenarios

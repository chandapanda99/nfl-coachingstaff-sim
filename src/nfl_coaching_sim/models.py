"""Versioned public domain models."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCENARIO_SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "1.0"
SIMULATOR_VERSION = "1.0"


class Action(StrEnum):
    RUN = "run"
    PASS = "pass"
    PUNT = "punt"
    FIELD_GOAL = "field_goal"
    GO_FOR_IT = "go_for_it"


class GameState(BaseModel):
    """A pre-snap state expressed from the possession team's perspective."""

    model_config = ConfigDict(frozen=True)

    game_id: str
    play_id: int
    season: int
    week: int
    quarter: int = Field(ge=1, le=4)
    game_seconds_remaining: int = Field(ge=0, le=3600)
    down: int = Field(ge=1, le=4)
    yards_to_go: float = Field(gt=0, le=99)
    yardline_100: float = Field(ge=0, le=100)
    possession_team: str = Field(min_length=2)
    defensive_team: str = Field(min_length=2)
    possession_score: int = Field(ge=0)
    defensive_score: int = Field(ge=0)
    possession_timeouts: int = Field(ge=0, le=3)
    defensive_timeouts: int = Field(ge=0, le=3)
    win_probability: float = Field(ge=0, le=1)
    expected_points: float

    @property
    def score_differential(self) -> int:
        return self.possession_score - self.defensive_score

    @property
    def legal_actions(self) -> tuple[Action, ...]:
        if self.down < 4:
            return (Action.RUN, Action.PASS)
        actions = [Action.PUNT, Action.GO_FOR_IT]
        # A 70-yard attempt is the generous physical limit for this lightweight model.
        if self.yardline_100 + 17 <= 70:
            actions.append(Action.FIELD_GOAL)
        return tuple(actions)

    @property
    def clock_display(self) -> str:
        minutes, seconds = divmod(self.game_seconds_remaining % 900, 60)
        return f"Q{self.quarter} {minutes}:{seconds:02d}"


class Decision(BaseModel):
    action: Action
    go_for_it_play: Action | None = None
    rationale: str = Field(min_length=1, max_length=3000)

    @model_validator(mode="after")
    def validate_subtype(self) -> Decision:
        if self.action == Action.GO_FOR_IT:
            if self.go_for_it_play not in (Action.RUN, Action.PASS):
                raise ValueError("go_for_it requires a run or pass subtype")
        elif self.go_for_it_play is not None:
            raise ValueError("go_for_it_play is only valid for go_for_it")
        return self

    def validate_for(self, state: GameState) -> Decision:
        if self.action not in state.legal_actions:
            raise ValueError(f"{self.action.value} is illegal on down {state.down}")
        return self


class Recommendation(BaseModel):
    role: str
    decision: Decision
    confidence: float = Field(ge=0, le=1)
    argument: str = Field(min_length=1, max_length=4000)
    concerns: list[str] = Field(default_factory=list, max_length=8)


class RevisedRecommendation(Recommendation):
    rebuttal: str = Field(min_length=1, max_length=4000)


class Scenario(BaseModel):
    schema_version: str = SCENARIO_SCHEMA_VERSION
    scenario_id: str
    state: GameState
    ep_baseline: dict[Action, float]
    source: str = "nflverse"
    source_license: str = "CC-BY-4.0"

    @model_validator(mode="after")
    def validate_baseline(self) -> Scenario:
        missing = set(self.state.legal_actions) - set(self.ep_baseline)
        if missing:
            raise ValueError(f"missing EP values for {sorted(a.value for a in missing)}")
        return self


class DebateTranscript(BaseModel):
    prompt_version: str = PROMPT_VERSION
    initial: list[Recommendation]
    revised: list[RevisedRecommendation]
    head_coach: Decision
    failures: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class DecisionTrace(BaseModel):
    strategy: str
    decision: Decision
    transcript: DebateTranscript | None = None
    model_id: str | None = None
    latency_seconds: float = Field(ge=0)
    model_calls: int = Field(ge=0)
    failures: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class ActionValue(BaseModel):
    simulator_version: str = SIMULATOR_VERSION
    decision: Decision
    expected_wpa: float
    expected_epa: float
    uncertainty: float = Field(ge=0)
    oracle_regret: float = Field(ge=0)


class BenchmarkResult(BaseModel):
    scenario_id: str
    strategy: str
    decision: Decision
    expected_wpa: float
    expected_epa: float
    oracle_regret: float
    best_action: bool
    latency_seconds: float
    model_calls: int
    fallback_used: bool
    failures: list[str]
    prompt_version: str = PROMPT_VERSION
    simulator_version: str = SIMULATOR_VERSION
    model_id: str | None = None


class StageEvent(BaseModel):
    stage: str
    message: str
    trace: DecisionTrace | None = None


def action_vote(
    recommendations: list[Recommendation], scenario: Scenario
) -> Decision:
    """Deterministic vote used only when head-coach synthesis fails."""

    legal = [
        rec
        for rec in recommendations
        if rec.decision.action in scenario.state.legal_actions
    ]
    if not legal:
        action = max(
            scenario.state.legal_actions,
            key=lambda candidate: (scenario.ep_baseline[candidate], candidate.value),
        )
        subtype = Action.PASS if action == Action.GO_FOR_IT else None
        return Decision(
            action=action,
            go_for_it_play=subtype,
            rationale="Expected-points fallback because no valid agent vote was available.",
        )
    counts = Counter(rec.decision.action for rec in legal)
    confidence = {
        action: sum(rec.confidence for rec in legal if rec.decision.action == action)
        for action in counts
    }
    winner = max(
        counts,
        key=lambda action: (
            counts[action],
            confidence[action],
            scenario.ep_baseline.get(action, float("-inf")),
            action.value,
        ),
    )
    winning = [rec for rec in legal if rec.decision.action == winner]
    subtype = None
    if winner == Action.GO_FOR_IT:
        subtype_counts = Counter(rec.decision.go_for_it_play for rec in winning)
        subtype = max(
            (Action.RUN, Action.PASS),
            key=lambda action: (subtype_counts[action], action.value),
        )
    return Decision(
        action=winner,
        go_for_it_play=subtype,
        rationale="Deterministic majority fallback after head-coach synthesis failed.",
    )


def jsonable(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value

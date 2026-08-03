"""Versioned public domain models."""

from __future__ import annotations

import re
from collections import Counter
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCENARIO_SCHEMA_VERSION = "1.0"
PROMPT_VERSION = "3.0"
SIMULATOR_VERSION = "1.0"


class Action(StrEnum):
    RUN = "run"
    PASS = "pass"
    PUNT = "punt"
    FIELD_GOAL = "field_goal"
    GO_FOR_IT = "go_for_it"

    @property
    def football_label(self) -> str:
        return {
            Action.RUN: "Run the ball",
            Action.PASS: "Drop back to pass",
            Action.PUNT: "Send out the punt team",
            Action.FIELD_GOAL: "Kick the field goal",
            Action.GO_FOR_IT: "Keep the offense on the field",
        }[self]


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
        quarter_seconds = self.game_seconds_remaining - max(0, 4 - self.quarter) * 900
        minutes, seconds = divmod(max(0, min(900, quarter_seconds)), 60)
        return f"Q{self.quarter} {minutes}:{seconds:02d}"

    @property
    def down_and_distance(self) -> str:
        ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[self.down]
        goal_to_go = self.yardline_100 <= 10 and abs(self.yards_to_go - self.yardline_100) < 0.1
        distance = "Goal" if goal_to_go else f"{self.yards_to_go:g}"
        return f"{ordinal} & {distance}"

    @property
    def field_position(self) -> str:
        if self.yardline_100 == 50:
            return "the 50-yard line"
        if self.yardline_100 < 50:
            return f"the {self.defensive_team} {self.yardline_100:g}"
        return f"the {self.possession_team} {100 - self.yardline_100:g}"


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

    @property
    def call_label(self) -> str:
        if self.action == Action.GO_FOR_IT and self.go_for_it_play is not None:
            return f"Go for it — {self.go_for_it_play.football_label.lower()}"
        return self.action.football_label


class EvidenceItem(BaseModel):
    """A pre-snap fact that an agent may cite in its recommendation."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    category: Literal["state", "clock", "score", "field_position", "baseline"]
    statement: str = Field(min_length=1, max_length=500)


class SituationBrief(BaseModel):
    """Deterministic football context derived without post-play information."""

    model_config = ConfigDict(frozen=True)

    score_context: str
    clock_priority: Literal["preserve", "drain", "balanced"]
    field_zone: str
    distance_bucket: Literal["short", "manageable", "medium", "long"]
    approximate_field_goal_yards: int = Field(ge=17, le=117)
    field_goal_score_effect: str
    minimum_scoring_possessions_to_tie: int = Field(ge=0)
    two_minute_warning_pending: bool
    first_down_can_end_game: bool
    first_down_clock_window_seconds: int = Field(ge=0)
    evidence: list[EvidenceItem] = Field(min_length=1)

    @property
    def evidence_ids(self) -> set[str]:
        return {item.evidence_id for item in self.evidence}


class ActionAssessment(BaseModel):
    """A concise, evidence-linked evaluation of one legal coaching option."""

    action: Action
    advantages: list[str] = Field(min_length=1, max_length=5)
    risks: list[str] = Field(min_length=1, max_length=5)
    clock_effect: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=10)
    support_score: float = Field(ge=0, le=1)

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def normalize_evidence_id_tokens(cls, value: Any) -> Any:
        """Remove prose and punctuation accidentally appended to citation tokens."""

        if not isinstance(value, (list, tuple)):
            return value
        normalized: list[Any] = []
        for citation in value:
            if not isinstance(citation, str):
                normalized.append(citation)
                continue
            token = re.match(r"^[\s\"'`]*([A-Z][A-Z0-9_]*)", citation.strip().upper())
            normalized.append(token.group(1) if token else citation.strip())
        return list(dict.fromkeys(normalized))


class Recommendation(BaseModel):
    role: str
    decision: Decision
    confidence: float = Field(ge=0, le=1)
    argument: str = Field(min_length=1, max_length=4000)
    concerns: list[str] = Field(default_factory=list, max_length=8)
    action_assessments: list[ActionAssessment] = Field(min_length=1, max_length=5)
    closest_alternative: Action
    switch_condition: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_action_assessments(self) -> Recommendation:
        actions = [assessment.action for assessment in self.action_assessments]
        if len(actions) != len(set(actions)):
            raise ValueError("action assessments must contain unique actions")
        if self.closest_alternative == self.decision.action:
            raise ValueError("closest_alternative must differ from the recommended action")
        return self


class RevisedRecommendation(Recommendation):
    rebuttal: str = Field(min_length=1, max_length=4000)


class Scenario(BaseModel):
    schema_version: str = SCENARIO_SCHEMA_VERSION
    scenario_id: str
    state: GameState
    ep_baseline: dict[Action, float]
    source: str = "nflverse"
    source_license: str = "CC-BY-4.0"
    name: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def validate_baseline(self) -> Scenario:
        missing = set(self.state.legal_actions) - set(self.ep_baseline)
        if missing:
            raise ValueError(f"missing EP values for {sorted(a.value for a in missing)}")
        return self

    @property
    def display_name(self) -> str:
        state = self.state
        game = f"{state.possession_team} {state.possession_score}–{state.defensive_score} {state.defensive_team}"
        field_position = state.field_position.removeprefix("the ")
        situation = f"{state.clock_display} · {game} · {state.down_and_distance} · {field_position}"
        return f"{self.name} · {situation}" if self.name else situation


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
    role: str | None = None
    recommendation: Recommendation | None = None
    revision: RevisedRecommendation | None = None
    failure: str | None = None
    trace: DecisionTrace | None = None


def action_vote(recommendations: list[Recommendation], scenario: Scenario) -> Decision:
    """Deterministic vote used only when head-coach synthesis fails."""

    legal = [rec for rec in recommendations if rec.decision.action in scenario.state.legal_actions]
    if not legal:
        action = max(
            scenario.state.legal_actions,
            key=lambda candidate: (scenario.ep_baseline[candidate], candidate.value),
        )
        subtype = Action.PASS if action == Action.GO_FOR_IT else None
        return Decision(
            action=action,
            go_for_it_play=subtype,
            rationale=("The staff could not get a clean call through, so the analytics booth " "sent in the highest-EPA option."),
        )
    counts = Counter(rec.decision.action for rec in legal)
    confidence = {action: sum(rec.confidence for rec in legal if rec.decision.action == action) for action in counts}
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
        rationale=("The head-coach model could not get the call in before the play clock, " "so the staff consensus won the tiebreak."),
    )


def jsonable(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value

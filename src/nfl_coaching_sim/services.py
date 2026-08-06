"""Framework-neutral application services for the coaching simulator."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

from nfl_coaching_sim.data import ExpectedPointsBaseline
from nfl_coaching_sim.models import ActionValue, DecisionTrace, GameState, Scenario, StageEvent
from nfl_coaching_sim.scenario_library import (
    CUSTOM_SCENARIO_SOURCE,
    default_custom_scenarios_path,
    delete_custom_scenario,
    load_custom_scenarios,
    save_custom_scenario,
)
from nfl_coaching_sim.settings import ApplicationSettings, get_application_settings
from nfl_coaching_sim.simulator import DeterministicSimulator


class CustomScenarioInput(BaseModel):
    """Coach-friendly values used to create or edit a saved situation."""

    name: str = Field(min_length=1, max_length=80)
    season: int = Field(ge=2000, le=2100)
    week: int = Field(ge=1, le=22)
    possession_team: str = Field(min_length=2, max_length=4)
    defensive_team: str = Field(min_length=2, max_length=4)
    possession_score: int = Field(ge=0)
    defensive_score: int = Field(ge=0)
    quarter: int = Field(ge=1, le=4)
    clock: str
    down: int = Field(ge=1, le=4)
    yards_to_go: float = Field(gt=0, le=99)
    field_side: Literal["offense", "midfield", "defense"]
    yard_line: float = Field(ge=1, le=49)
    possession_timeouts: int = Field(ge=0, le=3)
    defensive_timeouts: int = Field(ge=0, le=3)
    win_probability_percent: float | None = Field(default=None, ge=0, le=100)
    expected_points: float | None = None


class DeliberationInput(BaseModel):
    """Provider-neutral request for one coaching decision."""

    scenario_id: str
    strategy: Literal["expected_points", "single_agent", "multi_agent"] = "multi_agent"
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    upstream_url: HttpUrl | None = None
    model_license: str | None = None
    reasoning_effort: str | None = None


class ApplicationEvent(BaseModel):
    """One response-level update consumed by web and desktop clients."""

    stage: str
    message: str
    role: str | None = None
    recommendation: dict | None = None
    revision: dict | None = None
    failure: str | None = None
    trace: DecisionTrace | None = None
    score: ActionValue | None = None


def create_custom_scenario(values: CustomScenarioInput) -> Scenario:
    """Build a validated custom scenario without depending on a UI framework."""

    situation_name = values.name.strip()
    offense = values.possession_team.strip().upper()
    defense = values.defensive_team.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,4}", offense):
        raise ValueError("Offense must be a 2–4 character team abbreviation, such as BUF, CHI, or KC.")
    if not re.fullmatch(r"[A-Z0-9]{2,4}", defense):
        raise ValueError("Defense must be a 2–4 character team abbreviation, such as BUF, CHI, or KC.")
    if offense == defense:
        raise ValueError("Offense and defense must be different teams.")

    clock_match = re.fullmatch(r"\s*(\d{1,2}):([0-5]\d)\s*", values.clock)
    if clock_match is None:
        raise ValueError("Game clock must use MM:SS format, such as 2:35.")
    minutes, seconds = (int(value) for value in clock_match.groups())
    if minutes > 15 or (minutes == 15 and seconds != 0):
        raise ValueError("Game clock must be between 0:00 and 15:00.")
    seconds_in_quarter = minutes * 60 + seconds

    if values.field_side == "midfield":
        yardline_100 = 50.0
    elif values.field_side == "offense":
        yardline_100 = 100.0 - values.yard_line
    else:
        yardline_100 = values.yard_line

    game_seconds_remaining = (4 - values.quarter) * 900 + seconds_in_quarter
    score_differential = values.possession_score - values.defensive_score
    if values.win_probability_percent is None:
        elapsed_share = 1 - game_seconds_remaining / 3600
        log_odds = score_differential * (0.12 + 0.2 * elapsed_share) + (50 - yardline_100) * 0.012
        win_probability = 1 / (1 + math.exp(-log_odds))
    else:
        win_probability = values.win_probability_percent / 100

    expected_points = values.expected_points
    if expected_points is None:
        expected_points = max(
            -2.5,
            min(6.5, 6.5 - 0.075 * yardline_100 - 0.35 * (values.down - 1) - 0.04 * max(0, values.yards_to_go - 10)),
        )

    identity = "|".join(
        str(value)
        for value in (
            values.season,
            values.week,
            offense,
            defense,
            values.possession_score,
            values.defensive_score,
            values.quarter,
            seconds_in_quarter,
            values.down,
            values.yards_to_go,
            yardline_100,
            values.possession_timeouts,
            values.defensive_timeouts,
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    state = GameState(
        game_id=f"CUSTOM-{digest}",
        play_id=int(digest[:8], 16),
        season=values.season,
        week=values.week,
        quarter=values.quarter,
        game_seconds_remaining=game_seconds_remaining,
        down=values.down,
        yards_to_go=values.yards_to_go,
        yardline_100=yardline_100,
        possession_team=offense,
        defensive_team=defense,
        possession_score=values.possession_score,
        defensive_score=values.defensive_score,
        possession_timeouts=values.possession_timeouts,
        defensive_timeouts=values.defensive_timeouts,
        win_probability=win_probability,
        expected_points=expected_points,
    )
    baseline_row = {
        "down": state.down,
        "ydstogo": state.yards_to_go,
        "yardline_100": state.yardline_100,
        "game_seconds_remaining": state.game_seconds_remaining,
        "posteam_score": state.possession_score,
        "defteam_score": state.defensive_score,
    }
    return Scenario(
        scenario_id=f"custom-{digest}",
        state=state,
        ep_baseline=ExpectedPointsBaseline().values_for(baseline_row, state),
        source=CUSTOM_SCENARIO_SOURCE,
        source_license="User-provided",
        name=situation_name,
    )


class ScenarioRepository:
    """Read prebuilt scenarios and persist the current user's custom call sheet."""

    def __init__(self, prebuilt: Sequence[Scenario], custom_path: Path | None = None) -> None:
        self._prebuilt = list(prebuilt)
        self.custom_path = custom_path or default_custom_scenarios_path()

    def list(self, library: Literal["prebuilt", "custom", "all"] = "all") -> list[Scenario]:
        prebuilt = self._prebuilt if library in {"prebuilt", "all"} else []
        custom = load_custom_scenarios(self.custom_path) if library in {"custom", "all"} else []
        return [*prebuilt, *custom]

    def get(self, scenario_id: str) -> Scenario:
        try:
            return next(item for item in self.list() if item.scenario_id == scenario_id)
        except StopIteration as error:
            raise KeyError(f"unknown scenario: {scenario_id}") from error

    def save(self, values: CustomScenarioInput, replacing_scenario_id: str | None = None) -> Scenario:
        if replacing_scenario_id is not None and not any(
            item.scenario_id == replacing_scenario_id for item in load_custom_scenarios(self.custom_path)
        ):
            raise KeyError(f"unknown custom scenario: {replacing_scenario_id}")
        scenario = create_custom_scenario(values)
        save_custom_scenario(self.custom_path, scenario, replacing_scenario_id=replacing_scenario_id)
        return scenario

    def delete(self, scenario_id: str) -> None:
        delete_custom_scenario(self.custom_path, scenario_id)


class CoachingApplication:
    """Coordinates strategies and deterministic scoring for every presentation layer."""

    def __init__(
        self,
        scenarios: ScenarioRepository,
        simulator: DeterministicSimulator,
        settings: ApplicationSettings | None = None,
    ) -> None:
        self.scenarios = scenarios
        self.simulator = simulator
        self.settings = settings or get_application_settings()

    def _strategy(self, request: DeliberationInput):
        from nfl_coaching_sim.agents import ExpectedPointsStrategy, MultiAgentStrategy, SingleAgentStrategy, make_model
        from nfl_coaching_sim.providers.base import ModelConfiguration

        if request.strategy == "expected_points":
            return ExpectedPointsStrategy()
        provider = request.provider or self.settings.provider
        defaults = get_application_settings(provider)
        configuration = ModelConfiguration(
            provider=provider,
            model=request.model or defaults.model,
            base_url=request.base_url or defaults.base_url,
            upstream_url=request.upstream_url or (HttpUrl(defaults.upstream_url) if defaults.upstream_url else None),
            license=request.model_license or defaults.model_license,
            reasoning_effort=request.reasoning_effort or defaults.reasoning_effort,
        )
        model = make_model(configuration)
        return SingleAgentStrategy(model) if request.strategy == "single_agent" else MultiAgentStrategy(model)

    @staticmethod
    def _event(event: StageEvent) -> ApplicationEvent:
        return ApplicationEvent(
            stage=event.stage,
            message=event.message,
            role=event.role,
            recommendation=event.recommendation.model_dump(mode="json") if event.recommendation else None,
            revision=event.revision.model_dump(mode="json") if event.revision else None,
            failure=event.failure,
        )

    def iter_deliberation(self, request: DeliberationInput) -> Iterator[ApplicationEvent]:
        from nfl_coaching_sim.agents import MultiAgentStrategy

        scenario = self.scenarios.get(request.scenario_id)
        strategy = self._strategy(request)
        yield ApplicationEvent(stage="started", message="The headset is open and the call sheet is in the huddle.")
        if isinstance(strategy, MultiAgentStrategy):
            trace: DecisionTrace | None = None
            for stage_event in strategy.iter_decide(scenario):
                if stage_event.trace is not None:
                    trace = stage_event.trace
                yield self._event(stage_event)
            if trace is None:
                raise RuntimeError("The coaching staff finished without sending in a legal call.")
        else:
            trace = strategy.decide(scenario)
        score = self.simulator.score(scenario, trace.decision)
        yield ApplicationEvent(
            stage="completed",
            message=f"Call is in: {trace.decision.call_label}.",
            trace=trace,
            score=score,
        )

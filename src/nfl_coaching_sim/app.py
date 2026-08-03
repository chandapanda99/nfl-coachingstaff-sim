"""Gradio Blocks application with thin, reusable callbacks."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import find_dotenv, load_dotenv
from pydantic import HttpUrl

from nfl_coaching_sim.agents import (
    ExpectedPointsStrategy,
    ModelConfiguration,
    ModelProvider,
    MultiAgentStrategy,
    SingleAgentStrategy,
    make_model,
    model_provider_choices,
)
from nfl_coaching_sim.data import ExpectedPointsBaseline, demo_scenarios
from nfl_coaching_sim.football import build_situation_brief
from nfl_coaching_sim.models import (
    Action,
    ActionValue,
    Decision,
    DecisionTrace,
    GameState,
    Recommendation,
    RevisedRecommendation,
    Scenario,
)
from nfl_coaching_sim.scenario_library import (
    CUSTOM_SCENARIO_SOURCE,
    delete_custom_scenario,
    is_custom_scenario,
    load_custom_scenarios,
    save_custom_scenario,
)

if TYPE_CHECKING:
    from nfl_coaching_sim.simulator import DeterministicSimulator

load_dotenv(find_dotenv())


def load_app_css() -> str:
    """Load the packaged Gradio stylesheet."""

    return files("nfl_coaching_sim").joinpath("app.css").read_text(encoding="utf-8")


def create_app_theme() -> Any:
    """Create the football-inspired light and dark Gradio theme."""

    import gradio as gr

    return gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="blue",
        neutral_hue="slate",
        font=("Aptos", "ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=("ui-monospace", "Consolas", "monospace"),
    ).set(
        button_primary_background_fill="#c2410c",
        button_primary_background_fill_dark="#ea580c",
        button_primary_background_fill_hover="#9a3412",
        button_primary_background_fill_hover_dark="#f97316",
        button_primary_border_color="#9a3412",
        button_primary_border_color_dark="#fb923c",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
    )


def create_custom_scenario(
    name: str,
    season: int,
    week: int,
    possession_team: str,
    defensive_team: str,
    possession_score: int,
    defensive_score: int,
    quarter: int,
    clock: str,
    down: int,
    yards_to_go: float,
    field_side: str,
    yard_line: float,
    possession_timeouts: int,
    defensive_timeouts: int,
    win_probability_percent: float | None = None,
    expected_points: float | None = None,
) -> Scenario:
    """Build a validated user-created scenario from coach-friendly form values."""

    situation_name = name.strip()
    if not situation_name:
        raise ValueError("Give the situation a short name so it is easy to find on your call sheet.")

    offense = possession_team.strip().upper()
    defense = defensive_team.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,4}", offense):
        raise ValueError("Offense must be a 2–4 character team abbreviation, such as BUF, CHI, KC, etc...")
    if not re.fullmatch(r"[A-Z0-9]{2,4}", defense):
        raise ValueError("Defense must be a 2–4 character team abbreviation, such as BUF, CHI, KC, etc...")
    if offense == defense:
        raise ValueError("Offense and defense must be different teams.")

    clock_match = re.fullmatch(r"\s*(\d{1,2}):([0-5]\d)\s*", clock)
    if clock_match is None:
        raise ValueError("Game clock must use MM:SS format, such as 2:35.")
    minutes, seconds = (int(value) for value in clock_match.groups())
    if minutes > 15 or (minutes == 15 and seconds != 0):
        raise ValueError("Game clock must be between 0:00 and 15:00.")
    seconds_in_quarter = minutes * 60 + seconds

    normalized_side = field_side.strip().lower()
    if normalized_side == "midfield":
        yardline_100 = 50.0
    elif normalized_side == "offense":
        if not 1 <= yard_line <= 49:
            raise ValueError("On the offense's side, enter a yard line from 1 through 49.")
        yardline_100 = 100.0 - float(yard_line)
    elif normalized_side == "defense":
        if not 1 <= yard_line <= 49:
            raise ValueError("On the defense's side, enter a yard line from 1 through 49.")
        yardline_100 = float(yard_line)
    else:
        raise ValueError("Choose whether the ball is on the offense's side, at midfield, or on the defense's side.")

    game_seconds_remaining = (4 - int(quarter)) * 900 + seconds_in_quarter
    score_differential = int(possession_score) - int(defensive_score)
    if win_probability_percent is None:
        elapsed_share = 1 - game_seconds_remaining / 3600
        log_odds = score_differential * (0.12 + 0.2 * elapsed_share) + (50 - yardline_100) * 0.012
        win_probability = 1 / (1 + math.exp(-log_odds))
    else:
        if not 0 <= win_probability_percent <= 100:
            raise ValueError("Offense win probability must be between 0 and 100 percent.")
        win_probability = float(win_probability_percent) / 100

    if expected_points is None:
        expected_points_value = max(
            -2.5,
            min(6.5, 6.5 - 0.075 * yardline_100 - 0.35 * (int(down) - 1) - 0.04 * max(0, yards_to_go - 10)),
        )
    else:
        expected_points_value = float(expected_points)

    identity = "|".join(
        str(value)
        for value in (
            season,
            week,
            offense,
            defense,
            possession_score,
            defensive_score,
            quarter,
            seconds_in_quarter,
            down,
            yards_to_go,
            yardline_100,
            possession_timeouts,
            defensive_timeouts,
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    state = GameState(
        game_id=f"CUSTOM-{digest}",
        play_id=int(digest[:8], 16),
        season=int(season),
        week=int(week),
        quarter=int(quarter),
        game_seconds_remaining=game_seconds_remaining,
        down=int(down),
        yards_to_go=float(yards_to_go),
        yardline_100=yardline_100,
        possession_team=offense,
        defensive_team=defense,
        possession_score=int(possession_score),
        defensive_score=int(defensive_score),
        possession_timeouts=int(possession_timeouts),
        defensive_timeouts=int(defensive_timeouts),
        win_probability=win_probability,
        expected_points=expected_points_value,
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


def _live_status(message: str) -> str:
    return f"### Live Sideline\n\n## {message}"


def _safe(value: Any) -> str:
    return escape(str(value), quote=True)


def _pre_snap_card(scenario: Scenario) -> str:
    state = scenario.state
    differential = state.score_differential
    if differential > 0:
        game_margin = f"Leading by {differential}"
    elif differential < 0:
        game_margin = f"Trailing by {abs(differential)}"
    else:
        game_margin = "Tie game"
    legal_calls = "".join(f'<span class="call-chip">{_safe(action.football_label)}</span>' for action in state.legal_actions)
    win_probability = max(0.0, min(1.0, state.win_probability))
    return f"""
<section class="coach-card situation-card" aria-label="Pre-snap situation data">
  <header class="coach-card__header">
    <div>
      <span class="coach-card__eyebrow">Pre-Snap Situation Data</span>
      <h3>Situation at a Glance</h3>
    </div>
    <span class="game-tag">{state.season} · Week {state.week}</span>
  </header>
  <div class="score-strip">
    <div class="team-score">
      <small>Offense · Ball</small>
      <div class="team-score__line">
        <span>{_safe(state.possession_team)}</span>
        <strong>{state.possession_score}</strong>
      </div>
    </div>
    <span class="score-strip__divider">vs</span>
    <div class="team-score">
      <small>Defense</small>
      <div class="team-score__line">
        <span>{_safe(state.defensive_team)}</span>
        <strong>{state.defensive_score}</strong>
      </div>
    </div>
  </div>
  <div class="situation-grid">
    <div class="situation-fact"><span>Game Clock</span><strong>{_safe(state.clock_display)}</strong></div>
    <div class="situation-fact"><span>Down &amp; Distance</span><strong>{_safe(state.down_and_distance)}</strong></div>
    <div class="situation-fact"><span>Field Position</span><strong>{_safe(state.field_position)}</strong></div>
    <div class="situation-fact"><span>Score Situation</span><strong>{_safe(game_margin)}</strong></div>
    <div class="situation-fact"><span>Timeouts</span><strong>
      {state.possession_timeouts} Offense · {state.defensive_timeouts} Defense
    </strong></div>
    <div class="situation-fact"><span>Current Expected Points</span><strong>{state.expected_points:+.3f} points</strong></div>
  </div>
  <div class="probability-block">
    <div><span>Offense Win Probability</span><strong>{win_probability:.1%}</strong></div>
    <div class="probability-track" role="progressbar" aria-label="Offense win probability"
         aria-valuemin="0" aria-valuemax="100" aria-valuenow="{win_probability * 100:.1f}">
      <span style="width: {win_probability * 100:.1f}%"></span>
    </div>
  </div>
  <div class="legal-calls"><span>Calls Available</span><div>{legal_calls}</div></div>
</section>
""".strip()


def _analytics_card(scenario: Scenario) -> str:
    best_action = max(
        scenario.state.legal_actions,
        key=lambda action: (scenario.ep_baseline[action], action.value),
    )
    rows = []
    for action in scenario.state.legal_actions:
        expected_epa = scenario.ep_baseline[action]
        value_class = "metric-positive" if expected_epa >= 0 else "metric-negative"
        top_call = '<span class="top-call-chip">Top option</span>' if action == best_action else ""
        rows.append(f"""
      <tr class="{'analytics-table__best' if action == best_action else ''}">
        <td><strong>{_safe(action.football_label)}</strong>{top_call}</td>
        <td class="{value_class}"><strong>{expected_epa:+.3f}</strong></td>
      </tr>
""".rstrip())
    return f"""
<section class="coach-card analytics-card" aria-label="Analytics Booth expected points table">
  <header class="coach-card__header">
    <div>
      <span class="coach-card__eyebrow">Analytics Booth</span>
      <h3>Expected Value by Call</h3>
    </div>
    <span class="game-tag">EPA</span>
  </header>
  <table class="analytics-table">
    <thead><tr><th>Call Sheet Option</th><th>Expected EPA</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p class="analytics-note">Higher EPA indicates higher likelihood of scoring based on down, distance, yard line, and time remaining.</p>
</section>
""".strip()


def _decision_placeholder() -> str:
    return """
<section class="coach-card result-card result-card--empty" aria-label="Head Coach's call">
  <span class="coach-card__eyebrow">Head Coach's Call</span>
  <h3>Waiting on the sideline</h3>
  <p>Send in the call to see the selected play and the staff's reasoning.</p>
</section>
""".strip()


def _decision_card(trace: DecisionTrace) -> str:
    strategy_name = {
        "expected_points": "Analytics Booth",
        "single_agent": "Head Coach",
        "multi_agent": "Full Coaching Staff",
    }.get(trace.strategy, trace.strategy.replace("_", " ").title())
    icon = {
        Action.RUN: "🏃",
        Action.PASS: "🎯",
        Action.PUNT: "🦵",
        Action.FIELD_GOAL: "🦵🥅",
        Action.GO_FOR_IT: "📣",
    }[trace.decision.action]
    fallback = '<span class="warning-chip">Fallback Call Used</span>' if trace.fallback_used else ""
    model_name = _safe(trace.model_id or "Deterministic Analytics Policy")
    return f"""
<section class="coach-card result-card decision-card" aria-label="Head Coach's call">
  <header class="coach-card__header">
    <div>
      <span class="coach-card__eyebrow">Head Coach's Call</span>
      <h3>{_safe(strategy_name)}</h3>
    </div>
    {fallback}
  </header>
  <div class="play-call">
    <span class="play-call__icon" aria-hidden="true">{icon}</span>
    <div><span>Call Sent to the Huddle</span><strong>{_safe(trace.decision.call_label)}</strong></div>
  </div>
  <div class="call-rationale">
    <span>Why this call?</span>
    <p>{_safe(trace.decision.rationale)}</p>
  </div>
  <footer class="card-metadata">
    <span>{model_name}</span>
    <span>{trace.model_calls} model calls</span>
    <span>{trace.latency_seconds:.2f}s</span>
  </footer>
</section>
""".strip()


def _grade_placeholder() -> str:
    return """
<section class="coach-card result-card result-card--empty" aria-label="Postgame Decision Grade">
  <span class="coach-card__eyebrow">Postgame Decision Grade</span>
  <h3>No grade on the board yet</h3>
  <p>The simulator will grade the call after it reaches the huddle.</p>
</section>
""".strip()


def _conversation_placeholder() -> str:
    return (
        "### Coaches' Headset\n\n"
        "The headset is quiet. Choose who is making the call, then send the situation to the staff. "
        "Completed coaching responses will appear here as they come in."
    )


def _grade_card(value: ActionValue) -> str:
    regret = value.oracle_regret
    if regret <= 0.0005:
        grade, grade_detail, grade_class = "Best Available Call", "No measurable regret", "grade-banner--great"
    elif regret <= 0.005:
        grade, grade_detail, grade_class = "Strong Call", "Minimal gap from the top option", "grade-banner--good"
    elif regret <= 0.02:
        grade, grade_detail, grade_class = "Questionable Call", "Noticeable value left on the field", "grade-banner--caution"
    else:
        grade, grade_detail, grade_class = "Costly Call", "Significant value left on the field", "grade-banner--poor"
    wpa_class = "metric-positive" if value.expected_wpa >= 0 else "metric-negative"
    epa_class = "metric-positive" if value.expected_epa >= 0 else "metric-negative"
    return f"""
<section class="coach-card result-card grade-card" aria-label="Postgame decision grade">
  <header class="coach-card__header">
    <div>
      <span class="coach-card__eyebrow">Postgame Decision Grade</span>
      <h3>Simulator Review</h3>
    </div>
    <span class="game-tag">v{_safe(value.simulator_version)}</span>
  </header>
  <div class="grade-banner {grade_class}">
    <strong>{grade}</strong>
    <span>{grade_detail}</span>
  </div>
  <div class="grade-grid">
    <div class="grade-metric {wpa_class}">
      <span>Win Probability Added</span>
      <strong>{value.expected_wpa * 100:+.2f} pp</strong>
      <small>Expected change from this call</small>
    </div>
    <div class="grade-metric {epa_class}">
      <span>Expected Points Added</span>
      <strong>{value.expected_epa:+.3f}</strong>
      <small>Expected scoreboard value</small>
    </div>
    <div class="grade-metric">
      <span>Model Uncertainty</span>
      <strong>±{value.uncertainty * 100:.2f} pp</strong>
      <small>Estimated WPA error band</small>
    </div>
    <div class="grade-metric">
      <span>Gap from Best Call</span>
      <strong>{value.oracle_regret * 100:.2f} pp</strong>
      <small>Expected WPA regret</small>
    </div>
  </div>
</section>
""".strip()


def scenario_view(scenario_id: str, scenario_payloads: Sequence[dict[str, Any]]) -> tuple[str, str]:
    scenario = next(Scenario.model_validate(item) for item in scenario_payloads if item["scenario_id"] == scenario_id)
    return _pre_snap_card(scenario), _analytics_card(scenario)


def scenario_view_with_reset(scenario_id: str, scenario_payloads: Sequence[dict[str, Any]]) -> tuple[str, str, str, str, str, str]:
    """Render a fresh call sheet and clear analysis from the prior situation."""

    state_html, analytics_html = scenario_view(scenario_id, scenario_payloads)
    return (
        state_html,
        analytics_html,
        _live_status("New situation is on the call sheet. Ready for Coach to send in the call."),
        _conversation_placeholder(),
        _decision_placeholder(),
        _grade_placeholder(),
    )


def scenarios_for_library(library: str, scenario_payloads: Sequence[dict[str, Any]]) -> list[Scenario]:
    """Return the selected, consistently ordered call-sheet section."""

    scenarios = [Scenario.model_validate(item) for item in scenario_payloads]
    wants_custom = library == "custom"
    return [scenario for scenario in scenarios if is_custom_scenario(scenario) == wants_custom]


_NEW_SITUATION_FORM = (
    "Two-Minute Decision",
    2025,
    1,
    "BUF",
    "KC",
    20,
    20,
    4,
    "2:00",
    4,
    2,
    "defense",
    35,
    2,
    2,
    None,
    None,
)


def custom_scenario_form_values(scenario: Scenario) -> tuple[Any, ...]:
    """Translate a saved scenario back into the coach-friendly editor fields."""

    if not is_custom_scenario(scenario):
        raise ValueError("only custom situations can be edited")
    state = scenario.state
    quarter_seconds = state.game_seconds_remaining - max(0, 4 - state.quarter) * 900
    minutes, seconds = divmod(max(0, min(900, quarter_seconds)), 60)
    if state.yardline_100 == 50:
        field_side, yard_line = "midfield", 35
    elif state.yardline_100 > 50:
        field_side, yard_line = "offense", 100 - state.yardline_100
    else:
        field_side, yard_line = "defense", state.yardline_100
    return (
        scenario.name or "Custom Situation",
        state.season,
        state.week,
        state.possession_team,
        state.defensive_team,
        state.possession_score,
        state.defensive_score,
        state.quarter,
        f"{minutes}:{seconds:02d}",
        state.down,
        state.yards_to_go,
        field_side,
        yard_line,
        state.possession_timeouts,
        state.defensive_timeouts,
        state.win_probability * 100,
        state.expected_points,
    )


_COACH_TITLES = {
    "offensive_coordinator": "Offensive Coordinator",
    "defensive_coordinator": "Defensive Coordinator",
    "analytics_assistant": "Analytics Assistant",
    "clock_management_specialist": "Clock Management",
    "critical_reviewer": "Quality Control Coach",
}


def _spoken_text(value: str, scenario: Scenario) -> str:
    """Replace any accidentally spoken private evidence tag with its football fact."""

    text = value.strip()
    evidence = build_situation_brief(scenario).evidence
    for item in sorted(evidence, key=lambda candidate: len(candidate.evidence_id), reverse=True):
        text = text.replace(item.evidence_id, item.statement)
    return text


def _visible_headset_failure(value: str) -> str:
    """Keep troubleshooting useful without exposing structured-output internals."""

    speaker, separator, detail = value.partition(": ")
    speaker_label = speaker.replace("_", " ").title()
    if "unknown evidence" in detail.lower():
        detail = "The response referenced a call-sheet fact that could not be verified, so the analytics fallback took that rep."
    elif "structured model output failed" in detail.lower():
        detail = "The response did not come through cleanly before the play clock, so the analytics fallback took that rep."
    return f"{speaker_label}: {detail}" if separator else value


def _opening_coach_message(rec: Recommendation, scenario: Scenario) -> str:
    title = _COACH_TITLES.get(rec.role, rec.role.replace("_", " ").title())
    concerns = "; ".join(_spoken_text(item, scenario) for item in rec.concerns[:2]) or "No additional alert."
    return (
        f"#### {title}\n\n"
        f"> {_spoken_text(rec.argument, scenario)}\n\n"
        f"**My call:** {rec.decision.call_label}  \n"
        f"**Alert:** {concerns}  \n"
        f"**Check to {rec.closest_alternative.football_label.lower()} if:** "
        f"{_spoken_text(rec.switch_condition, scenario)}"
    )


def _revision_coach_message(rec: RevisedRecommendation, scenario: Scenario) -> str:
    title = _COACH_TITLES.get(rec.role, rec.role.replace("_", " ").title())
    return (
        f"#### {title} checks back in\n\n"
        f"> {_spoken_text(rec.rebuttal, scenario)}\n\n"
        f"**Final recommendation:** {rec.decision.call_label}  \n"
        f"**Last-second check:** {_spoken_text(rec.switch_condition, scenario)}"
    )


def format_live_coaching_conversation(
    scenario: Scenario,
    initial_by_role: Mapping[str, Recommendation],
    revised_by_role: Mapping[str, RevisedRecommendation],
    phase: str = "opening",
    head_coach: Decision | None = None,
    failures: Sequence[str] = (),
) -> str:
    """Render stable coach slots while completed structured responses arrive."""

    sections = [
        "### Opening Headset Check",
        "Each coach gets one clean turn before the staff challenges the call.",
    ]
    for role, title in _COACH_TITLES.items():
        recommendation = initial_by_role.get(role)
        if recommendation is None:
            sections.append(f"#### {title}\n\n> _Headset open — reviewing the call sheet…_")
        else:
            sections.append(_opening_coach_message(recommendation, scenario))

    if phase in {"revision", "decision"}:
        sections.extend(
            [
                "### Staff Challenge Round",
                "The coaches hear the other calls, challenge the weak spots, and check back in.",
            ]
        )
        for role, title in _COACH_TITLES.items():
            revision = revised_by_role.get(role)
            if revision is None:
                sections.append(f"#### {title} checks back in\n\n> _Listening to the staff challenge…_")
            else:
                sections.append(_revision_coach_message(revision, scenario))

    if phase == "decision":
        if head_coach is None:
            sections.append("### Head Coach Breaks the Huddle\n\n> _Weighing the staff's final recommendations…_")
        else:
            sections.append(
                "### Head Coach Breaks the Huddle\n\n"
                f"> {_spoken_text(head_coach.rationale, scenario)}\n\n"
                f"**Call:** {head_coach.call_label}"
            )
    if failures:
        sections.append("### Headset / Communication Issues\n\n" + "\n".join(f"- {_visible_headset_failure(item)}" for item in failures))
    return "\n\n".join(sections)


def format_coaching_conversation(trace: DecisionTrace, scenario: Scenario) -> str:
    """Render structured deliberation as a natural sideline conversation."""

    if trace.transcript is None:
        strategy_name = {
            "expected_points": "Analytics Booth",
            "single_agent": "Head Coach",
        }.get(trace.strategy, trace.strategy.replace("_", " ").title())
        return f"### {strategy_name}: {trace.decision.call_label}\n\n{_spoken_text(trace.decision.rationale, scenario)}"
    return format_live_coaching_conversation(
        scenario,
        {item.role: item for item in trace.transcript.initial},
        {item.role: item for item in trace.transcript.revised},
        phase="decision",
        head_coach=trace.transcript.head_coach,
        failures=trace.failures,
    )


def run_strategy_events(
    scenario_id: str,
    strategy_name: str,
    provider_name: str,
    model_name: str,
    upstream_url: str,
    model_license: str,
    base_url: str,
    scenario_payloads: Sequence[dict[str, Any]],
    simulator: DeterministicSimulator,
) -> Iterator[tuple[str, str, str, str]]:
    scenario = next(Scenario.model_validate(item) for item in scenario_payloads if item["scenario_id"] == scenario_id)
    yield (
        _live_status("Getting the Call Sheet ready..."),
        _conversation_placeholder(),
        _decision_placeholder(),
        _grade_placeholder(),
    )
    live_transcript = format_live_coaching_conversation(scenario, {}, {}, phase="opening") if strategy_name == "multi_agent" else ""
    yield _live_status("Team is huddling up..."), live_transcript, _decision_placeholder(), _grade_placeholder()
    if strategy_name == "expected_points":
        strategy: Any = ExpectedPointsStrategy()
        trace = strategy.decide(scenario)
    else:
        configuration = ModelConfiguration(
            provider=provider_name,
            model=model_name,
            upstream_url=HttpUrl(upstream_url) if upstream_url else None,
            license=model_license,
            base_url=base_url,
        )
        model = make_model(configuration)
        if strategy_name == "single_agent":
            trace = SingleAgentStrategy(model).decide(scenario)
        else:
            strategy = MultiAgentStrategy(model)
            trace = None
            initial_by_role: dict[str, Recommendation] = {}
            revised_by_role: dict[str, RevisedRecommendation] = {}
            live_failures: list[str] = []
            phase = "opening"
            for event in strategy.iter_decide(scenario):
                if event.role and event.recommendation is not None:
                    initial_by_role[event.role] = event.recommendation
                if event.role and event.revision is not None:
                    revised_by_role[event.role] = event.revision
                if event.failure and event.failure not in live_failures:
                    live_failures.append(event.failure)
                if event.stage == "recommendations":
                    phase = "revision"
                elif event.stage == "debate":
                    phase = "decision"
                if event.trace is not None:
                    trace = event.trace
                    phase = "decision"
                    live_failures = list(trace.failures)
                live_transcript = format_live_coaching_conversation(
                    scenario,
                    initial_by_role,
                    revised_by_role,
                    phase=phase,
                    head_coach=trace.decision if trace is not None else None,
                    failures=live_failures,
                )
                yield _live_status(event.message), live_transcript, _decision_placeholder(), _grade_placeholder()
            if trace is None:
                raise RuntimeError("The coaches' meeting ended without a legal call reaching the sideline.")
    value = simulator.score(scenario, trace.decision)
    yield (
        _live_status(f"CALL IS IN: **{trace.decision.call_label}**"),
        format_coaching_conversation(trace, scenario),
        _decision_card(trace),
        _grade_card(value),
    )


def create_app(
    scenarios: Sequence[Scenario] | None = None,
    simulator: DeterministicSimulator | None = None,
    custom_scenarios_path: Path | None = None,
) -> Any:
    import gradio as gr

    prebuilt_scenarios = list(scenarios or demo_scenarios())
    saved_custom_scenarios = load_custom_scenarios(custom_scenarios_path) if custom_scenarios_path is not None else []
    scenario_list = [*prebuilt_scenarios]
    known_ids = {item.scenario_id for item in scenario_list}
    scenario_list.extend(item for item in saved_custom_scenarios if item.scenario_id not in known_ids)
    if simulator is None:
        from nfl_coaching_sim.simulator import DeterministicSimulator

        evaluator = DeterministicSimulator()
    else:
        evaluator = simulator
    payloads = [item.model_dump(mode="json") for item in scenario_list]
    first = prebuilt_scenarios[0]
    initial_state, initial_analytics = scenario_view(first.scenario_id, payloads)

    def run_callback(
        scenario_id: str,
        strategy_name: str,
        provider_name: str,
        model: str,
        source: str,
        license_name: str,
        url: str,
        items: Sequence[dict[str, Any]],
    ) -> Iterator[tuple[str, str, str, str]]:
        yield from run_strategy_events(scenario_id, strategy_name, provider_name, model, source, license_name, url, items, evaluator)

    def create_situation_callback(
        situation_name: str,
        season: int,
        week: int,
        offense: str,
        defense: str,
        offense_score: int,
        defense_score: int,
        quarter: int,
        clock: str,
        down: int,
        distance: float,
        field_side: str,
        yard_line: float,
        offense_timeouts: int,
        defense_timeouts: int,
        win_probability: float | None,
        expected_points: float | None,
        editing_scenario_id: str | None,
        items: Sequence[dict[str, Any]],
    ) -> tuple[Any, Any, list[dict[str, Any]], str, str, Any, None, Any, str, str, str, str]:
        try:
            scenario = create_custom_scenario(
                situation_name,
                season,
                week,
                offense,
                defense,
                offense_score,
                defense_score,
                quarter,
                clock,
                down,
                distance,
                field_side,
                yard_line,
                offense_timeouts,
                defense_timeouts,
                win_probability,
                expected_points,
            )
            if custom_scenarios_path is not None:
                save_custom_scenario(custom_scenarios_path, scenario, replacing_scenario_id=editing_scenario_id)
        except Exception as error:
            raise gr.Error(f"The custom situation could not be saved: {error}") from error

        replaced_ids = {scenario.scenario_id}
        if editing_scenario_id:
            replaced_ids.add(editing_scenario_id)
        updated_payloads = [item for item in items if item["scenario_id"] not in replaced_ids]
        updated_payloads.append(scenario.model_dump(mode="json"))
        updated_scenarios = [Scenario.model_validate(item) for item in updated_payloads]
        custom_scenarios = [item for item in updated_scenarios if is_custom_scenario(item)]
        state_html, analytics_html, fresh_status, fresh_transcript, fresh_decision, fresh_grade = scenario_view_with_reset(
            scenario.scenario_id,
            updated_payloads,
        )
        return (
            gr.update(
                choices=[("Pre-Built", "prebuilt"), ("My Situations", "custom")],
                value="custom",
            ),
            gr.update(
                choices=[(item.display_name, item.scenario_id) for item in custom_scenarios],
                value=scenario.scenario_id,
            ),
            updated_payloads,
            state_html,
            analytics_html,
            gr.update(visible=False),
            None,
            gr.update(visible=True),
            fresh_status,
            fresh_transcript,
            fresh_decision,
            fresh_grade,
        )

    def change_library_callback(
        library: str,
        items: Sequence[dict[str, Any]],
    ) -> tuple[Any, Any, str, str, str, str, str, str]:
        available = scenarios_for_library(library, items)
        if not available:
            raise gr.Error("No saved situations are on this call sheet yet. Create one to get started.")
        selected = available[0]
        state_html, analytics_html, fresh_status, fresh_transcript, fresh_decision, fresh_grade = scenario_view_with_reset(
            selected.scenario_id,
            items,
        )
        return (
            gr.update(
                choices=[(item.display_name, item.scenario_id) for item in available],
                value=selected.scenario_id,
            ),
            gr.update(visible=library == "custom"),
            state_html,
            analytics_html,
            fresh_status,
            fresh_transcript,
            fresh_decision,
            fresh_grade,
        )

    def open_new_situation_callback() -> tuple[Any, ...]:
        return (
            gr.update(visible=True),
            None,
            "## Put a Custom Situation on the Call Sheet\n"
            "Enter what the offense sees before the snap. Give it a memorable name so you can find it in My Situations next time. "
            "Optional analytics can be left blank and will be estimated.",
            gr.update(value="Save to My Situations"),
            *_NEW_SITUATION_FORM,
        )

    def open_edit_situation_callback(
        scenario_id: str,
        items: Sequence[dict[str, Any]],
    ) -> tuple[Any, ...]:
        scenario = next((Scenario.model_validate(item) for item in items if item["scenario_id"] == scenario_id), None)
        if scenario is None or not is_custom_scenario(scenario):
            raise gr.Error("Choose a saved custom situation before opening the editor.")
        return (
            gr.update(visible=True),
            scenario.scenario_id,
            "## Edit Saved Situation\nUpdate the pre-snap details below. Saving will replace this entry in My Situations.",
            gr.update(value="Update Situation"),
            *custom_scenario_form_values(scenario),
        )

    def open_delete_confirmation_callback(
        scenario_id: str,
        items: Sequence[dict[str, Any]],
    ) -> tuple[Any, str, str]:
        scenario = next((Scenario.model_validate(item) for item in items if item["scenario_id"] == scenario_id), None)
        if scenario is None or not is_custom_scenario(scenario):
            raise gr.Error("Choose a saved custom situation before deleting it.")
        return (
            gr.update(visible=True),
            scenario.scenario_id,
            f"### Remove {_safe(scenario.name or 'this situation')} from My Situations?\n\n"
            "This permanently removes the saved situation from your local call sheet. This cannot be undone.",
        )

    def delete_situation_callback(
        scenario_id: str,
        items: Sequence[dict[str, Any]],
    ) -> tuple[Any, Any, list[dict[str, Any]], Any, Any, None, str, str, str, str, str, str]:
        selected = next((Scenario.model_validate(item) for item in items if item["scenario_id"] == scenario_id), None)
        if selected is None or not is_custom_scenario(selected):
            raise gr.Error("The selected custom situation is no longer available.")
        try:
            if custom_scenarios_path is not None:
                delete_custom_scenario(custom_scenarios_path, scenario_id)
        except Exception as error:
            raise gr.Error(f"The saved situation could not be deleted: {error}") from error

        updated_payloads = [item for item in items if item["scenario_id"] != scenario_id]
        remaining_custom = scenarios_for_library("custom", updated_payloads)
        if remaining_custom:
            library = "custom"
            available = remaining_custom
            library_choices = [("Pre-Built", "prebuilt"), ("My Situations", "custom")]
        else:
            library = "prebuilt"
            available = scenarios_for_library("prebuilt", updated_payloads)
            library_choices = [("Pre-Built", "prebuilt")]
        next_scenario = available[0]
        state_html, analytics_html, fresh_status, fresh_transcript, fresh_decision, fresh_grade = scenario_view_with_reset(
            next_scenario.scenario_id,
            updated_payloads,
        )
        return (
            gr.update(choices=library_choices, value=library),
            gr.update(choices=[(item.display_name, item.scenario_id) for item in available], value=next_scenario.scenario_id),
            updated_payloads,
            gr.update(visible=library == "custom"),
            gr.update(visible=False),
            None,
            state_html,
            analytics_html,
            fresh_status,
            fresh_transcript,
            fresh_decision,
            fresh_grade,
        )

    with gr.Blocks(title="NFL Virtual Coaching Staff", analytics_enabled=False) as demo:
        session_scenarios = gr.State(payloads)
        editing_scenario_id = gr.State(None)
        deleting_scenario_id = gr.State(None)
        with gr.Row(equal_height=False, elem_id="game-plan-row"):
            with gr.Column(scale=2, min_width=420, elem_id="game-situation-column"):
                gr.Markdown(
                    "# NFL Virtual Coaching Staff\n"
                    "Put the situation on the call sheet, hear every coordinator, and see what the head coach sends in.",
                    elem_id="app-title",
                )
                with gr.Group(elem_id="scenario-picker-card"):
                    with gr.Row(equal_height=True, elem_id="scenario-picker-heading-row"):
                        gr.HTML(
                            '<span class="scenario-picker-label">Game Situation</span>',
                            elem_id="game-situation-heading",
                        )
                        scenario_library = gr.Radio(
                            choices=[("Pre-Built", "prebuilt")] + ([("My Situations", "custom")] if saved_custom_scenarios else []),
                            value="prebuilt",
                            label="Situation Library",
                            show_label=False,
                            container=False,
                            elem_id="scenario-library-selector",
                        )
                    with gr.Row(equal_height=True, elem_id="scenario-picker-row"):
                        scenario_selector = gr.Dropdown(
                            choices=[(item.display_name, item.scenario_id) for item in prebuilt_scenarios],
                            value=first.scenario_id,
                            label="Game Situation",
                            show_label=False,
                            container=False,
                            elem_id="game-situation-selector",
                            scale=5,
                        )
                        open_custom_situation = gr.Button(
                            "＋ New Situation",
                            variant="secondary",
                            elem_id="open-custom-situation",
                            scale=1,
                            min_width=155,
                        )
                    with gr.Row(visible=False, elem_id="custom-situation-management") as custom_situation_management:
                        edit_custom_situation = gr.Button(
                            "✎ Edit Selected",
                            variant="secondary",
                            elem_id="edit-custom-situation",
                        )
                        delete_custom_situation_button = gr.Button(
                            "Delete Selected",
                            variant="stop",
                            elem_id="delete-custom-situation",
                        )
                state_card = gr.HTML(
                    initial_state,
                    elem_id="pre-snap-card",
                    elem_classes="coach-card-component",
                )
            with gr.Column(scale=2, min_width=360, elem_id="sideline-control-column"):
                with gr.Column(elem_id="settings-header-slot"):
                    with gr.Accordion("Sideline Connection & Model Settings", open=False, elem_id="provider-config"):
                        provider = gr.Dropdown(
                            choices=model_provider_choices(),
                            value=ModelProvider.AZURE_FOUNDRY.value,
                            label="Coaching Staff AI Model Provider",
                        )
                        model_name = gr.Textbox(
                            value=os.environ.get("FOUNDRY_MODEL"),
                            label="Model / Deployment for the Call Sheet",
                        )
                        upstream_url = gr.Textbox(
                            label="Model Card / Film Room URL (optional)",
                            placeholder="https://huggingface.co/organization/model",
                        )
                        model_license = gr.Dropdown(
                            ["Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "MPL-2.0"],
                            label="Model License (Ollama Only)",
                        )
                        base_url = gr.Textbox(
                            value=os.environ.get("FOUNDRY_ENDPOINT") or "http://127.0.0.1:11434",
                            label="Sideline Connection / Provider Endpoint",
                            info="Ollama URL or a Foundry endpoint (ending in /openai/v1/). "
                            "Foundry uses Entra ID unless AZURE_FOUNDRY_API_KEY is set.",
                        )
                strategy = gr.Radio(
                    choices=[
                        ("Analytics booth only", "expected_points"),
                        ("Head Coach only", "single_agent"),
                        ("Full Coaching Staff", "multi_agent"),
                    ],
                    value="multi_agent",
                    label="Who's Making the Calls?",
                )
                baseline = gr.HTML(
                    initial_analytics,
                    elem_id="analytics-booth",
                    elem_classes=["coach-card-component", "sideline-analytics"],
                )
                run = gr.Button("🏈 Send in the Call! 🏈", variant="primary", elem_id="send-play-call")
                with gr.Group(elem_id="play-call-status"):
                    status = gr.Markdown(_live_status("Waiting for Coach to put in the call..."), elem_id="live-play-call-status")
        transcript = gr.Markdown(
            _conversation_placeholder(),
            label="Coaches' Headset — Live Conversation",
            elem_id="coaches-meeting-transcript",
        )
        with gr.Row(equal_height=True, elem_id="decision-dashboard"):
            with gr.Column(scale=1, min_width=420, elem_classes="dashboard-column"):
                final_decision = gr.HTML(
                    _decision_placeholder(),
                    elem_id="head-coach-call",
                    elem_classes="coach-card-component",
                )
            with gr.Column(scale=1, min_width=420, elem_classes="dashboard-column"):
                score = gr.HTML(
                    _grade_placeholder(),
                    elem_id="decision-grade",
                    elem_classes="coach-card-component",
                )
        with gr.Group(visible=False, elem_id="custom-situation-modal") as custom_situation_modal:
            with gr.Column(elem_id="custom-situation-dialog"):
                custom_situation_intro = gr.Markdown(
                    "## Put a Custom Situation on the Call Sheet\n"
                    "Enter what the offense sees before the snap. Give it a memorable name so you can find it in My Situations next time. "
                    "Optional analytics can be left blank and will be estimated.",
                    elem_id="custom-situation-intro",
                )
                with gr.Column(elem_id="custom-situation-form-body"):
                    with gr.Row():
                        custom_situation_name = gr.Textbox(
                            value="Two-Minute Decision",
                            label="Situation Name",
                            info="For example: Goal-Line Stand, Four-Minute Offense, or Must-Have Fourth Down",
                        )
                    with gr.Row():
                        custom_season = gr.Number(value=2025, label="Season", precision=0, minimum=2000, maximum=2100)
                        custom_week = gr.Number(value=1, label="Week", precision=0, minimum=1, maximum=22)
                    with gr.Row():
                        custom_offense = gr.Textbox(value="BUF", label="Offense", info="2–4 character team abbreviation")
                        custom_defense = gr.Textbox(value="KC", label="Defense", info="2–4 character team abbreviation")
                    with gr.Row():
                        custom_offense_score = gr.Number(value=20, label="Offense Score", precision=0, minimum=0)
                        custom_defense_score = gr.Number(value=20, label="Defense Score", precision=0, minimum=0)
                    with gr.Row():
                        custom_quarter = gr.Dropdown([1, 2, 3, 4], value=4, label="Quarter")
                        custom_clock = gr.Textbox(value="2:00", label="Game Clock", info="Use MM:SS, from 0:00 through 15:00")
                    with gr.Row():
                        custom_down = gr.Dropdown([1, 2, 3, 4], value=4, label="Down")
                        custom_distance = gr.Number(value=2, label="Yards to Go", minimum=0.1, maximum=99)
                    with gr.Row():
                        custom_field_side = gr.Dropdown(
                            [
                                ("Offense's side of midfield", "offense"),
                                ("The 50-yard line", "midfield"),
                                ("Defense's side of midfield", "defense"),
                            ],
                            value="defense",
                            label="Field Side",
                        )
                        custom_yard_line = gr.Number(
                            value=35,
                            label="Yard Line",
                            precision=0,
                            minimum=1,
                            maximum=49,
                            info="Ignored when the ball is at midfield",
                        )
                    with gr.Row():
                        custom_offense_timeouts = gr.Number(value=2, label="Offense Timeouts", precision=0, minimum=0, maximum=3)
                        custom_defense_timeouts = gr.Number(value=2, label="Defense Timeouts", precision=0, minimum=0, maximum=3)
                    with gr.Accordion("Optional Analytics Overrides", open=False, elem_id="custom-analytics-overrides"):
                        with gr.Row():
                            custom_win_probability = gr.Number(
                                value=None,
                                label="Offense Win Probability (%)",
                                minimum=0,
                                maximum=100,
                                info="Leave blank to estimate from score, clock, and field position",
                            )
                            custom_expected_points = gr.Number(
                                value=None,
                                label="Current Expected Points",
                                info="Leave blank to estimate from down, distance, and field position",
                            )
                with gr.Row(elem_id="custom-situation-actions"):
                    cancel_custom_situation = gr.Button("Cancel", variant="secondary")
                    save_custom_situation = gr.Button("Save to My Situations", variant="primary")

        with gr.Group(visible=False, elem_id="delete-situation-modal") as delete_situation_modal:
            with gr.Column(elem_id="delete-situation-dialog"):
                delete_situation_message = gr.Markdown(elem_id="delete-situation-message")
                with gr.Row(elem_id="delete-situation-actions"):
                    cancel_delete_situation = gr.Button("Keep Situation", variant="secondary")
                    confirm_delete_situation = gr.Button(
                        "Delete Permanently",
                        variant="stop",
                        elem_id="confirm-delete-situation",
                    )

        custom_form_fields = [
            custom_situation_name,
            custom_season,
            custom_week,
            custom_offense,
            custom_defense,
            custom_offense_score,
            custom_defense_score,
            custom_quarter,
            custom_clock,
            custom_down,
            custom_distance,
            custom_field_side,
            custom_yard_line,
            custom_offense_timeouts,
            custom_defense_timeouts,
            custom_win_probability,
            custom_expected_points,
        ]
        custom_editor_outputs = [
            custom_situation_modal,
            editing_scenario_id,
            custom_situation_intro,
            save_custom_situation,
            *custom_form_fields,
        ]

        scenario_library.change(
            change_library_callback,
            inputs=[scenario_library, session_scenarios],
            outputs=[scenario_selector, custom_situation_management, state_card, baseline, status, transcript, final_decision, score],
            queue=False,
        )

        scenario_selector.change(
            scenario_view_with_reset,
            inputs=[scenario_selector, session_scenarios],
            outputs=[state_card, baseline, status, transcript, final_decision, score],
            queue=False,
        )
        open_custom_situation.click(
            open_new_situation_callback,
            outputs=custom_editor_outputs,
            queue=False,
        )
        edit_custom_situation.click(
            open_edit_situation_callback,
            inputs=[scenario_selector, session_scenarios],
            outputs=custom_editor_outputs,
            queue=False,
        )
        delete_custom_situation_button.click(
            open_delete_confirmation_callback,
            inputs=[scenario_selector, session_scenarios],
            outputs=[delete_situation_modal, deleting_scenario_id, delete_situation_message],
            queue=False,
        )
        cancel_custom_situation.click(
            lambda: (gr.update(visible=False), None),
            outputs=[custom_situation_modal, editing_scenario_id],
            queue=False,
        )
        cancel_delete_situation.click(
            lambda: (gr.update(visible=False), None),
            outputs=[delete_situation_modal, deleting_scenario_id],
            queue=False,
        )
        confirm_delete_situation.click(
            delete_situation_callback,
            inputs=[deleting_scenario_id, session_scenarios],
            outputs=[
                scenario_library,
                scenario_selector,
                session_scenarios,
                custom_situation_management,
                delete_situation_modal,
                deleting_scenario_id,
                state_card,
                baseline,
                status,
                transcript,
                final_decision,
                score,
            ],
            queue=False,
        )
        save_custom_situation.click(
            create_situation_callback,
            inputs=[
                custom_situation_name,
                custom_season,
                custom_week,
                custom_offense,
                custom_defense,
                custom_offense_score,
                custom_defense_score,
                custom_quarter,
                custom_clock,
                custom_down,
                custom_distance,
                custom_field_side,
                custom_yard_line,
                custom_offense_timeouts,
                custom_defense_timeouts,
                custom_win_probability,
                custom_expected_points,
                editing_scenario_id,
                session_scenarios,
            ],
            outputs=[
                scenario_library,
                scenario_selector,
                session_scenarios,
                state_card,
                baseline,
                custom_situation_modal,
                editing_scenario_id,
                custom_situation_management,
                status,
                transcript,
                final_decision,
                score,
            ],
            queue=False,
        )
        run.click(
            run_callback,
            inputs=[
                scenario_selector,
                strategy,
                provider,
                model_name,
                upstream_url,
                model_license,
                base_url,
                session_scenarios,
            ],
            outputs=[status, transcript, final_decision, score],
            concurrency_limit=1,
            trigger_mode="once",
        )
    return demo.queue(default_concurrency_limit=1, max_size=16)

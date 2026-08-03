"""Deterministic pre-snap football context for models and user-facing analysis."""

from __future__ import annotations
import math
from typing import Literal
from nfl_coaching_sim.models import EvidenceItem, Scenario, SituationBrief


def _distance_bucket(yards_to_go: float) -> Literal["short", "manageable", "medium", "long"]:
    if yards_to_go <= 1:
        return "short"
    if yards_to_go <= 3:
        return "manageable"
    if yards_to_go <= 6:
        return "medium"
    return "long"


def _field_zone(yardline_100: float) -> str:
    if yardline_100 <= 10:
        return "goal-to-go area"
    if yardline_100 <= 20:
        return "red zone"
    if yardline_100 <= 40:
        return "scoring territory"
    if yardline_100 <= 50:
        return "plus territory"
    return "own territory"


def _score_context(score_differential: int) -> str:
    if score_differential > 0:
        return f"leading by {score_differential}"
    if score_differential < 0:
        return f"trailing by {abs(score_differential)}"
    return "tied"


def _field_goal_effect(score_differential: int) -> str:
    post_kick = score_differential + 3
    if post_kick == 0:
        return "A made field goal would tie the game."
    if score_differential <= 0 < post_kick:
        return f"A made field goal would give a {post_kick}-point lead."
    if post_kick < 0:
        return f"A made field goal would still leave the offense trailing by {abs(post_kick)}."
    return f"A made field goal would give a {post_kick}-point lead."


def build_situation_brief(scenario: Scenario) -> SituationBrief:
    """Derive an outcome-free football briefing from the released pre-snap state."""

    state = scenario.state
    differential = state.score_differential
    clock_priority = "preserve" if differential < 0 else "drain" if differential > 0 else "balanced"
    first_down_clock_window = max(0, 3 - state.defensive_timeouts) * 40
    first_down_can_end_game = (
        state.quarter == 4 and differential > 0 and first_down_clock_window > 0 and state.game_seconds_remaining <= first_down_clock_window
    )
    approximate_field_goal_yards = int(round(state.yardline_100 + 17))
    points_to_tie = max(0, -differential)
    minimum_scoring_possessions = math.ceil(points_to_tie / 8) if points_to_tie else 0
    two_minute_warning_pending = state.quarter < 4 or (state.quarter == 4 and state.game_seconds_remaining > 120)
    score_context = _score_context(differential)
    distance_bucket = _distance_bucket(state.yards_to_go)
    field_zone = _field_zone(state.yardline_100)
    field_goal_effect = _field_goal_effect(differential)

    evidence = [
        EvidenceItem(
            evidence_id="STATE_DOWN_DISTANCE",
            category="state",
            statement=f"The offense faces {state.down_and_distance}; this is a {distance_bucket} distance bucket.",
        ),
        EvidenceItem(
            evidence_id="STATE_SCORE_CONTEXT",
            category="score",
            statement=f"The possession team is {score_context}.",
        ),
        EvidenceItem(
            evidence_id="STATE_CLOCK_CONTEXT",
            category="clock",
            statement=(
                f"The game clock shows {state.clock_display}. The offense has {state.possession_timeouts} timeout(s), "
                f"and the defense has {state.defensive_timeouts}."
            ),
        ),
        EvidenceItem(
            evidence_id="CLOCK_PRIORITY",
            category="clock",
            statement=f"The score state makes {clock_priority} the primary clock objective.",
        ),
        EvidenceItem(
            evidence_id="TWO_MINUTE_WARNING",
            category="clock",
            statement=(
                "The fourth-quarter two-minute warning remains later in regulation, but it is not imminent."
                if state.quarter < 4
                else (
                    "The fourth-quarter two-minute warning is still pending."
                    if two_minute_warning_pending
                    else "The fourth-quarter two-minute warning has already occurred."
                )
            ),
        ),
        EvidenceItem(
            evidence_id="FIRST_DOWN_ENDGAME_LEVERAGE",
            category="clock",
            statement=(
                "An in-bounds first down can exhaust the remaining regulation clock against the available defensive timeouts."
                if first_down_can_end_game
                else (
                    (
                        f"A new series can consume about {first_down_clock_window} seconds of play-clock runoff before accounting for "
                        "the snap and play itself; a first down does not automatically end the game."
                    )
                    if first_down_clock_window
                    else (
                        "With all three defensive timeouts available, the defense can stop the normal runoff after each of the first "
                        "three plays of a new series; a first down does not automatically end the game."
                    )
                )
            ),
        ),
        EvidenceItem(
            evidence_id="FIELD_POSITION_ZONE",
            category="field_position",
            statement=f"The ball is at {state.field_position}, categorized as {field_zone}.",
        ),
        EvidenceItem(
            evidence_id="FIELD_GOAL_DISTANCE",
            category="field_position",
            statement=f"A field-goal attempt from this spot would be approximately {approximate_field_goal_yards} yards.",
        ),
        EvidenceItem(
            evidence_id="FIELD_GOAL_SCORE_EFFECT",
            category="score",
            statement=field_goal_effect,
        ),
        EvidenceItem(
            evidence_id="MINIMUM_SCORING_POSSESSIONS",
            category="score",
            statement=(
                f"The offense needs at least {minimum_scoring_possessions} scoring possession(s) to tie, assuming up to eight points per "
                f"possession."
                if minimum_scoring_possessions
                else "The offense does not currently need a scoring possession to draw level."
            ),
        ),
        EvidenceItem(
            evidence_id="CURRENT_WIN_PROBABILITY",
            category="baseline",
            statement=f"The released pre-snap offense win probability is {state.win_probability:.1%}.",
        ),
        EvidenceItem(
            evidence_id="CURRENT_EXPECTED_POINTS",
            category="baseline",
            statement=f"The released pre-snap expected-points value is {state.expected_points:+.3f}.",
        ),
    ]
    evidence.extend(
        EvidenceItem(
            evidence_id=f"EP_BASELINE_{action.value.upper()}",
            category="baseline",
            statement=f"The released simple EP baseline estimates {scenario.ep_baseline[action]:+.3f} EPA for {action.football_label.lower()}.",
        )
        for action in state.legal_actions
    )
    return SituationBrief(
        score_context=score_context,
        clock_priority=clock_priority,
        field_zone=field_zone,
        distance_bucket=distance_bucket,
        approximate_field_goal_yards=approximate_field_goal_yards,
        field_goal_score_effect=field_goal_effect,
        minimum_scoring_possessions_to_tie=minimum_scoring_possessions,
        two_minute_warning_pending=two_minute_warning_pending,
        first_down_can_end_game=first_down_can_end_game,
        first_down_clock_window_seconds=first_down_clock_window,
        evidence=evidence,
    )

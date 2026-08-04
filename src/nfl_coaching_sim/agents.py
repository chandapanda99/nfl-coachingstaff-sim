"""Role-based deliberation strategies and provider-neutral LangChain integration."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from nfl_coaching_sim.football import build_situation_brief
from nfl_coaching_sim.models import (
    Action,
    ActionAssessment,
    DebateTranscript,
    Decision,
    DecisionTrace,
    Recommendation,
    RevisedRecommendation,
    Scenario,
    SituationBrief,
    StageEvent,
    action_vote,
)
from nfl_coaching_sim.providers import (
    ModelConfiguration,
    ModelProvider,
    ProviderCapabilities,
    get_provider,
    model_provider_choices,
    register_model_provider,
    register_provider,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)
_EVIDENCE_ID_ALIASES = {
    "CLOCK_CONTEXT": "STATE_CLOCK_CONTEXT",
}


@dataclass(frozen=True)
class RoleProfile:
    title: str
    mission: str
    voice: str
    checklist: tuple[str, ...]
    guardrail: str


ROLE_PROFILES = {
    "offensive_coordinator": RoleProfile(
        title="Offensive Coordinator",
        mission="Build the most executable call plan for the offense in this down, distance, field position, score, and clock state.",
        voice=(
            "Sound decisive and economical, like an offensive coordinator on the headset. Use natural phrases such as "
            "'I like', 'stay on schedule', 'protect it', 'alert', and 'check with me' when they fit; do not force catchphrases."
        ),
        checklist=(
            "Compare the execution burden and required yardage for every legal call.",
            "Account for protection, sack, turnover, negative-play, and incompletion risk without inventing personnel details.",
            "Explain how success and failure affect the next down or the remaining possession.",
            "Treat fronts, box counts, coverage shells, and player matchups as unknown unless supplied.",
        ),
        guardrail="Discuss unavailable defensive looks only as explicit if/then pre-snap checks, never as observed facts.",
    ),
    "defensive_coordinator": RoleProfile(
        title="Defensive Coordinator",
        mission="Think like the opposing defensive play caller and identify which offensive choice best attacks the defense's situational objective.",
        voice=(
            "Talk like the defensive coordinator advising the offense: identify what the defense must take away, where pressure can come from, "
            "and which call makes the defense defend the most grass. Keep any front or coverage discussion conditional."
        ),
        checklist=(
            "State what the defense must prevent given the sticks, score, field position, and clock.",
            "Compare how every legal call can be defeated by pressure, coverage, or numbers at the point of attack.",
            "Use conditional counters for light boxes, loaded boxes, pressure looks, and two-high shells.",
            "Identify which offensive call gives the defense the easiest clock or field-position outcome.",
        ),
        guardrail="The actual front, coverage, pressure, box count, personnel, and matchup quality are unavailable; never claim to have observed them.",
    ),
    "analytics_assistant": RoleProfile(
        title="Analytics Assistant",
        mission="Translate the released evidence into a calibrated comparison of every legal action while keeping the hidden evaluator private.",
        voice=(
            "Be the concise booth voice in the coach's ear. Translate the numbers into football consequences; say 'the numbers lean' or "
            "'the gap is small' instead of reading database fields or sounding like a report."
        ),
        checklist=(
            "Compare the simple EPA baseline for every legal call, not only the leader.",
            "Separate expected-points value from win-probability, clock, and possession effects.",
            "Call out small numerical differences that should not be treated as decisive.",
            "Use only supplied evidence identifiers and never manufacture rates, probabilities, or sample sizes.",
        ),
        guardrail="The simple EP table is evidence, not an oracle; do not claim access to richer simulator estimates.",
    ),
    "clock_management_specialist": RoleProfile(
        title="Clock Management",
        mission="Protect the team's possession and clock objectives while accounting for both teams' timeout leverage.",
        voice=(
            "Speak urgently and concretely about preserving, bleeding, or trading clock. Use sideline language such as 'bank the timeout', "
            "'the clock is the opponent', and 'we still need a possession' only when the game state supports it."
        ),
        checklist=(
            "Start with whether the offense should preserve, drain, or balance the clock.",
            "Compare likely clock-stop and runoff consequences for every legal action.",
            "Use the two-minute-warning and first-down endgame evidence explicitly.",
            "Explain how success and failure change the number and quality of remaining possessions.",
        ),
        guardrail="Do not assert an exact runoff or future possession count unless the supplied evidence states it.",
    ),
    "critical_reviewer": RoleProfile(
        title="Quality Control Coach",
        mission="Audit the staff's football logic and choose the call that survives the strongest factual and situational objections.",
        voice=(
            "Be the staff's respectful skeptic. Open naturally with language such as 'hold on', 'the problem with that call', or "
            "'make sure we're not assuming' when appropriate, then identify the call that survives the objection."
        ),
        checklist=(
            "Check that every legal action received a fair comparison.",
            "Reject unsupported claims about formations, personnel, weather, tendencies, coverage, or player ability.",
            "Check score arithmetic, field-goal distance, timeout usage, and clock logic against the evidence packet.",
            "Identify the closest alternative and the concrete pre-snap condition that would justify switching.",
        ),
        guardrail="Prioritize factual consistency and decision robustness over inventing a new tactical narrative.",
    ),
}


class StructuredModel(Protocol):
    model_id: str

    def invoke(self, schema: type[SchemaT], system_prompt: str, user_prompt: str) -> SchemaT: ...


class LangChainStructuredModel:
    """Provider-neutral structured-output adapter used by every coaching strategy."""

    def __init__(
        self,
        configuration: ModelConfiguration,
        chat_model: Any,
        authentication: str,
        capabilities: ProviderCapabilities,
        effective_generation_parameters: dict[str, Any],
    ) -> None:
        self.configuration = configuration
        self.model_id = configuration.model_id
        self.authentication = authentication
        self.capabilities = capabilities
        self.effective_generation_parameters = effective_generation_parameters
        self._model = chat_model

    def invoke(self, schema: type[SchemaT], system_prompt: str, user_prompt: str) -> SchemaT:
        return _invoke_with_repair(self._model, schema, system_prompt, user_prompt)


def _invoke_with_repair(model: Any, schema: type[SchemaT], system_prompt: str, user_prompt: str) -> SchemaT:
    """Apply the same schema-repair policy to every LangChain provider."""

    last_error: Exception | None = None
    repair = ""
    for attempt in range(3):
        try:
            structured = model.with_structured_output(schema)
            result = structured.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt + repair),
                ]
            )
            return schema.model_validate(result)
        except Exception as error:
            if not _is_repairable_output_error(error):
                raise
            last_error = error
            repair = (
                "\n\nYour previous response was invalid. Return only content that matches the required schema. "
                f"Repair attempt {attempt + 1}."
            )
    raise RuntimeError(f"structured model output failed: {last_error}") from last_error


def _is_repairable_output_error(error: Exception) -> bool:
    """Retry malformed model output without replaying transport or authentication failures."""

    return isinstance(error, (ValidationError, json.JSONDecodeError, ValueError)) or type(error).__name__ in {
        "OutputParserException",
        "StructuredOutputError",
    }


def _scenario_prompt(scenario: Scenario) -> str:
    state = scenario.state
    brief = build_situation_brief(scenario)
    baseline = {action.value: round(value, 4) for action, value in scenario.ep_baseline.items()}
    return json.dumps(
        {
            "scenario_id": scenario.scenario_id,
            "game_state": {
                "clock": state.clock_display,
                "game_seconds_remaining": state.game_seconds_remaining,
                "possession": state.possession_team,
                "defense": state.defensive_team,
                "score": {
                    state.possession_team: state.possession_score,
                    state.defensive_team: state.defensive_score,
                },
                "down": state.down,
                "yards_to_go": state.yards_to_go,
                "yardline_100": state.yardline_100,
                "timeouts": {
                    state.possession_team: state.possession_timeouts,
                    state.defensive_team: state.defensive_timeouts,
                },
                "current_win_probability": state.win_probability,
                "current_expected_points": state.expected_points,
            },
            "legal_actions": [action.value for action in state.legal_actions],
            "simple_ep_baseline": baseline,
            "situation_brief": brief.model_dump(mode="json", exclude={"evidence"}),
            "allowed_evidence_ids": sorted(brief.evidence_ids),
            "evidence_packet": [item.model_dump(mode="json") for item in brief.evidence],
        },
        indent=2,
    )


def _role_system_prompt(role: str, phase: str) -> str:
    profile = ROLE_PROFILES[role]
    checklist = "\n".join(f"{index}. {item}" for index, item in enumerate(profile.checklist, start=1))
    phase_instruction = (
        "Make an independent recommendation before seeing another coach's preference."
        if phase == "opening"
        else "Audit the anonymized opening calls, rebut the weakest material claim, and then keep or revise your recommendation."
    )
    return (
        f"You are an NFL team's {profile.title}. {profile.mission}\n\n"
        f"HEADSET VOICE: {profile.voice}\n\n"
        f"COACHING CHECKLIST:\n{checklist}\n\n"
        f"GUARDRAIL: {profile.guardrail}\n\n"
        f"{phase_instruction} Write the argument and rebuttal as part of a brief, natural exchange among an NFL coaching staff—not an essay or data report. "
        "Use football language, short sentences, and directly reference the supplied clock, score, sticks, field position, timeouts, and EP values "
        "when they matter. Assess every legal action exactly once in the supplied order. For each action, provide advantages, risks, clock effect, "
        "a 0-to-1 support score, and only evidence IDs from allowed_evidence_ids. Evidence IDs are private validation tags: copy them verbatim only "
        "inside the evidence_ids arrays. Never say or embed an evidence ID in the argument, rebuttal, advantages, risks, clock effect, concerns, "
        "rationale, or switch condition. "
        "Choose one legal call, name a different legal action as the closest alternative, and state a concrete condition that would switch the call. "
        "Separate supplied facts from conditional football judgment. If going for it, specify run or pass."
    )


def _validate_recommendation(recommendation: Recommendation, scenario: Scenario, brief: SituationBrief) -> Recommendation:
    normalized_assessments = [
        assessment.model_copy(
            update={"evidence_ids": [_EVIDENCE_ID_ALIASES.get(evidence_id, evidence_id) for evidence_id in assessment.evidence_ids]}
        )
        for assessment in recommendation.action_assessments
    ]
    recommendation = recommendation.model_copy(update={"action_assessments": normalized_assessments})
    recommendation.decision.validate_for(scenario.state)
    legal_order = scenario.state.legal_actions
    assessed_order = tuple(assessment.action for assessment in recommendation.action_assessments)
    if assessed_order != legal_order:
        raise ValueError(
            "action assessments must cover every legal action in supplied order; "
            f"expected={[action.value for action in legal_order]}, actual={[action.value for action in assessed_order]}"
        )
    if recommendation.closest_alternative not in set(legal_order):
        raise ValueError("closest alternative must be legal for the supplied game state")
    cited = {evidence_id for assessment in recommendation.action_assessments for evidence_id in assessment.evidence_ids}
    unknown = cited - brief.evidence_ids
    if unknown:
        raise ValueError(f"recommendation cited unknown evidence IDs: {sorted(unknown)}")
    return recommendation


def _fallback_assessments(scenario: Scenario) -> list[ActionAssessment]:
    best_value = max(scenario.ep_baseline[action] for action in scenario.state.legal_actions)
    return [
        ActionAssessment(
            action=action,
            advantages=[f"The released EP baseline estimates {scenario.ep_baseline[action]:+.3f} EPA."],
            risks=["No role-specific model assessment reached the sideline."],
            clock_effect="Use the deterministic situation brief for clock context; no additional model judgment is available.",
            evidence_ids=[f"EP_BASELINE_{action.value.upper()}", "CLOCK_PRIORITY"],
            support_score=1.0 if scenario.ep_baseline[action] == best_value else 0.0,
        )
        for action in scenario.state.legal_actions
    ]


def _ep_decision(scenario: Scenario, rationale: str) -> Decision:
    action = max(scenario.state.legal_actions, key=lambda candidate: (scenario.ep_baseline[candidate], candidate.value))
    return Decision(
        action=action,
        go_for_it_play=Action.PASS if action == Action.GO_FOR_IT else None,
        rationale=rationale,
    )


def _fallback_recommendation(role: str, scenario: Scenario, error: Exception) -> Recommendation:
    decision = _ep_decision(scenario, "The coordinator's headset went down, so the analytics booth sent in the highest-EPA call")
    alternatives = [action for action in scenario.state.legal_actions if action != decision.action]
    closest_alternative = max(alternatives, key=lambda action: (scenario.ep_baseline[action], action.value))
    return Recommendation(
        role=role,
        decision=decision,
        confidence=0,
        argument="No clean recommendation came through from this position group; defer to the analytics card",
        concerns=[f"Headset communication: {str(error)[:280]}"],
        action_assessments=_fallback_assessments(scenario),
        closest_alternative=closest_alternative,
        switch_condition="Switch only if verified pre-snap information materially changes the released evidence comparison",
    )


class ExpectedPointsStrategy:
    name = "expected_points"

    def decide(self, scenario: Scenario) -> DecisionTrace:
        started = time.perf_counter()
        return DecisionTrace(
            strategy=self.name,
            decision=_ep_decision(scenario, "The analytics booth sends in the legal call with the best expected points added"),
            latency_seconds=time.perf_counter() - started,
            model_calls=0,
        )


class SingleAgentStrategy:
    name = "single_agent"

    def __init__(self, model: StructuredModel) -> None:
        self.model = model

    def decide(self, scenario: Scenario) -> DecisionTrace:
        started = time.perf_counter()
        failures: list[str] = []
        fallback = False
        try:
            decision = self.model.invoke(
                Decision,
                (
                    "You are the NFL head coach making a live sideline decision. Compare every legal action using the supplied game state, deterministic situation brief, "
                    "and evidence packet before making one legal decision. Balance win probability, expected points, clock, possession value, "
                    "execution risk, and score. Treat formations, personnel, matchups, weather, fronts, coverages, and tendencies as unknown. "
                    "Never invent unavailable facts or claim access to the hidden evaluator. Give a short, decisive football rationale that could "
                    "actually be said over a headset. Evidence IDs are private metadata and must never appear in the rationale."
                ),
                _scenario_prompt(scenario),
            )
            decision.validate_for(scenario.state)
        except Exception as error:
            failures.append(f"head_coach: {error}")
            fallback = True
            decision = _ep_decision(
                scenario,
                "The head coach did not get a clean call through before the play clock, so the analytics card takes over.",
            )
        return DecisionTrace(
            strategy=self.name,
            decision=decision,
            model_id=self.model.model_id,
            latency_seconds=time.perf_counter() - started,
            model_calls=1,
            failures=failures,
            fallback_used=fallback,
        )


class MultiAgentStrategy:
    name = "multi_agent"

    def __init__(self, model: StructuredModel, max_parallel_calls: int = 5) -> None:
        if not 1 <= max_parallel_calls <= len(ROLE_PROFILES):
            raise ValueError(f"max_parallel_calls must be between 1 and {len(ROLE_PROFILES)}")
        self.model = model
        self.max_parallel_calls = max_parallel_calls

    def _initial(self, role: str, scenario: Scenario) -> Recommendation:
        brief = build_situation_brief(scenario)
        result = self.model.invoke(
            Recommendation,
            _role_system_prompt(role, "opening"),
            _scenario_prompt(scenario),
        )
        result = result.model_copy(update={"role": role})
        return _validate_recommendation(result, scenario, brief)

    def _revise(self, role: str, scenario: Scenario, initial: list[Recommendation]) -> RevisedRecommendation:
        arguments = [
            {
                "speaker": f"staff_member_{index + 1}",
                **rec.model_dump(mode="json", exclude={"role"}),
            }
            for index, rec in enumerate(initial)
        ]
        result = self.model.invoke(
            RevisedRecommendation,
            _role_system_prompt(role, "revision"),
            _scenario_prompt(scenario) + "\n\nANONYMIZED INITIAL RECOMMENDATIONS:\n" + json.dumps(arguments, indent=2),
        )
        result = result.model_copy(update={"role": role})
        return RevisedRecommendation.model_validate(
            _validate_recommendation(result, scenario, build_situation_brief(scenario)).model_dump()
        )

    def iter_decide(self, scenario: Scenario) -> Iterator[StageEvent]:
        started = time.perf_counter()
        failures: list[str] = []
        role_order = tuple(ROLE_PROFILES)
        calls = len(role_order)
        initial_by_role: dict[str, Recommendation] = {}
        initial_errors: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel_calls, thread_name_prefix="opening-staff-call") as executor:
            future_roles = {executor.submit(self._initial, role, scenario): role for role in role_order}
            for future in as_completed(future_roles):
                role = future_roles[future]
                try:
                    recommendation = future.result()
                except Exception as error:
                    initial_errors[role] = error
                    recommendation = _fallback_recommendation(role, scenario, error)
                initial_by_role[role] = recommendation
                yield StageEvent(
                    stage=f"recommendation:{role}",
                    message=f"{role.replace('_', ' ').title()} sends in " f"{recommendation.decision.call_label}",
                    role=role,
                    recommendation=recommendation,
                    failure=f"{role} initial: {initial_errors[role]}" if role in initial_errors else None,
                )
        for role in role_order:
            if role in initial_errors:
                failures.append(f"{role} initial: {initial_errors[role]}")
        initial = [initial_by_role[role] for role in role_order]
        yield StageEvent(stage="recommendations", message="Coordinators checking the front, the clock, and the call sheet...")

        calls += len(role_order)
        revised_by_role: dict[str, RevisedRecommendation] = {}
        revision_errors: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel_calls, thread_name_prefix="challenge-round-call") as executor:
            future_roles = {executor.submit(self._revise, role, scenario, initial): role for role in role_order}
            for future in as_completed(future_roles):
                role = future_roles[future]
                try:
                    recommendation = future.result()
                except Exception as error:
                    revision_errors[role] = error
                    fallback = _fallback_recommendation(role, scenario, error)
                    recommendation = RevisedRecommendation(
                        **fallback.model_dump(),
                        rebuttal="No adjustments came through on the headset.",
                    )
                revised_by_role[role] = recommendation
                yield StageEvent(
                    stage=f"revision:{role}",
                    message=f"{role.replace('_', ' ').title()} finishes the discussion " f"with {recommendation.decision.call_label}!",
                    role=role,
                    revision=recommendation,
                    failure=f"{role} revision: {revision_errors[role]}" if role in revision_errors else None,
                )
        for role in role_order:
            if role in revision_errors:
                failures.append(f"{role} revision: {revision_errors[role]}")
        revised = [revised_by_role[role] for role in role_order]
        yield StageEvent(
            stage="debate",
            message="The staff has challenged tendencies, clock math, and situational risk; the adjustments are in...",
        )

        calls += 1
        fallback_used = False
        try:
            decision = self.model.invoke(
                Decision,
                (
                    "You are the head coach breaking the huddle after a live sideline discussion. Synthesize the revised staff debate and choose "
                    "one legal action. Resolve disagreement explicitly, prioritize winning "
                    "the game over raw expected points, verify claims against the evidence packet, "
                    "and explain why the closest competing action lost. Use no outside facts or hidden evaluator values. Give a short, decisive "
                    "football rationale suitable for the headset. Evidence IDs are private metadata and must never appear in the rationale."
                ),
                _scenario_prompt(scenario)
                + "\n\nREVISED STAFF DEBATE:\n"
                + json.dumps([item.model_dump(mode="json") for item in revised], indent=2),
            )
            decision.validate_for(scenario.state)
        except Exception as error:
            failures.append(f"head_coach: {error}")
            decision = action_vote(revised, scenario)
            fallback_used = True

        transcript = DebateTranscript(
            initial=initial,
            revised=revised,
            head_coach=decision,
            failures=failures,
            fallback_used=fallback_used,
        )
        trace = DecisionTrace(
            strategy=self.name,
            decision=decision,
            transcript=transcript,
            model_id=self.model.model_id,
            latency_seconds=time.perf_counter() - started,
            model_calls=calls,
            failures=failures,
            fallback_used=fallback_used,
        )
        yield StageEvent(
            stage="decision",
            message=f"The head coach sends in: {decision.call_label}.",
            trace=trace,
        )

    def decide(self, scenario: Scenario) -> DecisionTrace:
        final: DecisionTrace | None = None
        for event in self.iter_decide(scenario):
            if event.trace is not None:
                final = event.trace
        if final is None:  # defensive invariant
            raise RuntimeError("The coaches' meeting ended without a legal call reaching the sideline!")
        return final


def make_model(configuration: ModelConfiguration) -> StructuredModel:
    adapter = get_provider(configuration.provider)
    provider_model = adapter.build(configuration)
    return LangChainStructuredModel(
        configuration,
        provider_model.chat_model,
        provider_model.authentication,
        adapter.capabilities,
        provider_model.effective_generation_parameters,
    )

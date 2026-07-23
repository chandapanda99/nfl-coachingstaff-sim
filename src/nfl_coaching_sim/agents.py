"""Role-based deliberation strategies and LangChain/Ollama integration."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from enum import StrEnum
from typing import Any, Protocol, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, Field, HttpUrl, model_validator

from nfl_coaching_sim.models import (
    Action,
    DebateTranscript,
    Decision,
    DecisionTrace,
    PROMPT_VERSION,
    Recommendation,
    RevisedRecommendation,
    Scenario,
    StageEvent,
    action_vote,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

ROLES = {
    "offensive_coordinator": "Assess execution, down-and-distance, likely defensive response, and the run/pass mechanics of the call.",
    "defensive_coordinator": (
        "Think like the opponent: identify the coverage, pressure, box count, and counter-strategy the offense is likely to face."),
    "analytics_assistant": (
        "Use the provided expected-points evidence, win-probability context, field position, and uncertainty. Do not invent statistics."
    ),
    "clock_management_specialist": "Prioritize possession value, remaining clock, timeouts, clock runoff, and the score state.",
    "critical_reviewer": (
        "Stress-test assumptions, surface tail risks, and prefer the most defensible choice after challenging the other perspectives."
    ),
}

APPROVED_LICENSES = {
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "MPL-2.0",
}


class ModelProvider(StrEnum):
    OLLAMA = "ollama"
    AZURE_FOUNDRY = "azure_foundry"


class ModelConfiguration(BaseModel):
    provider: ModelProvider = ModelProvider.OLLAMA
    model: str = Field(min_length=1)
    base_url: str = "http://127.0.0.1:11434"
    upstream_url: HttpUrl | None = None
    license: str
    temperature: float = Field(default=0.0, ge=0, le=2)
    seed: int = 2026

    @model_validator(mode="after")
    def validate_open_model(self) -> ModelConfiguration:
        if self.license not in APPROVED_LICENSES:
            raise ValueError(f"model license must be one of {sorted(APPROVED_LICENSES)}")
        endpoint = urlparse(self.base_url)
        if not endpoint.scheme or not endpoint.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) endpoint")
        if self.provider == ModelProvider.AZURE_FOUNDRY:
            if endpoint.scheme != "https":
                raise ValueError("Azure Foundry endpoints must use HTTPS")
            if not endpoint.path.rstrip("/").endswith("/openai/v1"):
                raise ValueError("Azure Foundry base_url must end with /openai/v1/")
        return self

    @property
    def model_id(self) -> str:
        return f"{self.provider.value}:{self.model}"


class StructuredModel(Protocol):
    model_id: str

    def invoke(self, schema: type[SchemaT], system_prompt: str, user_prompt: str) -> SchemaT: ...


class LangChainOllamaModel:
    """Small adapter that keeps LangChain out of the research core."""

    def __init__(self, configuration: ModelConfiguration) -> None:
        from langchain_ollama import ChatOllama

        self.configuration = configuration
        self.model_id = configuration.model_id
        self._model = ChatOllama(
            model=configuration.model,
            base_url=configuration.base_url,
            temperature=configuration.temperature,
            seed=configuration.seed,
            validate_model_on_init=True,
        )

    def invoke(self, schema: type[SchemaT], system_prompt: str, user_prompt: str) -> SchemaT:
        return _invoke_with_repair(self._model, schema, system_prompt, user_prompt)


class AzureFoundryStructuredModel:
    """LangChain adapter for Foundry model deployments using secretless auth."""

    def __init__(self, configuration: ModelConfiguration) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise RuntimeError("Azure Foundry support requires the 'azure' optional dependency: pip install -e '.[azure]'") from error

        api_key = os.environ.get("AZURE_FOUNDRY_API_KEY")
        if api_key:
            credential: Any = api_key
            self.authentication = "api_key_environment"
        else:
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            except ImportError as error:
                raise RuntimeError(
                    "Microsoft Entra ID authentication requires azure-identity; install the 'azure' optional dependency") from error
            credential = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
            self.authentication = "default_azure_credential"

        self.configuration = configuration
        self.model_id = configuration.model_id
        self._model = ChatOpenAI(
            model=configuration.model,
            base_url=configuration.base_url,
            api_key=credential,
            temperature=configuration.temperature,
            seed=configuration.seed,
        )

    def invoke(self, schema: type[SchemaT], system_prompt: str, user_prompt: str) -> SchemaT:
        return _invoke_with_repair(self._model, schema, system_prompt, user_prompt)


def _invoke_with_repair(
        model: Any,
        schema: type[SchemaT],
        system_prompt: str,
        user_prompt: str,
) -> SchemaT:
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
        except Exception as error:  # provider parsing errors vary by model
            last_error = error
            repair = (
                "\n\nYour previous response was invalid. Return only content that matches the required schema. "
                f"Repair attempt {attempt + 1}."
            )
    raise RuntimeError(f"structured model output failed: {last_error}") from last_error


def _scenario_prompt(scenario: Scenario) -> str:
    state = scenario.state
    baseline = {action.value: round(value, 4) for action, value in scenario.ep_baseline.items()}
    return json.dumps(
        {
            "scenario_id": scenario.scenario_id,
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
            "legal_actions": [action.value for action in state.legal_actions],
            "simple_ep_baseline": baseline,
        },
        indent=2,
    )


def _ep_decision(scenario: Scenario, rationale: str) -> Decision:
    action = max(scenario.state.legal_actions, key=lambda candidate: (scenario.ep_baseline[candidate], candidate.value))
    return Decision(
        action=action,
        go_for_it_play=Action.PASS if action == Action.GO_FOR_IT else None,
        rationale=rationale,
    )


def _fallback_recommendation(role: str, scenario: Scenario, error: Exception) -> Recommendation:
    return Recommendation(
        role=role,
        decision=_ep_decision(scenario, "Expected-points fallback after a specialist response failed"),
        confidence=0,
        argument="The specialist model response was unavailable; using the public EP baseline.",
        concerns=[str(error)[:300]],
    )


class ExpectedPointsStrategy:
    name = "expected_points"

    def decide(self, scenario: Scenario) -> DecisionTrace:
        started = time.perf_counter()
        return DecisionTrace(
            strategy=self.name,
            decision=_ep_decision(scenario, "Choose the legal action with the highest bucketed expected EPA"),
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
                    "You are the NFL head coach. Make one legal decision using only the "
                    "supplied game state and simple EP evidence. Balance win probability, "
                    "clock, execution risk, and score. Never invent unavailable facts."
                ),
                _scenario_prompt(scenario),
            )
            decision.validate_for(scenario.state)
        except Exception as error:
            failures.append(f"head_coach: {error}")
            fallback = True
            decision = _ep_decision(scenario, "Expected-points fallback after the head-coach call failed")
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

    def __init__(self, model: StructuredModel) -> None:
        self.model = model

    def _initial(self, role: str, focus: str, scenario: Scenario) -> Recommendation:
        result = self.model.invoke(
            Recommendation,
            (
                f"You are the NFL staff's {role.replace('_', ' ')}. {focus} "
                "Recommend exactly one legal action. If going for it, specify run or pass. Use only the provided information."
            ),
            _scenario_prompt(scenario),
        )
        result.decision.validate_for(scenario.state)
        return result.model_copy(update={"role": role})

    def _revise(
            self,
            role: str,
            focus: str,
            scenario: Scenario,
            initial: list[Recommendation],
    ) -> RevisedRecommendation:
        arguments = [
            {
                "speaker": f"staff_member_{index + 1}",
                "decision": rec.decision.model_dump(mode="json"),
                "confidence": rec.confidence,
                "argument": rec.argument,
                "concerns": rec.concerns,
            }
            for index, rec in enumerate(initial)
        ]
        result = self.model.invoke(
            RevisedRecommendation,
            (
                f"You are the NFL staff's {role.replace('_', ' ')}. {focus} "
                "Review the anonymized staff recommendations, rebut their weakest claims, "
                "then keep or revise your legal decision."
            ),
            _scenario_prompt(scenario)
            + "\n\nANONYMIZED INITIAL RECOMMENDATIONS:\n"
            + json.dumps(arguments, indent=2),
        )
        result.decision.validate_for(scenario.state)
        return result.model_copy(update={"role": role})

    def iter_decide(self, scenario: Scenario) -> Iterator[StageEvent]:
        started = time.perf_counter()
        failures: list[str] = []
        calls = 0
        initial: list[Recommendation] = []
        for role, focus in ROLES.items():
            calls += 1
            try:
                initial.append(self._initial(role, focus, scenario))
            except Exception as error:
                failures.append(f"{role} initial: {error}")
                initial.append(_fallback_recommendation(role, scenario, error))
        yield StageEvent(
            stage="recommendations",
            message="All five specialists submitted independent recommendations.",
        )

        revised: list[RevisedRecommendation] = []
        for role, focus in ROLES.items():
            calls += 1
            try:
                revised.append(self._revise(role, focus, scenario, initial))
            except Exception as error:
                failures.append(f"{role} revision: {error}")
                fallback = _fallback_recommendation(role, scenario, error)
                revised.append(
                    RevisedRecommendation(
                        **fallback.model_dump(),
                        rebuttal="No model rebuttal was available.",
                    )
                )
        yield StageEvent(
            stage="debate",
            message="The staff reviewed anonymized arguments and revised its calls.",
        )

        calls += 1
        fallback_used = False
        try:
            decision = self.model.invoke(
                Decision,
                (
                    "You are the head coach. Synthesize the revised staff debate and choose "
                    "one legal action. Resolve disagreement explicitly, prioritize winning "
                    "the game over raw expected points, and use no outside facts."
                ),
                _scenario_prompt(scenario)
                + "\n\nREVISED STAFF DEBATE:\n"
                + json.dumps(
                    [item.model_dump(mode="json") for item in revised], indent=2
                ),
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
            message=f"Head coach selected {decision.action.value}.",
            trace=trace,
        )

    def decide(self, scenario: Scenario) -> DecisionTrace:
        final: DecisionTrace | None = None
        for event in self.iter_decide(scenario):
            if event.trace is not None:
                final = event.trace
        if final is None:  # defensive invariant
            raise RuntimeError("deliberation ended without a decision")
        return final


def make_model(configuration: ModelConfiguration) -> StructuredModel:
    if configuration.provider == ModelProvider.OLLAMA:
        return LangChainOllamaModel(configuration)
    if configuration.provider == ModelProvider.AZURE_FOUNDRY:
        return AzureFoundryStructuredModel(configuration)
    raise ValueError(f"unsupported model provider: {configuration.provider}")

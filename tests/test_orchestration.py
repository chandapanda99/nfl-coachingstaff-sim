import json
import time
from threading import Lock
from typing import TypeVar

from pydantic import BaseModel

from nfl_coaching_sim.agents import (
    ModelConfiguration,
    ModelProvider,
    MultiAgentStrategy,
    make_model,
    register_model_provider,
)
from nfl_coaching_sim.app import format_coaching_conversation, format_live_coaching_conversation, run_strategy_events
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.football import build_situation_brief
from nfl_coaching_sim.models import (
    Action,
    Decision,
    Recommendation,
    RevisedRecommendation,
)
from nfl_coaching_sim.simulator import DeterministicSimulator

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class FakeStructuredModel:
    model_id = "fake:critical-path"

    def __init__(self) -> None:
        self._lock = Lock()
        self._active_calls = 0
        self.max_active_calls = 0
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def invoke(self, schema: type[SchemaT], system_prompt: str, user_prompt: str) -> SchemaT:
        with self._lock:
            self._active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self._active_calls)
            self.system_prompts.append(system_prompt)
            self.user_prompts.append(user_prompt)
        try:
            time.sleep(0.02)
            if schema is Decision:
                raise ValueError("malformed head-coach output after retries")
            payload, _ = json.JSONDecoder().raw_decode(user_prompt)
            legal_actions = [Action(value) for value in payload["legal_actions"]]
            preferred = Action.GO_FOR_IT if Action.GO_FOR_IT in legal_actions else legal_actions[-1]
            closest_alternative = next(action for action in legal_actions if action != preferred)
            dirty_citations = (
                ["CLOCK_CONTEXT", "MINIMUM_SCORING_POSSESSIONS no", "MINIMUM_SCORING_POSSESSIONSเป็น??"]
                if schema is RevisedRecommendation
                else ["CLOCK_CONTEXT", "MINIMUM_SCORING_POSSESSIONS,", "MINIMUM_SCORING_POSSESSIONS no"]
            )
            base = {
                "role": "replaced-by-strategy",
                "decision": {
                    "action": preferred,
                    "go_for_it_play": Action.PASS if preferred == Action.GO_FOR_IT else None,
                    "rationale": "Use the evidence packet to balance conversion value, clock, and field position.",
                },
                "confidence": 0.7,
                "argument": "I like the call here. CLOCK_PRIORITY supports the clock plan, and it keeps us attacking the sticks.",
                "concerns": ["failed-play field position"],
                "action_assessments": [
                    {
                        "action": action,
                        "advantages": [f"The released evidence provides a comparison for {action.football_label.lower()}."],
                        "risks": ["Failure can reduce possession or field-position value."],
                        "clock_effect": "The call must support the stated clock priority.",
                        "evidence_ids": [
                            f"EP_BASELINE_{action.value.upper()}",
                            dirty_citations[index],
                        ],
                        "support_score": 0.7 if action == preferred else 0.3,
                    }
                    for index, action in enumerate(legal_actions)
                ],
                "closest_alternative": closest_alternative,
                "switch_condition": "Switch if verified pre-snap information materially changes the conversion or field-position tradeoff.",
            }
            if schema is RevisedRecommendation:
                return schema.model_validate({**base, "rebuttal": "The run argument understates clock urgency."})
            return schema.model_validate(base)
        finally:
            with self._lock:
                self._active_calls -= 1


def test_full_debate_uses_deterministic_vote_when_synthesis_fails(
    monkeypatch,
) -> None:
    foundry = ModelConfiguration(
        provider=ModelProvider.AZURE_FOUNDRY,
        model="open-model-deployment",
        base_url="https://example.services.ai.azure.com/openai/v1/",
        license="Apache-2.0",
        temperature=0.25,
        seed=77,
    )
    assert foundry.model_id == "azure_foundry:open-model-deployment"
    assert foundry.upstream_url is None
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "environment-only-test-key")
    foundry_model = make_model(foundry)
    assert foundry_model.model_id == foundry.model_id
    assert foundry_model.authentication == "api_key_environment"
    assert foundry_model.capabilities.api_mode == "responses"
    assert foundry_model.effective_generation_parameters == {"temperature": 0.25}
    assert foundry_model._model.use_responses_api is True
    foundry_payload = foundry_model._model._get_request_payload("test play call")
    assert foundry_payload["temperature"] == 0.25
    assert "seed" not in foundry_payload
    assert foundry_model._model._use_responses_api(foundry_payload) is True

    ollama_model = make_model(
        ModelConfiguration(
            provider=ModelProvider.OLLAMA,
            model="local-open-model",
            base_url="http://127.0.0.1:11434",
            license="Apache-2.0",
            temperature=0.25,
            seed=77,
        )
    )
    assert ollama_model.capabilities.api_mode == "native"
    assert ollama_model.effective_generation_parameters == {"temperature": 0.25, "seed": 77}
    ollama_payload = ollama_model._model._chat_params([], None)
    assert ollama_payload["options"] == {"temperature": 0.25, "seed": 77}

    register_model_provider("test_provider", lambda configuration: (object(), "test_auth"))
    registered = make_model(
        ModelConfiguration(
            provider="test_provider",
            model="registered-model",
            base_url="https://provider.example/v1",
            license="MIT",
        )
    )
    assert registered.model_id == "test_provider:registered-model"
    assert registered.authentication == "test_auth"

    scenario = next(item for item in demo_scenarios() if item.state.down == 4)
    brief = build_situation_brief(scenario)
    fake_model = FakeStructuredModel()
    stage_events = list(MultiAgentStrategy(fake_model).iter_decide(scenario))
    trace = next(event.trace for event in stage_events if event.trace is not None)
    opening_events = [event for event in stage_events if event.recommendation is not None]
    revision_events = [event for event in stage_events if event.revision is not None]
    streamed_opening: dict[str, Recommendation] = {}
    opening_snapshots: list[str] = []
    for event in opening_events:
        assert event.role is not None
        streamed_opening[event.role] = event.recommendation
        opening_snapshots.append(format_live_coaching_conversation(scenario, streamed_opening, {}, phase="opening"))
    streamed_revisions = {event.role: event.revision for event in revision_events if event.role and event.revision}
    revision_snapshot = format_live_coaching_conversation(
        scenario,
        streamed_opening,
        streamed_revisions,
        phase="revision",
    )
    streaming_model = FakeStructuredModel()
    monkeypatch.setattr("nfl_coaching_sim.app.make_model", lambda configuration: streaming_model)
    ui_updates = list(
        run_strategy_events(
            scenario.scenario_id,
            "multi_agent",
            "ollama",
            "fake-model",
            "",
            "Apache-2.0",
            "http://127.0.0.1:11434",
            [scenario.model_dump(mode="json")],
            DeterministicSimulator(),
        )
    )
    conversation = format_coaching_conversation(trace, scenario)
    sanitized_trace = trace.model_copy(
        update={"failures": ["critical_reviewer initial: recommendation cited unknown evidence IDs: ['RANDOM_TAG']"]}
    )
    sanitized_conversation = format_coaching_conversation(sanitized_trace, scenario)

    assert trace.model_calls == 11
    assert len(opening_events) == 5
    assert len(revision_events) == 5
    assert len(stage_events) == 13
    assert all(event.recommendation is not None and event.role == event.recommendation.role for event in opening_events)
    assert all(event.revision is not None and event.role == event.revision.role for event in revision_events)
    assert opening_snapshots[0].count("Headset open — reviewing the call sheet") == 4
    assert opening_snapshots[-1].count("Headset open — reviewing the call sheet") == 0
    assert "Staff Challenge Round" in revision_snapshot
    assert revision_snapshot.count("Listening to the staff challenge") == 0
    assert len(ui_updates) == 16
    assert "Completed coaching responses will appear here" in ui_updates[0][1]
    assert ui_updates[1][1].count("Headset open — reviewing the call sheet") == 5
    assert any(update[1].count("Headset open — reviewing the call sheet") == 4 for update in ui_updates)
    assert any("Staff Challenge Round" in update[1] for update in ui_updates)
    assert any("Listening to the staff challenge" in update[1] for update in ui_updates)
    assert "Head Coach Breaks the Huddle" in ui_updates[-1][1]
    assert "CALL IS IN:" in ui_updates[-1][0]
    assert fake_model.max_active_calls > 1
    assert trace.fallback_used is True
    assert trace.decision.action == Action.GO_FOR_IT
    assert trace.decision.go_for_it_play == Action.PASS
    assert trace.transcript is not None
    assert "Opening Headset Check" in conversation
    assert "Offensive Coordinator" in conversation
    assert "Head Coach Breaks the Huddle" in conversation
    assert "Evidence:" not in conversation
    assert "CLOCK_PRIORITY" not in conversation
    assert "EP_BASELINE_" not in conversation
    assert "RANDOM_TAG" not in sanitized_conversation
    assert "could not be verified" in sanitized_conversation
    assert len(trace.transcript.initial) == 5
    assert len(trace.transcript.revised) == 5
    assert not any("unknown evidence" in failure for failure in trace.failures)
    for recommendation in [*trace.transcript.initial, *trace.transcript.revised]:
        cited = {evidence_id for assessment in recommendation.action_assessments for evidence_id in assessment.evidence_ids}
        assert cited <= brief.evidence_ids
    assert [item.role for item in trace.transcript.initial] == [
        "offensive_coordinator",
        "defensive_coordinator",
        "analytics_assistant",
        "clock_management_specialist",
        "critical_reviewer",
    ]
    assert brief.approximate_field_goal_yards == round(scenario.state.yardline_100 + 17)
    assert {item.evidence_id for item in brief.evidence} >= {
        "STATE_DOWN_DISTANCE",
        "CLOCK_PRIORITY",
        "FIELD_GOAL_DISTANCE",
        *{f"EP_BASELINE_{action.value.upper()}" for action in scenario.state.legal_actions},
    }
    assert all(
        {assessment.action for assessment in recommendation.action_assessments} == set(scenario.state.legal_actions)
        for recommendation in trace.transcript.initial
    )
    assert any("COACHING CHECKLIST" in prompt for prompt in fake_model.system_prompts)
    assert any("private validation tags" in prompt for prompt in fake_model.system_prompts)
    assert any("HEADSET VOICE" in prompt for prompt in fake_model.system_prompts)
    assert any("actual front" in prompt.lower() and "unavailable" in prompt.lower() for prompt in fake_model.system_prompts)
    assert all("evidence_packet" in prompt for prompt in fake_model.user_prompts)
    assert any("head_coach" in failure for failure in trace.failures)

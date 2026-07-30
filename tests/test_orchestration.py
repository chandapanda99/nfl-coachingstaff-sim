import time
from threading import Lock
from typing import TypeVar

from pydantic import BaseModel

from nfl_coaching_sim.agents import (
    ModelConfiguration,
    ModelProvider,
    MultiAgentStrategy,
    make_model,
)
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.models import (
    Action,
    Decision,
    Recommendation,
    RevisedRecommendation,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class FakeStructuredModel:
    model_id = "fake:critical-path"

    def __init__(self) -> None:
        self._lock = Lock()
        self._active_calls = 0
        self.max_active_calls = 0

    def invoke(self, schema: type[SchemaT], system_prompt: str, user_prompt: str) -> SchemaT:
        with self._lock:
            self._active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self._active_calls)
        try:
            time.sleep(0.02)
            if schema is Decision:
                raise ValueError("malformed head-coach output after retries")
            base = {
                "role": "replaced-by-strategy",
                "decision": {
                    "action": Action.PASS,
                    "rationale": "Preserve clock and attack the sticks.",
                },
                "confidence": 0.7,
                "argument": "Pass has the best balance of clock and conversion value.",
                "concerns": ["sack risk"],
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
    )
    assert foundry.model_id == "azure_foundry:open-model-deployment"
    assert foundry.upstream_url is None
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "environment-only-test-key")
    foundry_model = make_model(foundry)
    assert foundry_model.model_id == foundry.model_id
    assert foundry_model.authentication == "api_key_environment"

    scenario = demo_scenarios()[1]  # second down: pass is legal
    fake_model = FakeStructuredModel()
    trace = MultiAgentStrategy(fake_model).decide(scenario)

    assert trace.model_calls == 11
    assert fake_model.max_active_calls > 1
    assert trace.fallback_used is True
    assert trace.decision.action == Action.PASS
    assert trace.transcript is not None
    assert len(trace.transcript.initial) == 5
    assert len(trace.transcript.revised) == 5
    assert [item.role for item in trace.transcript.initial] == [
        "offensive_coordinator",
        "defensive_coordinator",
        "analytics_assistant",
        "clock_management_specialist",
        "critical_reviewer",
    ]
    assert any("head_coach" in failure for failure in trace.failures)

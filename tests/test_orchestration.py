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

    def invoke(
        self, schema: type[SchemaT], system_prompt: str, user_prompt: str
    ) -> SchemaT:
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
            return schema.model_validate(
                {**base, "rebuttal": "The run argument understates clock urgency."}
            )
        return schema.model_validate(base)


def test_full_debate_uses_deterministic_vote_when_synthesis_fails(
    monkeypatch,
) -> None:
    foundry = ModelConfiguration(
        provider=ModelProvider.AZURE_FOUNDRY,
        model="open-model-deployment",
        base_url="https://example.services.ai.azure.com/openai/v1/",
        upstream_url="https://example.org/open-model",
        license="Apache-2.0",
    )
    assert foundry.model_id == "azure_foundry:open-model-deployment"
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "environment-only-test-key")
    foundry_model = make_model(foundry)
    assert foundry_model.model_id == foundry.model_id
    assert foundry_model.authentication == "api_key_environment"

    scenario = demo_scenarios()[1]  # second down: pass is legal
    trace = MultiAgentStrategy(FakeStructuredModel()).decide(scenario)

    assert trace.model_calls == 11
    assert trace.fallback_used is True
    assert trace.decision.action == Action.PASS
    assert trace.transcript is not None
    assert len(trace.transcript.initial) == 5
    assert len(trace.transcript.revised) == 5
    assert any("head_coach" in failure for failure in trace.failures)

"""Gradio Blocks application with thin, reusable callbacks."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any

from nfl_coaching_sim.agents import (
    ExpectedPointsStrategy,
    ModelConfiguration,
    ModelProvider,
    MultiAgentStrategy,
    SingleAgentStrategy,
    make_model,
)
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.models import DecisionTrace, Scenario
from nfl_coaching_sim.simulator import DeterministicSimulator


def scenario_view(
    scenario_id: str, scenario_payloads: Sequence[dict[str, Any]]
) -> tuple[str, dict[str, Any], list[list[Any]]]:
    scenario = next(
        Scenario.model_validate(item)
        for item in scenario_payloads
        if item["scenario_id"] == scenario_id
    )
    state = scenario.state
    scoreboard = (
        f"## {state.possession_team} {state.possession_score} — "
        f"{state.defensive_team} {state.defensive_score}\n"
        f"**{state.clock_display} · {state.down} & {state.yards_to_go:g} · "
        f"ball at the opponent {state.yardline_100:g}**"
    )
    state_data = state.model_dump(mode="json")
    table = [
        [action.value, round(scenario.ep_baseline[action], 4)]
        for action in state.legal_actions
    ]
    return scoreboard, state_data, table


def _transcript_markdown(trace: DecisionTrace) -> str:
    if trace.transcript is None:
        return f"### {trace.strategy}\n\n{trace.decision.rationale}"
    sections = ["### Independent recommendations"]
    for rec in trace.transcript.initial:
        sections.append(
            f"**{rec.role.replace('_', ' ').title()} — {rec.decision.action.value} "
            f"({rec.confidence:.0%})**\n\n{rec.argument}"
        )
    sections.append("### Revised debate")
    for rec in trace.transcript.revised:
        sections.append(
            f"**{rec.role.replace('_', ' ').title()} — {rec.decision.action.value}**"
            f"\n\n{rec.rebuttal}"
        )
    if trace.failures:
        sections.append("### Recorded failures\n\n" + "\n".join(f"- {item}" for item in trace.failures))
    return "\n\n".join(sections)


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
) -> Iterator[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    scenario = next(
        Scenario.model_validate(item)
        for item in scenario_payloads
        if item["scenario_id"] == scenario_id
    )
    yield "Preparing strategy…", "", {}, {}
    if strategy_name == "Expected points":
        strategy: Any = ExpectedPointsStrategy()
        trace = strategy.decide(scenario)
    else:
        configuration = ModelConfiguration(
            provider=ModelProvider(provider_name),
            model=model_name,
            upstream_url=upstream_url,
            license=model_license,
            base_url=base_url,
        )
        model = make_model(configuration)
        if strategy_name == "Single agent":
            trace = SingleAgentStrategy(model).decide(scenario)
        else:
            strategy = MultiAgentStrategy(model)
            trace = None
            for event in strategy.iter_decide(scenario):
                yield event.message, "", {}, {}
                if event.trace is not None:
                    trace = event.trace
            if trace is None:
                raise RuntimeError("multi-agent run did not return a decision")
    value = simulator.score(scenario, trace.decision)
    yield (
        f"Complete: {trace.decision.action.value}",
        _transcript_markdown(trace),
        trace.decision.model_dump(mode="json"),
        value.model_dump(mode="json"),
    )


def create_app(
    scenarios: Sequence[Scenario] | None = None,
    simulator: DeterministicSimulator | None = None,
) -> Any:
    import gradio as gr

    scenario_list = list(scenarios or demo_scenarios())
    evaluator = simulator or DeterministicSimulator()
    payloads = [item.model_dump(mode="json") for item in scenario_list]
    first = scenario_list[0]
    initial_scoreboard, initial_state, initial_table = scenario_view(
        first.scenario_id, payloads
    )

    def run_callback(
        scenario_id: str,
        strategy_name: str,
        provider_name: str,
        model: str,
        source: str,
        license_name: str,
        url: str,
        items: Sequence[dict[str, Any]],
    ) -> Iterator[tuple[str, str, dict[str, Any], dict[str, Any]]]:
        yield from run_strategy_events(
            scenario_id,
            strategy_name,
            provider_name,
            model,
            source,
            license_name,
            url,
            items,
            evaluator,
        )

    with gr.Blocks(title="NFL Coaching Staff Simulator") as demo:
        gr.Markdown(
            "# Multi-Agent NFL Coaching Simulator\n"
            "Compare an expected-points policy, one head coach, and a deliberating staff."
        )
        session_scenarios = gr.State(payloads)
        with gr.Row():
            with gr.Column(scale=2):
                scenario_selector = gr.Dropdown(
                    choices=[item.scenario_id for item in scenario_list],
                    value=first.scenario_id,
                    label="Scenario",
                )
                scoreboard = gr.Markdown(initial_scoreboard)
                state_json = gr.JSON(initial_state, label="Game state")
                baseline = gr.Dataframe(
                    value=initial_table,
                    headers=["Legal action", "Simple expected EPA"],
                    datatype=["str", "number"],
                    interactive=False,
                    label="Expected-points baseline",
                )
            with gr.Column(scale=2):
                strategy = gr.Radio(
                    ["Expected points", "Single agent", "Multi-agent"],
                    value="Expected points",
                    label="Strategy",
                )
                provider = gr.Dropdown(
                    choices=[
                        (ModelProvider.OLLAMA.value, ModelProvider.OLLAMA.value),
                        (
                            "Azure AI Foundry",
                            ModelProvider.AZURE_FOUNDRY.value,
                        ),
                    ],
                    value=ModelProvider.OLLAMA.value,
                    label="Model provider",
                )
                model_name = gr.Textbox(
                    label="Model tag or deployment name (required for LLM strategies)"
                )
                upstream_url = gr.Textbox(
                    label="Upstream model URL",
                    placeholder="https://huggingface.co/organization/model",
                )
                model_license = gr.Dropdown(
                    ["Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "MPL-2.0"],
                    label="Model SPDX license",
                )
                base_url = gr.Textbox(
                    value="http://127.0.0.1:11434",
                    label="Provider endpoint",
                    info=(
                        "Ollama URL, or a Foundry endpoint ending in /openai/v1/. "
                        "Foundry uses Entra ID unless AZURE_FOUNDRY_API_KEY is set."
                    ),
                )
                run = gr.Button("Make the call", variant="primary")
                status = gr.Markdown()
        with gr.Row():
            transcript = gr.Markdown(label="Debate")
            final_decision = gr.JSON(label="Final decision")
            score = gr.JSON(label="Deterministic score")

        scenario_selector.change(
            scenario_view,
            inputs=[scenario_selector, session_scenarios],
            outputs=[scoreboard, state_json, baseline],
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

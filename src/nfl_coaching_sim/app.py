"""Gradio Blocks application with thin, reusable callbacks."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from typing import Any
from dotenv import load_dotenv, find_dotenv
from pydantic import HttpUrl

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

load_dotenv(find_dotenv())


def scenario_view(scenario_id: str, scenario_payloads: Sequence[dict[str, Any]]) -> tuple[str, dict[str, Any], list[list[Any]]]:
    scenario = next(Scenario.model_validate(item) for item in scenario_payloads if item["scenario_id"] == scenario_id)
    state = scenario.state
    scoreboard = (
        f"## {state.possession_team} {state.possession_score} — {state.defensive_team} {state.defensive_score}\n"
        f"**{state.possession_team} Ball · {state.clock_display} · "
        f"{state.down_and_distance} · Ball on {state.field_position}**\n\n"
        f"Timeouts: {state.possession_team} {state.possession_timeouts} · "
        f"{state.defensive_team} {state.defensive_timeouts}"
    )
    state_data = state.model_dump(mode="json")
    table = [[action.football_label, round(scenario.ep_baseline[action], 4)] for action in state.legal_actions]
    return scoreboard, state_data, table


def _transcript_markdown(trace: DecisionTrace) -> str:
    if trace.transcript is None:
        strategy_name = {
            "expected_points": "Analytics Booth",
            "single_agent": "Head Coach",
        }.get(trace.strategy, trace.strategy.replace("_", " ").title())
        return f"### {strategy_name}: {trace.decision.call_label}\n\n{trace.decision.rationale}"
    sections = ["### Opening Staff Calls"]
    for rec in trace.transcript.initial:
        sections.append(
            f"**{rec.role.replace('_', ' ').title()} — {rec.decision.call_label} " f"({rec.confidence:.0%} confidence)**\n\n{rec.argument}"
        )
    sections.append("### Challenge-and-Adjust Round")
    for rec in trace.transcript.revised:
        sections.append(f"**{rec.role.replace('_', ' ').title()} — {rec.decision.call_label}**\n\n{rec.rebuttal}")
    if trace.failures:
        sections.append("### Headset / Communication Issues\n\n" + "\n".join(f"- {item}" for item in trace.failures))
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
    scenario = next(Scenario.model_validate(item) for item in scenario_payloads if item["scenario_id"] == scenario_id)
    yield "Breaking the Huddle & Getting the Call Sheet Ready…", "", {}, {}
    if strategy_name == "expected_points":
        strategy: Any = ExpectedPointsStrategy()
        trace = strategy.decide(scenario)
    else:
        configuration = ModelConfiguration(
            provider=ModelProvider(provider_name),
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
            for event in strategy.iter_decide(scenario):
                yield event.message, "", {}, {}
                if event.trace is not None:
                    trace = event.trace
            if trace is None:
                raise RuntimeError("The coaches' meeting ended without a legal call reaching the sideline.")
    value = simulator.score(scenario, trace.decision)
    yield (
        f"CALL IS IN: **{trace.decision.call_label}**",
        _transcript_markdown(trace),
        trace.decision.model_dump(mode="json"),
        value.model_dump(mode="json"),
    )


def create_app(scenarios: Sequence[Scenario] | None = None, simulator: DeterministicSimulator | None = None) -> Any:
    import gradio as gr

    scenario_list = list(scenarios or demo_scenarios())
    evaluator = simulator or DeterministicSimulator()
    payloads = [item.model_dump(mode="json") for item in scenario_list]
    first = scenario_list[0]
    initial_scoreboard, initial_state, initial_table = scenario_view(first.scenario_id, payloads)

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
        yield from run_strategy_events(scenario_id, strategy_name, provider_name, model, source, license_name, url, items, evaluator)

    with gr.Blocks(title="NFL Coaching Staff Simulator") as demo:
        gr.Markdown(
            "# NFL Virtual Coaching Staff\n"
            "Put the situation on the call sheet, hear every coordinator, and see what the head coach sends in."
        )
        session_scenarios = gr.State(payloads)
        with gr.Row():
            with gr.Column(scale=2):
                scenario_selector = gr.Dropdown(
                    choices=[(item.display_name, item.scenario_id) for item in scenario_list],
                    value=first.scenario_id,
                    label="Game Situation",
                )
                scoreboard = gr.Markdown(initial_scoreboard)
                state_json = gr.JSON(initial_state, label="Pre-Snap Situation Data")
                baseline = gr.Dataframe(
                    value=initial_table,
                    headers=["Call Sheet Option", "Expected EPA"],
                    datatype=["str", "number"],
                    interactive=False,
                    label="Analytics Booth: Expected Points by Call",
                )
            with gr.Column(scale=2):
                strategy = gr.Radio(
                    choices=[
                        ("Analytics booth only", "expected_points"),
                        ("Head Coach only", "single_agent"),
                        ("Full Coaching Staff", "multi_agent"),
                    ],
                    value="expected_points",
                    label="Who Makes the Call?",
                )
                provider = gr.Dropdown(
                    choices=[
                        ("Local sideline (Ollama)", ModelProvider.OLLAMA.value),
                        ("Azure AI Foundry", ModelProvider.AZURE_FOUNDRY.value),
                    ],
                    value=ModelProvider.AZURE_FOUNDRY.value,
                    label="Coaching Staff Model Provider",
                )
                model_name = gr.Textbox(value=os.environ.get("FOUNDRY_MODEL"), label="Model / Deployment on the Call Sheet")
                upstream_url = gr.Textbox(
                    label="Model Card / Film Room URL (optional)",
                    placeholder="https://huggingface.co/organization/model",
                )
                model_license = gr.Dropdown(
                    ["Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "MPL-2.0"],
                    label="Model License",
                )
                base_url = gr.Textbox(
                    value=os.environ.get("FOUNDRY_ENDPOINT") or "http://127.0.0.1:11434",
                    label="Sideline Connection / Provider Endpoint",
                    info="Ollama URL, or a Foundry endpoint ending in /openai/v1/. "
                    "Foundry uses Entra ID unless AZURE_FOUNDRY_API_KEY is set.",
                )
                run = gr.Button("Send in the Call!", variant="primary")
                status = gr.Markdown()
        with gr.Row():
            transcript = gr.Markdown(label="Coaches' Meeting")
            final_decision = gr.JSON(label="Head Coach's Call")
            score = gr.JSON(label="Postgame Decision Grade")

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

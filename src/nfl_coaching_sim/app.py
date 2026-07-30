"""Gradio Blocks application with thin, reusable callbacks."""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from html import escape
from importlib.resources import files
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
)
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.models import Action, ActionValue, DecisionTrace, Scenario

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
        secondary_hue="indigo",
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
    fallback = '<span class="warning-chip">Fallback call used</span>' if trace.fallback_used else ""
    model_name = _safe(trace.model_id or "Deterministic analytics policy")
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
    sections.append("### Debate-and-Discuss Round")
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
) -> Iterator[tuple[str, str, str, str]]:
    scenario = next(Scenario.model_validate(item) for item in scenario_payloads if item["scenario_id"] == scenario_id)
    yield (
        _live_status("Breaking the huddle and getting the call sheet ready…"),
        "",
        _decision_placeholder(),
        _grade_placeholder(),
    )
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
                yield _live_status(event.message), "", _decision_placeholder(), _grade_placeholder()
                if event.trace is not None:
                    trace = event.trace
            if trace is None:
                raise RuntimeError("The coaches' meeting ended without a legal call reaching the sideline.")
    value = simulator.score(scenario, trace.decision)
    yield (
        _live_status(f"CALL IS IN: **{trace.decision.call_label}**"),
        _transcript_markdown(trace),
        _decision_card(trace),
        _grade_card(value),
    )


def create_app(scenarios: Sequence[Scenario] | None = None, simulator: DeterministicSimulator | None = None) -> Any:
    import gradio as gr

    scenario_list = list(scenarios or demo_scenarios())
    if simulator is None:
        from nfl_coaching_sim.simulator import DeterministicSimulator

        evaluator = DeterministicSimulator()
    else:
        evaluator = simulator
    payloads = [item.model_dump(mode="json") for item in scenario_list]
    first = scenario_list[0]
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

    with gr.Blocks(title="NFL Virtual Coaching Staff", analytics_enabled=False) as demo:
        gr.Markdown(
            "# NFL Virtual Coaching Staff\n"
            "Put the situation on the call sheet, hear every coordinator, and see what the head coach sends in."
        )
        session_scenarios = gr.State(payloads)
        with gr.Row(equal_height=False, elem_id="game-plan-row"):
            with gr.Column(scale=2, min_width=420, elem_id="game-situation-column"):
                scenario_selector = gr.Dropdown(
                    choices=[(item.display_name, item.scenario_id) for item in scenario_list],
                    value=first.scenario_id,
                    label="Game Situation",
                    elem_id="game-situation-selector",
                )
                state_card = gr.HTML(
                    initial_state,
                    elem_id="pre-snap-card",
                    elem_classes="coach-card-component",
                )
            with gr.Column(scale=2, min_width=360, elem_id="sideline-control-column"):
                strategy = gr.Radio(
                    choices=[
                        ("Analytics booth only", "expected_points"),
                        ("Head Coach only", "single_agent"),
                        ("Full Coaching Staff", "multi_agent"),
                    ],
                    value="multi_agent",
                    label="Who's Making the Calls?",
                )
                with gr.Accordion("Sideline Connection & Model Settings", open=False, elem_id="provider-config"):
                    provider = gr.Dropdown(
                        choices=[
                            ("Local sideline (Ollama)", ModelProvider.OLLAMA.value),
                            ("Azure AI Foundry", ModelProvider.AZURE_FOUNDRY.value),
                        ],
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
                run = gr.Button("🏈 Send in the Call! 🏈", variant="primary", elem_id="send-play-call")
                with gr.Group(elem_id="play-call-status"):
                    status = gr.Markdown(_live_status("Waiting for Coach to put in the call..."), elem_id="live-play-call-status")
                baseline = gr.HTML(
                    initial_analytics,
                    elem_id="analytics-booth",
                    elem_classes=["coach-card-component", "sideline-analytics"],
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
        transcript = gr.Markdown(label="Coaches' Meeting")

        scenario_selector.change(
            scenario_view,
            inputs=[scenario_selector, session_scenarios],
            outputs=[state_card, baseline],
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

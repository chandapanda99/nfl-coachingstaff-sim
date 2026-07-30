from html import escape

from typer.testing import CliRunner

from nfl_coaching_sim import cli
from nfl_coaching_sim.app import create_app, create_app_theme, load_app_css, run_strategy_events, scenario_view
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.simulator import DeterministicSimulator


def test_gradio_app_builds_and_critical_callbacks_run_without_model_server(
    monkeypatch,
) -> None:
    scenarios = demo_scenarios()
    payloads = [item.model_dump(mode="json") for item in scenarios]
    state, baseline = scenario_view(scenarios[0].scenario_id, payloads)
    events = list(
        run_strategy_events(
            scenarios[0].scenario_id,
            "expected_points",
            "ollama",
            "",
            "",
            "",
            "",
            payloads,
            DeterministicSimulator(),
        )
    )
    app = create_app(scenarios[:2], DeterministicSimulator())
    theme = create_app_theme()
    stylesheet = load_app_css()
    launches = []
    monkeypatch.setattr(cli, "_launch_app", lambda *args, **kwargs: launches.append(True))
    result = CliRunner().invoke(cli.app, [])

    assert "CALL IS IN:" in events[-1][0]
    assert "Analytics Booth" in events[-1][1]
    assert "Situation at a Glance" in state
    assert escape(scenarios[0].state.down_and_distance) in state
    assert "Expected Value by Call" in baseline
    assert "Top option" in baseline
    assert "Head Coach's Call" in events[-1][2]
    assert "Call Sent to the Huddle" in events[-1][2]
    assert "Win Probability Added" in events[-1][3]
    assert app is not None
    assert theme is not None
    assert "noinspection" not in stylesheet
    assert "var(--" not in stylesheet
    assert '#live-play-call-status > [data-testid="status-tracker"]' in stylesheet
    assert ".grade-banner > strong" in stylesheet
    assert ".grade-banner > span" in stylesheet
    assert "send-play-call" in str(app.config)
    assert "game-plan-row" in str(app.config)
    assert "game-situation-selector" in str(app.config)
    assert "decision-dashboard" in str(app.config)
    assert "analytics-booth" in str(app.config)
    assert "sideline-analytics" in str(app.config)
    assert "Sideline Connection & Model Settings" in str(app.config)
    assert "live-play-call-status" in str(app.config)
    assert scenarios[0].display_name in str(app.config)
    assert scenarios[0].scenario_id not in scenarios[0].display_name
    assert result.exit_code == 0
    assert launches == [True]

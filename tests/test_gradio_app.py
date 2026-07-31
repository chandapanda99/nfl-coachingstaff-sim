from html import escape

from typer.testing import CliRunner

from nfl_coaching_sim import cli
from nfl_coaching_sim.app import (
    create_app,
    create_app_theme,
    create_custom_scenario,
    load_app_css,
    run_strategy_events,
    scenario_view,
    scenario_view_with_reset,
)
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.simulator import DeterministicSimulator


def test_gradio_app_builds_and_critical_callbacks_run_without_model_server(
    monkeypatch,
) -> None:
    scenarios = demo_scenarios()
    payloads = [item.model_dump(mode="json") for item in scenarios]
    state, baseline = scenario_view(scenarios[0].scenario_id, payloads)
    reset_view = scenario_view_with_reset(scenarios[1].scenario_id, payloads)
    custom = create_custom_scenario(
        2025,
        18,
        "chi",
        "gb",
        24,
        27,
        4,
        "1:12",
        4,
        3,
        "defense",
        38,
        1,
        2,
    )
    custom_state, custom_baseline = scenario_view(custom.scenario_id, [*payloads, custom.model_dump(mode="json")])
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
    assert "New situation is on the call sheet" in reset_view[2]
    assert reset_view[3] == ""
    assert "Waiting on the sideline" in reset_view[4]
    assert "No grade on the board yet" in reset_view[5]
    assert "Top option" in baseline
    assert "CHI" in custom_state
    assert "GB 38" in custom_state
    assert "Q4 1:12" in custom_state
    assert "Keep the offense on the field" in custom_state
    assert "Expected Value by Call" in custom_baseline
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
    assert "open-custom-situation" in str(app.config)
    assert "custom-situation-modal" in str(app.config)
    assert "New Situation" in str(app.config)
    assert "Add to Call Sheet" in str(app.config)
    assert "decision-dashboard" in str(app.config)
    assert "analytics-booth" in str(app.config)
    assert "sideline-analytics" in str(app.config)
    assert "Sideline Connection & Model Settings" in str(app.config)
    assert "live-play-call-status" in str(app.config)
    assert scenarios[0].display_name in str(app.config)
    assert scenarios[0].scenario_id not in scenarios[0].display_name
    assert result.exit_code == 0
    assert launches == [True]

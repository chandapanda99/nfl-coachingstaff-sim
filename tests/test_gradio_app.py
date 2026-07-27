from typer.testing import CliRunner

from nfl_coaching_sim import cli
from nfl_coaching_sim.app import create_app, run_strategy_events, scenario_view
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.simulator import DeterministicSimulator


def test_gradio_app_builds_and_critical_callbacks_run_without_model_server(
    monkeypatch,
) -> None:
    scenarios = demo_scenarios()
    payloads = [item.model_dump(mode="json") for item in scenarios]
    scoreboard, state, baseline = scenario_view(scenarios[0].scenario_id, payloads)
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
    launches = []
    monkeypatch.setattr(cli, "_launch_app", lambda *args, **kwargs: launches.append(True))
    result = CliRunner().invoke(cli.app, [])

    assert "Q" in scoreboard
    assert f"{scenarios[0].state.possession_team} Ball" in scoreboard
    assert scenarios[0].state.down_and_distance in scoreboard
    assert "CALL IS IN:" in events[-1][0]
    assert "Analytics Booth" in events[-1][1]
    assert state["down"] == scenarios[0].state.down
    assert baseline
    assert events[-1][2]["action"]
    assert app is not None
    assert scenarios[0].display_name in str(app.config)
    assert scenarios[0].scenario_id not in scenarios[0].display_name
    assert result.exit_code == 0
    assert launches == [True]

from html import escape

from typer.testing import CliRunner

from nfl_coaching_sim import cli
from nfl_coaching_sim.app import (
    create_app,
    create_app_theme,
    create_custom_scenario,
    custom_scenario_form_values,
    scenarios_for_library,
    load_app_css,
    load_app_js,
    run_strategy_events,
    scenario_view,
    scenario_view_with_reset,
)
from nfl_coaching_sim.data import demo_scenarios
from nfl_coaching_sim.simulator import DeterministicSimulator
from nfl_coaching_sim.scenario_library import delete_custom_scenario, load_custom_scenarios, save_custom_scenario
from nfl_coaching_sim.settings import ApplicationSettings


def test_gradio_app_builds_and_critical_callbacks_run_without_model_server(
    monkeypatch,
    tmp_path,
) -> None:
    scenarios = demo_scenarios()
    payloads = [item.model_dump(mode="json") for item in scenarios]
    state, baseline = scenario_view(scenarios[0].scenario_id, payloads)
    reset_view = scenario_view_with_reset(scenarios[1].scenario_id, payloads)
    custom = create_custom_scenario(
        "Must-Have Fourth Down",
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
    custom_path = tmp_path / "custom-scenarios.jsonl"
    save_custom_scenario(custom_path, custom)
    reloaded_custom = load_custom_scenarios(custom_path)
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
    field_goal_events = list(
        run_strategy_events(
            scenarios[3].scenario_id,
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
    model_defaults = ApplicationSettings(
        provider="azure_foundry",
        model="default-foundry-deployment",
        base_url="https://example.services.ai.azure.com/openai/v1/",
        upstream_url="https://example.org/open-model",
        model_license="Apache-2.0",
        reasoning_effort="medium",
    )
    app = create_app(
        scenarios[:2],
        DeterministicSimulator(),
        custom_scenarios_path=custom_path,
        settings=model_defaults,
    )
    edited_custom = create_custom_scenario(
        "Edited Fourth Down",
        2025,
        18,
        "chi",
        "gb",
        24,
        27,
        4,
        "0:48",
        4,
        2,
        "defense",
        36,
        1,
        2,
        17.5,
        1.25,
    )
    save_custom_scenario(custom_path, edited_custom, replacing_scenario_id=custom.scenario_id)
    edited_values = custom_scenario_form_values(edited_custom)
    edited_library = load_custom_scenarios(custom_path)
    remaining_library = delete_custom_scenario(custom_path, edited_custom.scenario_id)
    theme = create_app_theme()
    stylesheet = load_app_css()
    browser_script = load_app_js()
    launches = []
    monkeypatch.setattr(cli, "_launch_app", lambda *args, **kwargs: launches.append(True))
    result = CliRunner().invoke(cli.app, [])

    assert "CALL IS IN:" in events[-1][0]
    assert "Analytics Booth" in events[-1][1]
    assert "Sideline Tablet" in state
    assert "football-field" in state
    assert "Line of scrimmage" in state
    assert "Line to gain" in state
    assert "driving" in state
    assert escape(scenarios[0].state.down_and_distance) in state
    assert "Expected Value by Call" in baseline
    assert "New situation is on the call sheet" in reset_view[2]
    assert "Completed responses will appear here in arrival order" in reset_view[3]
    assert "Waiting on the sideline" in reset_view[4]
    assert "No grade on the board yet" in reset_view[5]
    assert "Top option" in baseline
    assert "CHI" in custom_state
    assert "GB 38" in custom_state
    assert "Q4 1:12" in custom_state
    assert "Keep the offense on the field" in custom_state
    assert "Approx. 55-yard FG" in custom_state
    assert "Expected Value by Call" in custom_baseline
    assert reloaded_custom == [custom]
    assert edited_library == [edited_custom]
    assert edited_values[0] == "Edited Fourth Down"
    assert edited_values[8] == "0:48"
    assert edited_values[11:13] == ("defense", 36.0)
    assert edited_values[-2:] == (17.5, 1.25)
    assert remaining_library == []
    assert load_custom_scenarios(custom_path) == []
    assert scenarios_for_library("custom", [*payloads, custom.model_dump(mode="json")]) == [custom]
    assert custom.name in custom.display_name
    assert "Head Coach's Call" in events[-1][2]
    assert "Call Sent to the Huddle" in events[-1][2]
    assert "Win Probability Added" in events[-1][3]
    assert "Kick the field goal" in field_goal_events[-1][2]
    assert 'class="play-call__goalpost"' in field_goal_events[-1][2]
    assert "🦵🥅" not in field_goal_events[-1][2]
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
    assert "scenario-library-selector" in str(app.config)
    assert "My Situations" in str(app.config)
    assert "open-custom-situation" in str(app.config)
    assert "custom-situation-modal" in str(app.config)
    assert "custom-situation-form-body" in str(app.config)
    assert "custom-analytics-overrides" in str(app.config)
    assert "New Situation" in str(app.config)
    assert "Edit Selected" in str(app.config)
    assert "Delete Selected" in str(app.config)
    assert "Delete Permanently" in str(app.config)
    assert "delete-situation-modal" in str(app.config)
    assert "Save to My Situations" in str(app.config)
    assert "decision-dashboard" in str(app.config)
    assert "analytics-booth" in str(app.config)
    assert "sideline-analytics" in str(app.config)
    assert "Sideline Connection & Model Settings" in str(app.config)
    assert "default-foundry-deployment" in str(app.config)
    assert "https://example.services.ai.azure.com/openai/v1/" in str(app.config)
    assert "live-play-call-status" in str(app.config)
    assert "coaches-meeting-transcript" in str(app.config)
    assert "headset-timeline--empty" in str(app.config)
    assert ".headset-message__bubble" in stylesheet
    assert ".headset-pending__dots" in stylesheet
    assert '#send-play-call' in browser_script
    assert '#coaches-meeting-transcript' in browser_script
    assert 'new MutationObserver(scheduleFollow)' in browser_script
    assert 'feed.scrollTop = feed.scrollHeight' in browser_script
    assert 'transcript.scrollIntoView' in browser_script
    assert 'fitHeadsetToViewport' in browser_script
    assert 'availableHeight' in browser_script
    assert 'max-height: calc(100dvh - 2rem)' in stylesheet
    assert 'max-height: min(38rem, calc(100dvh - 7rem))' in stylesheet
    assert browser_script.startswith('(() => {')
    assert browser_script.rstrip().endswith('})();')
    assert scenarios[0].display_name in str(app.config)
    assert scenarios[0].scenario_id not in scenarios[0].display_name
    assert result.exit_code == 0
    assert launches == [True]

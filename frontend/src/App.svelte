<script lang="ts">
  import { onMount, tick } from "svelte";
  import { deleteScenario, getScenarios, getSettings, saveScenario, startDesktopBackend, streamDeliberation } from "./api";
  import type {
    Action,
    ActionValue,
    AppSettings,
    CoachingEvent,
    CustomScenarioInput,
    DecisionTrace,
    Recommendation,
    Scenario,
    ScenarioEnvelope
  } from "./types";

  type HeadsetItem = {
    kind: "coach" | "phase" | "warning" | "head_coach";
    label: string;
    text: string;
    detail?: string;
    call?: string;
  };

  const actionLabels: Record<Action, string> = {
    run: "Run the ball",
    pass: "Drop back to pass",
    punt: "Send out the punt team",
    field_goal: "Kick the field goal",
    go_for_it: "Keep the offense on the field"
  };
  const roleLabels: Record<string, string> = {
    offensive_coordinator: "Offensive Coordinator",
    defensive_coordinator: "Defensive Coordinator",
    analytics_assistant: "Analytics Assistant",
    clock_management_specialist: "Clock Management",
    critical_reviewer: "Quality Control Coach"
  };
  const roleInitials: Record<string, string> = {
    offensive_coordinator: "OC",
    defensive_coordinator: "DC",
    analytics_assistant: "AN",
    clock_management_specialist: "CK",
    critical_reviewer: "QC",
    head_coach: "HC"
  };

  let scenarios: ScenarioEnvelope[] = [];
  let library: "prebuilt" | "custom" = "prebuilt";
  let selectedId = "";
  let settings: AppSettings | null = null;
  let strategy: "expected_points" | "single_agent" | "multi_agent" = "multi_agent";
  let busy = false;
  let status = "Waiting for Coach to put in the call…";
  let headset: HeadsetItem[] = [];
  let trace: DecisionTrace | null = null;
  let score: ActionValue | null = null;
  let error = "";
  let feed: HTMLDivElement;
  let modalOpen = false;
  let editingId: string | undefined;
  let lastSelectedId = "";
  let form: CustomScenarioInput = emptyForm();

  $: filteredScenarios = scenarios.filter((item) => item.library === library);
  $: selectedEnvelope = scenarios.find((item) => item.scenario.scenario_id === selectedId);
  $: selected = selectedEnvelope?.scenario;
  $: if (selectedId && selectedId !== lastSelectedId) {
    lastSelectedId = selectedId;
    resetAnalysis("New situation is on the call sheet.");
  }

  function emptyForm(): CustomScenarioInput {
    return {
      name: "Two-Minute Decision",
      season: 2025,
      week: 1,
      possession_team: "BUF",
      defensive_team: "KC",
      possession_score: 20,
      defensive_score: 20,
      quarter: 4,
      clock: "2:00",
      down: 4,
      yards_to_go: 2,
      field_side: "defense",
      yard_line: 35,
      possession_timeouts: 2,
      defensive_timeouts: 2,
      win_probability_percent: null,
      expected_points: null
    };
  }

  function resetAnalysis(message = "Waiting for Coach to put in the call…") {
    status = message;
    headset = [];
    trace = null;
    score = null;
    error = "";
  }

  function pickLibrary(next: "prebuilt" | "custom") {
    library = next;
    const first = scenarios.find((item) => item.library === next);
    if (first) selectedId = first.scenario.scenario_id;
  }

  function clockDisplay(scenario: Scenario): string {
    const state = scenario.state;
    const quarterSeconds = Math.max(0, Math.min(900, state.game_seconds_remaining - Math.max(0, 4 - state.quarter) * 900));
    return `Q${state.quarter} ${Math.floor(quarterSeconds / 60)}:${String(quarterSeconds % 60).padStart(2, "0")}`;
  }

  function downDistance(scenario: Scenario): string {
    const state = scenario.state;
    const ordinal = ["", "1st", "2nd", "3rd", "4th"][state.down];
    const distance = state.yardline_100 <= 10 && Math.abs(state.yards_to_go - state.yardline_100) < 0.1 ? "Goal" : `${state.yards_to_go}`;
    return `${ordinal} & ${distance}`;
  }

  function fieldPosition(scenario: Scenario): string {
    const state = scenario.state;
    if (state.yardline_100 === 50) return "the 50-yard line";
    return state.yardline_100 < 50
      ? `the ${state.defensive_team} ${state.yardline_100}`
      : `the ${state.possession_team} ${100 - state.yardline_100}`;
  }

  function gameSituation(scenario: Scenario): string {
    const differential = scenario.state.possession_score - scenario.state.defensive_score;
    return differential > 0 ? `Leading by ${differential}` : differential < 0 ? `Trailing by ${Math.abs(differential)}` : "Tie game";
  }

  function timeoutDots(remaining: number): boolean[] {
    return [0, 1, 2].map((index) => index < remaining);
  }

  function addCoachMessage(recommendation: Recommendation, revision: boolean) {
    headset = [
      ...headset,
      {
        kind: "coach",
        label: roleLabels[recommendation.role] ?? recommendation.role,
        call: actionLabels[recommendation.decision.action],
        text: recommendation.argument,
        detail: revision ? recommendation.rebuttal : recommendation.concerns?.join(" · ")
      }
    ];
  }

  async function followLatest() {
    await tick();
    if (feed) feed.scrollTop = feed.scrollHeight;
  }

  async function handleEvent(event: CoachingEvent) {
    status = event.message;
    if (event.recommendation) addCoachMessage(event.recommendation, false);
    if (event.revision) addCoachMessage(event.revision, true);
    if (event.failure) {
      headset = [...headset, { kind: "warning", label: "Headset Check", text: event.failure }];
    }
    if (event.stage === "recommendations") {
      headset = [...headset, { kind: "phase", label: "Challenge Round", text: "The staff tests the opening calls against clock, risk, and game context." }];
    } else if (event.stage === "debate") {
      headset = [...headset, { kind: "phase", label: "Head Coach", text: "The revised calls are in. Coach is breaking the tie." }];
    }
    if (event.stage === "completed" && event.trace && event.score) {
      trace = event.trace;
      score = event.score;
      headset = [
        ...headset,
        {
          kind: "head_coach",
          label: "Head Coach",
          call: event.trace.decision.action === "go_for_it" && event.trace.decision.go_for_it_play
            ? `${actionLabels.go_for_it} — ${actionLabels[event.trace.decision.go_for_it_play]}`
            : actionLabels[event.trace.decision.action],
          text: event.trace.decision.rationale
        }
      ];
    }
    if (event.stage === "error") error = event.message;
    await followLatest();
  }

  async function sendCall() {
    if (!selected || !settings || busy) return;
    resetAnalysis("Breaking the huddle and getting the call sheet ready…");
    busy = true;
    headset = [{ kind: "phase", label: "Opening Headset Check", text: "Each coach gets one clean turn before the staff challenges the call." }];
    await tick();
    document.getElementById("coaches-headset")?.scrollIntoView({ behavior: "smooth", block: "start" });
    try {
      await streamDeliberation(
        {
          scenario_id: selected.scenario_id,
          strategy,
          provider: settings.provider,
          model: settings.model,
          base_url: settings.base_url,
          upstream_url: settings.upstream_url || null,
          model_license: settings.model_license,
          reasoning_effort: settings.reasoning_effort || null
        },
        (event) => void handleEvent(event)
      );
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
      status = "The headset connection dropped before the call reached the huddle.";
    } finally {
      busy = false;
    }
  }

  function openNew() {
    editingId = undefined;
    form = emptyForm();
    modalOpen = true;
  }

  function openEdit() {
    if (!selected || selectedEnvelope?.library !== "custom") return;
    const state = selected.state;
    const seconds = state.game_seconds_remaining - (4 - state.quarter) * 900;
    const fieldSide = state.yardline_100 === 50 ? "midfield" : state.yardline_100 < 50 ? "defense" : "offense";
    editingId = selected.scenario_id;
    form = {
      name: selected.name ?? "Saved Situation",
      season: state.season,
      week: state.week,
      possession_team: state.possession_team,
      defensive_team: state.defensive_team,
      possession_score: state.possession_score,
      defensive_score: state.defensive_score,
      quarter: state.quarter,
      clock: `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`,
      down: state.down,
      yards_to_go: state.yards_to_go,
      field_side: fieldSide,
      yard_line: fieldSide === "offense" ? 100 - state.yardline_100 : fieldSide === "defense" ? state.yardline_100 : 25,
      possession_timeouts: state.possession_timeouts,
      defensive_timeouts: state.defensive_timeouts,
      win_probability_percent: state.win_probability * 100,
      expected_points: state.expected_points
    };
    modalOpen = true;
  }

  async function submitScenario() {
    error = "";
    try {
      const saved = await saveScenario(form, editingId);
      scenarios = await getScenarios();
      library = "custom";
      selectedId = saved.scenario.scenario_id;
      modalOpen = false;
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    }
  }

  async function removeSelected() {
    if (!selected || selectedEnvelope?.library !== "custom" || !confirm(`Delete “${selected.name ?? "this situation"}”?`)) return;
    await deleteScenario(selected.scenario_id);
    scenarios = await getScenarios();
    selectedId = scenarios.find((item) => item.library === "custom")?.scenario.scenario_id ?? "";
  }

  onMount(async () => {
    try {
      await startDesktopBackend();
      [scenarios, settings] = await Promise.all([getScenarios(), getSettings()]);
      selectedId = scenarios.find((item) => item.library === "prebuilt")?.scenario.scenario_id ?? scenarios[0]?.scenario.scenario_id ?? "";
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught);
    }
  });
</script>

<svelte:head><title>NFL Virtual Coaching Staff</title></svelte:head>

<main class="app-shell">
  <header class="app-header">
    <div>
      <p class="eyebrow">NFL Decision Lab</p>
      <h1>NFL Virtual Coaching Staff</h1>
      <p>Put the situation on the call sheet, hear every coordinator, and see what the head coach sends in.</p>
    </div>
    <details class="settings-panel">
      <summary>Sideline Connection &amp; Model Settings</summary>
      {#if settings}
        <div class="settings-grid">
          <label><span>Provider <span class="required-marker" aria-hidden="true">*</span></span><input required bind:value={settings.provider} /></label>
          <label><span>Model deployment <span class="required-marker" aria-hidden="true">*</span></span><input required bind:value={settings.model} /></label>
          <label><span>Endpoint <span class="required-marker" aria-hidden="true">*</span></span><input required bind:value={settings.base_url} /></label>
          <label>Upstream model URL<input bind:value={settings.upstream_url} /></label>
          <label>Model license<input bind:value={settings.model_license} /></label>
          <label>Reasoning effort<input bind:value={settings.reasoning_effort} /></label>
        </div>
        <p class:configured={settings.api_key_configured} class="key-state">
          {settings.api_key_configured ? "Foundry credential detected in the environment" : "No Foundry credential detected"}
        </p>
      {/if}
    </details>
  </header>

  {#if error}<div class="error-banner" role="alert">⚠ {error}</div>{/if}

  <section class="game-plan-grid">
    <div class="left-rail">
      <section class="picker-card">
        <div class="section-heading">
          <span>Game Situation</span>
          <div class="segment-control" aria-label="Scenario library">
            <button class:active={library === "prebuilt"} on:click={() => pickLibrary("prebuilt")}>Pre-Built</button>
            <button class:active={library === "custom"} on:click={() => pickLibrary("custom")}>My Situations</button>
          </div>
        </div>
        <div class="picker-row">
          <select bind:value={selectedId} aria-label="Game situation">
            {#each filteredScenarios as item}<option value={item.scenario.scenario_id}>{item.display_name}</option>{/each}
          </select>
          <button class="secondary" on:click={openNew}>＋ New Situation</button>
        </div>
        {#if library === "custom" && selected}
          <div class="management-row">
            <button class="quiet" on:click={openEdit}>Edit situation</button>
            <button class="quiet danger" on:click={removeSelected}>Delete situation</button>
          </div>
        {/if}
      </section>

      {#if selected}
        {@const state = selected.state}
        {@const los = Math.max(0, Math.min(100, 100 - state.yardline_100))}
        {@const gain = Math.max(0, Math.min(100, los + state.yards_to_go))}
        <section class="card situation-card">
          <header class="card-header">
            <div><span class="eyebrow">Pre-Snap Situation Data</span><h2>Sideline Tablet</h2></div>
            <span class="tag">{state.season} · Week {state.week}</span>
          </header>
          <div class="scoreboard">
            <div class="team offense">
              <small>◆ Ball</small><div><b>{state.possession_team}</b><strong>{state.possession_score}</strong></div>
              <span>Timeouts {#each timeoutDots(state.possession_timeouts) as available}<i class:used={!available}></i>{/each}</span>
            </div>
            <div class="game-clock"><b>{clockDisplay(selected)}</b><strong>{downDistance(selected)}</strong><small>{gameSituation(selected)}</small></div>
            <div class="team defense">
              <small>Defense</small><div><b>{state.defensive_team}</b><strong>{state.defensive_score}</strong></div>
              <span>Timeouts {#each timeoutDots(state.defensive_timeouts) as available}<i class:used={!available}></i>{/each}</span>
            </div>
          </div>
          <div class="field-wrap">
            <div class="football-field" role="img" aria-label={`${state.possession_team} ball on ${fieldPosition(selected)}, driving toward the ${state.defensive_team} end zone`}>
              <div class="end-zone offense-zone"><span>{state.possession_team}</span></div>
              <div class="playing-field">
                {#each [10,20,30,40,50,60,70,80,90] as yard}<span class="yard" style={`left:${yard}%`}>{Math.min(yard, 100-yard)}</span>{/each}
                <span class="field-line los" style={`left:${los}%`}><b>LOS</b></span>
                <span class="field-line gain" style={`left:${gain}%`}><b>TO GAIN</b></span>
                <span class="football" style={`left:${los}%`}></span>
              </div>
              <div class="end-zone defense-zone"><span>{state.defensive_team}</span></div>
            </div>
            <div class="field-legend">
              <strong>Ball on {fieldPosition(selected)}</strong>
              <span class="drive-badge">{state.possession_team} driving →</span>
              <span><i class="blue"></i>Line of scrimmage</span><span><i class="gold"></i>Line to gain</span>
            </div>
          </div>
          <div class="decision-strip">
            <div><span>Down &amp; Distance</span><b>{downDistance(selected)}</b></div>
            <div><span>Field Position</span><b>{fieldPosition(selected)}</b></div>
            <div><span>Game Situation</span><b>{gameSituation(selected)}</b></div>
            <div><span>Expected Points</span><b>{state.expected_points >= 0 ? "+" : ""}{state.expected_points.toFixed(3)}</b></div>
          </div>
          <div class="probability"><span>Offense Win Probability <b>{(state.win_probability * 100).toFixed(1)}%</b></span><div><i style={`width:${state.win_probability*100}%`}></i></div></div>
          <div class="legal-calls"><span>Calls Available</span><div>{#each Object.keys(selected.ep_baseline) as action}<b>{actionLabels[action as Action]}</b>{/each}</div></div>
        </section>
      {/if}
    </div>

    <aside class="right-rail">
      <section class="mode-card">
        <span>Who’s Making the Calls?</span>
        <div class="segment-control modes">
          <button class:active={strategy === "expected_points"} on:click={() => strategy = "expected_points"}>Analytics booth only</button>
          <button class:active={strategy === "single_agent"} on:click={() => strategy = "single_agent"}>Head Coach only</button>
          <button class:active={strategy === "multi_agent"} on:click={() => strategy = "multi_agent"}>Full Coaching Staff</button>
        </div>
      </section>
      {#if selected}
        <section class="card analytics-card">
          <header class="card-header"><div><span class="eyebrow">Analytics Booth</span><h2>Expected Value by Call</h2></div><span class="tag">EPA</span></header>
          <table><thead><tr><th>Call Sheet Option</th><th>Expected EPA</th></tr></thead><tbody>
            {#each Object.entries(selected.ep_baseline).sort((a,b) => b[1]-a[1]) as [action, value], index}
              <tr class:best={index === 0}><td>{actionLabels[action as Action]} {#if index === 0}<small>Top option</small>{/if}</td><td class:positive={value >= 0}>{value >= 0 ? "+" : ""}{value.toFixed(3)}</td></tr>
            {/each}
          </tbody></table>
          <p>Higher EPA indicates stronger expected scoreboard value from this down, distance, field position, and game clock.</p>
        </section>
      {/if}
      <button class="send-call" disabled={busy || !selected} on:click={sendCall}>{busy ? "Coaches are on the headset…" : "🏈 Send in the Call! 🏈"}</button>
      <section class="live-status"><span>Live Sideline</span><strong>{status}</strong></section>
    </aside>
  </section>

  <section class="card headset-card" id="coaches-headset">
    <header class="card-header"><div><span class="eyebrow">Coaches’ Headset</span><h2>Live Staff Conversation</h2></div><span class:live-indicator={busy} class="tag headset-state" aria-live="polite">{#if busy}<i aria-hidden="true"></i>Live{:else}{trace ? "Final" : "Ready"}{/if}</span></header>
    <div class="headset-feed" bind:this={feed}>
      {#if headset.length === 0}<div class="empty-state"><b>Headsets are quiet</b><span>Send in the call to hear the coaching staff work the situation.</span></div>{/if}
      {#each headset as item}
        {#if item.kind === "phase"}
          <div class="phase-divider"><span>{item.label}</span><p>{item.text}</p></div>
        {:else}
          <article class:warning={item.kind === "warning"} class:head-coach={item.kind === "head_coach"} class="headset-message">
            <div class="avatar">{roleInitials[item.kind === "head_coach" ? "head_coach" : Object.keys(roleLabels).find((key) => roleLabels[key] === item.label) ?? ""] ?? "!"}</div>
            <div class="message-body"><header><b>{item.label}</b>{#if item.call}<span>{item.call}</span>{/if}</header><p>{item.text}</p>{#if item.detail}<details><summary>Call-sheet notes</summary><p>{item.detail}</p></details>{/if}</div>
          </article>
        {/if}
      {/each}
    </div>
  </section>

  <section class="results-grid">
    <section class="card result-card">
      <header class="card-header"><div><span class="eyebrow">Head Coach’s Call</span><h2>{trace ? (strategy === "multi_agent" ? "Full Coaching Staff" : strategy === "single_agent" ? "Head Coach Only" : "Analytics Booth") : "Waiting on the sideline"}</h2></div></header>
      {#if trace}<div class="call-result"><span>Call sent to the huddle</span><strong>{actionLabels[trace.decision.action]}</strong><p>{trace.decision.rationale}</p><footer>{trace.model_id ?? "Deterministic EPA policy"} · {trace.model_calls} model calls · {trace.latency_seconds.toFixed(2)}s</footer></div>{:else}<div class="empty-state"><span>Send in the call to see the selected play and the staff’s reasoning.</span></div>{/if}
    </section>
    <section class="card result-card">
      <header class="card-header"><div><span class="eyebrow">Postgame Decision Grade</span><h2>Simulator Review</h2></div>{#if score}<span class="tag">v{score.simulator_version}</span>{/if}</header>
      {#if score}
        <div class:strong={score.oracle_regret < 0.01} class="grade-banner"><b>{score.oracle_regret < 0.01 ? "Strong Call" : score.oracle_regret < 0.04 ? "Competitive Call" : "Costly Call"}</b><span>{score.oracle_regret < 0.01 ? "Minimal gap from the top option" : `Gap from the best call: ${(score.oracle_regret*100).toFixed(2)} percentage points`}</span></div>
        <div class="grade-grid"><div><span>Win Probability Added</span><b>{score.expected_wpa >= 0 ? "+" : ""}{(score.expected_wpa*100).toFixed(2)}%</b></div><div><span>Expected Points Added</span><b>{score.expected_epa >= 0 ? "+" : ""}{score.expected_epa.toFixed(3)}</b></div><div><span>Uncertainty</span><b>±{(score.uncertainty*100).toFixed(1)}%</b></div><div><span>Gap From Best Call</span><b>{(score.oracle_regret*100).toFixed(2)} pts</b></div></div>
        <p class="decision-gap-note"><b>Lower is better.</b> This is the win-probability gap between the call sent in and the simulator’s top-rated legal call. A 0.00-point gap means they matched.</p>
      {:else}<div class="empty-state"><span>The simulator will grade the call after it reaches the huddle.</span></div>{/if}
    </section>
  </section>
</main>

{#if modalOpen}
  <div class="modal-backdrop" role="presentation" on:click={(event) => event.currentTarget === event.target && (modalOpen = false)}>
    <div class="scenario-dialog" role="dialog" aria-modal="true" aria-label="Custom game situation">
      <header><div><span class="eyebrow">My Situations</span><h2>{editingId ? "Edit Game Situation" : "Build a New Game Situation"}</h2></div><button class="icon-button" on:click={() => modalOpen = false}>×</button></header>
      <form on:submit|preventDefault={submitScenario}>
        <label class="full">Situation name<input required bind:value={form.name} /></label>
        <label>Season<input type="number" min="2000" max="2100" bind:value={form.season} /></label><label>Week<input type="number" min="1" max="22" bind:value={form.week} /></label>
        <label>Offense<input required maxlength="4" bind:value={form.possession_team} /></label><label>Defense<input required maxlength="4" bind:value={form.defensive_team} /></label>
        <label>Offense score<input type="number" min="0" bind:value={form.possession_score} /></label><label>Defense score<input type="number" min="0" bind:value={form.defensive_score} /></label>
        <label>Quarter<select bind:value={form.quarter}>{#each [1,2,3,4] as quarter}<option value={quarter}>{quarter}</option>{/each}</select></label><label>Game clock<input required pattern="[0-9][0-9]?:[0-5][0-9]" bind:value={form.clock} /></label>
        <label>Down<select bind:value={form.down}>{#each [1,2,3,4] as down}<option value={down}>{down}</option>{/each}</select></label><label>Yards to go<input type="number" min="0.1" max="99" step="0.1" bind:value={form.yards_to_go} /></label>
        <label>Field side<select bind:value={form.field_side}><option value="offense">Offense’s side</option><option value="midfield">Midfield</option><option value="defense">Defense’s side</option></select></label><label>Yard line<input type="number" min="1" max="49" disabled={form.field_side === "midfield"} bind:value={form.yard_line} /></label>
        <label>Offense timeouts<input type="number" min="0" max="3" bind:value={form.possession_timeouts} /></label><label>Defense timeouts<input type="number" min="0" max="3" bind:value={form.defensive_timeouts} /></label>
        <details class="full overrides"><summary>Optional analytics overrides</summary><div><label>Win probability %<input type="number" min="0" max="100" step="0.1" bind:value={form.win_probability_percent} /></label><label>Expected points<input type="number" step="0.001" bind:value={form.expected_points} /></label></div></details>
        <footer class="full"><button type="button" class="secondary" on:click={() => modalOpen = false}>Cancel</button><button type="submit" class="primary">{editingId ? "Update Situation" : "Save to My Situations"}</button></footer>
      </form>
    </div>
  </div>
{/if}

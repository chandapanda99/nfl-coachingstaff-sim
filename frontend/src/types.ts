export type Action = "run" | "pass" | "punt" | "field_goal" | "go_for_it";

export interface GameState {
  game_id: string;
  play_id: number;
  season: number;
  week: number;
  quarter: number;
  game_seconds_remaining: number;
  down: number;
  yards_to_go: number;
  yardline_100: number;
  possession_team: string;
  defensive_team: string;
  possession_score: number;
  defensive_score: number;
  possession_timeouts: number;
  defensive_timeouts: number;
  win_probability: number;
  expected_points: number;
}

export interface Scenario {
  scenario_id: string;
  state: GameState;
  ep_baseline: Record<Action, number>;
  source: string;
  source_license: string;
  name?: string;
}

export interface ScenarioEnvelope {
  scenario: Scenario;
  display_name: string;
  library: "prebuilt" | "custom";
}

export interface AppSettings {
  provider: string;
  model: string;
  base_url: string;
  upstream_url: string;
  model_license: string;
  reasoning_effort?: string;
  api_key_configured: boolean;
}

export interface Decision {
  action: Action;
  go_for_it_play?: Action;
  rationale: string;
}

export interface Recommendation {
  role: string;
  decision: Decision;
  confidence: number;
  argument: string;
  concerns: string[];
  rebuttal?: string;
}

export interface DecisionTrace {
  strategy: string;
  decision: Decision;
  model_id?: string;
  latency_seconds: number;
  model_calls: number;
  failures: string[];
  fallback_used: boolean;
}

export interface ActionValue {
  simulator_version: string;
  decision: Decision;
  expected_wpa: number;
  expected_epa: number;
  uncertainty: number;
  oracle_regret: number;
}

export interface CoachingEvent {
  stage: string;
  message: string;
  role?: string;
  recommendation?: Recommendation;
  revision?: Recommendation;
  failure?: string;
  trace?: DecisionTrace;
  score?: ActionValue;
}

export interface CustomScenarioInput {
  name: string;
  season: number;
  week: number;
  possession_team: string;
  defensive_team: string;
  possession_score: number;
  defensive_score: number;
  quarter: number;
  clock: string;
  down: number;
  yards_to_go: number;
  field_side: "offense" | "midfield" | "defense";
  yard_line: number;
  possession_timeouts: number;
  defensive_timeouts: number;
  win_probability_percent: number | null;
  expected_points: number | null;
}

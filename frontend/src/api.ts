import type { AppSettings, CoachingEvent, CustomScenarioInput, ScenarioEnvelope } from "./types";

const apiBase = window.__TAURI_INTERNALS__ ? "http://127.0.0.1:8765" : "";
let backendChild: { kill: () => Promise<void> } | undefined;

export async function startDesktopBackend(): Promise<void> {
  if (!window.__TAURI_INTERNALS__ || import.meta.env.DEV || backendChild) return;
  const { Command } = await import("@tauri-apps/plugin-shell");
  backendChild = await Command.sidecar("binaries/nfl-coach-backend", ["serve", "--host", "127.0.0.1", "--port", "8765"]).spawn();
  window.addEventListener("beforeunload", () => void backendChild?.kill());
  for (let attempt = 0; attempt < 240; attempt += 1) {
    try {
      const response = await fetch(`${apiBase}/api/health`);
      if (response.ok) return;
    } catch {
      // The packaged Python runtime is still warming up.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("The local coaching service did not report ready.");
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, init);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail ?? response.statusText);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export const getSettings = () => json<AppSettings>("/api/settings");
export const getScenarios = () => json<ScenarioEnvelope[]>("/api/scenarios");

export function saveScenario(values: CustomScenarioInput, scenarioId?: string): Promise<ScenarioEnvelope> {
  return json<ScenarioEnvelope>(scenarioId ? `/api/scenarios/${scenarioId}` : "/api/scenarios", {
    method: scenarioId ? "PUT" : "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values)
  });
}

export const deleteScenario = (scenarioId: string) => json<void>(`/api/scenarios/${scenarioId}`, { method: "DELETE" });

export async function streamDeliberation(
  body: Record<string, unknown>,
  onEvent: (event: CoachingEvent) => void
): Promise<void> {
  const response = await fetch(`${apiBase}/api/deliberations/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!response.ok || !response.body) throw new Error(`The headset connection failed (${response.status}).`);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let pending = "";
  while (true) {
    const { value, done } = await reader.read();
    pending += value ?? "";
    const lines = pending.split("\n");
    pending = lines.pop() ?? "";
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
    if (done) break;
  }
  if (pending.trim()) onEvent(JSON.parse(pending));
}

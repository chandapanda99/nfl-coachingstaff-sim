"""Deterministic counterfactual EPA/WPA evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import Lock
from typing import Any

from nfl_coaching_sim.data import observed_action
from nfl_coaching_sim.models import Action, ActionValue, Decision, GameState, Scenario

ACTIONS = tuple(Action)


def _features(state: GameState, action: Action) -> list[float]:
    return [
        state.quarter,
        state.game_seconds_remaining / 1800,
        state.down,
        state.yards_to_go / 10,
        state.yardline_100 / 100,
        state.score_differential / 16,
        state.possession_timeouts / 3,
        state.defensive_timeouts / 3,
        state.win_probability,
        state.expected_points / 7,
        *[1.0 if action == candidate else 0.0 for candidate in ACTIONS],
    ]


class DeterministicSimulator:
    """Scores legal choices with fixed-seed regressors or an offline heuristic."""

    def __init__(
        self,
        wpa_model: Any | None = None,
        epa_model: Any | None = None,
        residual_uncertainty: float = 0.08,
    ) -> None:
        self.wpa_model = wpa_model
        self.epa_model = epa_model
        self.residual_uncertainty = residual_uncertainty

    @classmethod
    def deferred(cls, path: Path) -> DeterministicSimulator:
        """Create a scorer that loads its trained artifact on first use."""

        simulator = cls()
        simulator._deferred_path = path
        simulator._deferred_lock = Lock()
        return simulator

    def _ensure_loaded(self) -> None:
        path = getattr(self, "_deferred_path", None)
        if path is None:
            return
        lock = getattr(self, "_deferred_lock", None)
        if lock is None:
            lock = Lock()
            self._deferred_lock = lock
        with lock:
            path = getattr(self, "_deferred_path", None)
            if path is None:
                return
            loaded = type(self).load(path)
            self.wpa_model = loaded.wpa_model
            self.epa_model = loaded.epa_model
            self.residual_uncertainty = loaded.residual_uncertainty
            self._deferred_path = None
        self._deferred_lock = None

    @property
    def trained(self) -> bool:
        self._ensure_loaded()
        return self.wpa_model is not None and self.epa_model is not None

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> DeterministicSimulator:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingRegressor

        self._deferred_path = None
        from nfl_coaching_sim.data import is_late_game_candidate, state_from_row

        features: list[list[float]] = []
        wpa: list[float] = []
        epa: list[float] = []
        for row in rows:
            result = observed_action(row)
            if result is None or row.get("wpa") is None or row.get("epa") is None or not is_late_game_candidate(row):
                continue
            action, _ = result
            features.append(_features(state_from_row(row), action))
            wpa.append(float(row["wpa"]))
            epa.append(float(row["epa"]))
        if len(features) < 20:
            raise ValueError("at least 20 eligible training plays are required")
        x = np.asarray(features, dtype=float)
        self.wpa_model = HistGradientBoostingRegressor(max_iter=150, max_depth=5, learning_rate=0.06, random_state=2026).fit(
            x, np.asarray(wpa)
        )
        self.epa_model = HistGradientBoostingRegressor(max_iter=150, max_depth=5, learning_rate=0.06, random_state=2026).fit(
            x, np.asarray(epa)
        )
        wpa_residuals = np.asarray(wpa) - self.wpa_model.predict(x)
        self.residual_uncertainty = float(np.std(wpa_residuals))
        return self

    def _predict(self, scenario: Scenario, action: Action) -> tuple[float, float, float]:
        state = scenario.state
        if self.trained:
            import numpy as np

            x = np.asarray([_features(state, action)], dtype=float)
            return (
                float(self.wpa_model.predict(x)[0]),
                float(self.epa_model.predict(x)[0]),
                self.residual_uncertainty,
            )

        epa = scenario.ep_baseline[action]
        clock_pressure = max(0.0, (600 - state.game_seconds_remaining) / 600)
        trailing = state.score_differential < 0
        leading = state.score_differential > 0
        adjustment = 0.0
        if action == Action.RUN:
            adjustment += 0.012 * clock_pressure if leading else -0.008 * clock_pressure
        elif action == Action.PASS:
            adjustment += 0.012 * clock_pressure if trailing else -0.004 * clock_pressure
        elif action == Action.PUNT and trailing:
            adjustment -= 0.025 * clock_pressure
        elif action == Action.FIELD_GOAL and abs(state.score_differential) <= 3:
            adjustment += 0.015 * clock_pressure
        elif action == Action.GO_FOR_IT and trailing:
            adjustment += 0.02 * clock_pressure
        wpa = epa * (0.012 + 0.025 * clock_pressure) + adjustment
        uncertainty = 0.10 if action == Action.GO_FOR_IT else 0.07
        return wpa, epa, uncertainty

    def candidates(self, scenario: Scenario) -> dict[Action, tuple[float, float, float]]:
        return {action: self._predict(scenario, action) for action in scenario.state.legal_actions}

    def score(self, scenario: Scenario, decision: Decision) -> ActionValue:
        decision.validate_for(scenario.state)
        candidates = self.candidates(scenario)
        wpa, epa, uncertainty = candidates[decision.action]
        oracle = max(value[0] for value in candidates.values())
        return ActionValue(
            decision=decision,
            expected_wpa=wpa,
            expected_epa=epa,
            uncertainty=uncertainty,
            oracle_regret=max(0.0, oracle - wpa),
        )

    def save(self, path: Path) -> None:
        import joblib

        self._ensure_loaded()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> DeterministicSimulator:
        import joblib

        simulator = joblib.load(path)
        if not isinstance(simulator, cls):
            raise TypeError("artifact is not a DeterministicSimulator")
        return simulator

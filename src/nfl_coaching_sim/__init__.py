"""Multi-agent NFL coaching simulator."""

from nfl_coaching_sim.models import (
    Action,
    ActionValue,
    BenchmarkResult,
    DebateTranscript,
    Decision,
    DecisionTrace,
    GameState,
    Recommendation,
    Scenario,
)
from nfl_coaching_sim.simulator import DeterministicSimulator

__all__ = [
    "Action",
    "ActionValue",
    "BenchmarkResult",
    "DebateTranscript",
    "Decision",
    "DecisionTrace",
    "DeterministicSimulator",
    "GameState",
    "Recommendation",
    "Scenario",
]

__version__ = "0.1.0"

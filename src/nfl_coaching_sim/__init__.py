"""Multi-agent NFL coaching simulator."""

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
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


def __getattr__(name: str) -> Any:
    if name == "DeterministicSimulator":
        from nfl_coaching_sim.simulator import DeterministicSimulator

        return DeterministicSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

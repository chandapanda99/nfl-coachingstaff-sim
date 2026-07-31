"""Multi-agent NFL coaching simulator."""

from typing import TYPE_CHECKING, Any

from nfl_coaching_sim.football import build_situation_brief
from nfl_coaching_sim.models import (
    Action,
    ActionAssessment,
    ActionValue,
    BenchmarkResult,
    DebateTranscript,
    Decision,
    DecisionTrace,
    EvidenceItem,
    GameState,
    Recommendation,
    Scenario,
    SituationBrief,
)

if TYPE_CHECKING:
    from nfl_coaching_sim.simulator import DeterministicSimulator

__all__ = [
    "Action",
    "ActionAssessment",
    "ActionValue",
    "BenchmarkResult",
    "DebateTranscript",
    "Decision",
    "DecisionTrace",
    "DeterministicSimulator",
    "EvidenceItem",
    "GameState",
    "Recommendation",
    "Scenario",
    "SituationBrief",
    "build_situation_brief",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "DeterministicSimulator":
        from nfl_coaching_sim.simulator import DeterministicSimulator

        return DeterministicSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

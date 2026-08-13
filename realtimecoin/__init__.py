"""Real-time COIN motor-learning model (Python translation of ``@RealTimeCOIN``).

The COntextual INference model of Heald, Lengyel & Wolpert (2021), run as a
state machine: :meth:`~realtimecoin.model.RealTimeCOIN.observe_q` stages the cue
for the upcoming trial, :meth:`~realtimecoin.model.RealTimeCOIN.observe_y`
processes the feedback and advances the trial, and the query methods expose the
current posterior and predictive summaries.

Examples
--------
>>> from realtimecoin import RealTimeCOIN
>>> model = RealTimeCOIN(num_particles=50, rng=0)
>>> model.Trial
0
"""

from .ensemble import (
    EnsembleLockstepError,
    RealTimeCOINEnsemble,
    SimulationTraces,
)
from .exceptions import (
    BiasNotInferredError,
    ModelFormatError,
    NameValuePairsError,
    NoCuesError,
    RealTimeCOINError,
    ScalarModelOnlyError,
)
from .model import RealTimeCOIN
from .state import ParticleState

__version__ = "0.1.0"

__all__ = [
    "RealTimeCOIN",
    "RealTimeCOINEnsemble",
    "SimulationTraces",
    "ParticleState",
    "RealTimeCOINError",
    "NoCuesError",
    "BiasNotInferredError",
    "ScalarModelOnlyError",
    "NameValuePairsError",
    "ModelFormatError",
    "EnsembleLockstepError",
]

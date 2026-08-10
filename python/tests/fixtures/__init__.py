"""Hand-crafted particle-state fixtures for the test-suite.

Python counterpart of the MATLAB ``tests/+testutil`` fixture builders. Each
builder returns a fully controlled :class:`~realtimecoin.RealTimeCOIN` via
``RealTimeCOIN.from_state`` - the Python stand-in for
``testutil.loadFixtureModel`` - so queries can be driven from a known particle
configuration WITHOUT running ``observe_y``. That matters while the inference
pipelines are still being translated, and it is what gives the alignment tests a
verifiable ground truth.
"""

from .alignment_fixtures import (
    ALIGNMENT_GLOBAL_CUE,
    ALIGNMENT_GLOBAL_DRIFT,
    ALIGNMENT_GLOBAL_PREDICTED,
    ALIGNMENT_GLOBAL_RESPONSIBILITIES,
    ALIGNMENT_GLOBAL_RETENTION,
    ALIGNMENT_GLOBAL_STATE_MEAN,
    ALIGNMENT_GLOBAL_TRANSITION,
    MD_GLOBAL_DRIFT,
    MD_GLOBAL_RETENTION,
    MD_GLOBAL_STATE_MEAN,
    alignment_fixture,
    large_assignment_fixture,
    md_alignment_fixture,
)

__all__ = [
    "alignment_fixture",
    "large_assignment_fixture",
    "md_alignment_fixture",
    "ALIGNMENT_GLOBAL_STATE_MEAN",
    "ALIGNMENT_GLOBAL_RETENTION",
    "ALIGNMENT_GLOBAL_DRIFT",
    "ALIGNMENT_GLOBAL_CUE",
    "ALIGNMENT_GLOBAL_TRANSITION",
    "ALIGNMENT_GLOBAL_PREDICTED",
    "ALIGNMENT_GLOBAL_RESPONSIBILITIES",
    "MD_GLOBAL_STATE_MEAN",
    "MD_GLOBAL_RETENTION",
    "MD_GLOBAL_DRIFT",
]

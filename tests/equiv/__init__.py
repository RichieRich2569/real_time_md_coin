"""Cross-language equivalence battery (Python port of MATLAB ``tests/+equiv``).

The MATLAB package compares two MATLAB code versions bit-for-bit. That is
impossible across languages: MATLAB's Mersenne-Twister stream and NumPy's PCG64
stream cannot be made to emit the same variates, so a Python run and a MATLAB
run of the same scenario never follow the same particle trajectory.

This port therefore replaces the bit-identical (``classA``) test with a
DISTRIBUTIONAL one: each scenario is replayed under many independent seeds on
each side, and the two run-averages are compared inside a Monte-Carlo band. See
:mod:`tests.equiv.compare_runs` for the band definition.

Modules
-------
scenario_battery
    Scenario definitions; the (q, y) inputs are LOADED from the frozen MATLAB
    fixtures so both languages provably consume identical inputs.
capture_run
    Streams one scenario through Python ``RealTimeCOIN`` and records the same
    observable surface as ``equiv.captureRun``.
compare_runs
    Reduces captures to run-averages and applies the Monte-Carlo band.

Not ported: ``equiv.profileHot`` (a MATLAB profiler wrapper with no
cross-language meaning) and ``equiv.captureAll`` (a thin loop, inlined into
:func:`tests.equiv.compare_runs.python_run_average`).
"""

from __future__ import annotations

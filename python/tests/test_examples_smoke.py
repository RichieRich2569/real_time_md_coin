"""End-to-end smoke test for the example scripts in ``python/examples``.

The two example scripts (``scalar_examples.py`` and ``md_examples.py``) are the
Python counterparts of the MATLAB live scripts, and they are the only place the
plot helpers in ``examples/viz.py`` are exercised. This module runs each script
as a program, headless, with ``RTCOIN_EXAMPLES_FAST=1`` so every long block and
particle count shrinks by about 4x, and asserts that it completes without
raising and produces figures.

The scripts are slow even in fast mode (the filter costs milliseconds per
particle per trial), so everything here is marked ``slow``; run it with
``pytest -m slow`` or as part of a full ``pytest tests/``.

Both scripts guard their ensemble sections behind ``ImportError``, so this test
passes whether or not ``RealTimeCOINEnsemble`` is present.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

#: ``python/examples``, resolved from this file so the test is cwd-independent.
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

#: The scripts under test, in the order the notebooks reference each other.
SCRIPTS = ("scalar_examples.py", "md_examples.py")


def _has_matplotlib():
    """Whether matplotlib is importable in this interpreter."""
    try:
        import matplotlib   # noqa: F401
    except ImportError:
        return False
    return True


requires_matplotlib = pytest.mark.skipif(
    not _has_matplotlib(),
    reason="matplotlib is not installed (pip install matplotlib for the examples)",
)


def _import_viz():
    """Import ``examples/viz.py``, which is a loose module, not a package.

    ``examples/`` is deliberately not a package (the scripts are cell-mode
    scripts, not library code), so the directory goes on ``sys.path`` just long
    enough for the import. The module is left in ``sys.modules`` afterwards, so
    repeat calls are cheap and every test sees the same object.
    """
    if "viz" in sys.modules:
        return sys.modules["viz"]
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        import viz
    finally:
        sys.path.remove(str(EXAMPLES_DIR))
    return viz


@requires_matplotlib
def test_viz_module_imports():
    """``examples/viz.py`` imports and exposes the eight ``+coinviz`` helpers."""
    viz = _import_viz()

    for name in ("palette", "state_trace", "density_lines", "density_evolution",
                 "context_bars", "context_density_evolution", "trajectory_2d",
                 "density_heatmaps"):
        assert callable(getattr(viz, name)), name
    # The palette cycles the eight Okabe-Ito base colours.
    assert viz.palette().shape == (8, 3)
    assert viz.palette(10).shape == (10, 3)
    assert (viz.palette(10)[8:] == viz.palette(2)).all()


@requires_matplotlib
@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS)
def test_example_script_runs_headless(script, tmp_path):
    """Each example script runs end-to-end under Agg in fast mode.

    Runs in a subprocess so the ``matplotlib`` backend choice, the module-level
    ``sys.path`` edit each script performs and the dozens of open figures cannot
    leak into the rest of the suite.
    """
    path = EXAMPLES_DIR / script
    assert path.is_file(), path

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"            # headless: no display needed
    env["RTCOIN_EXAMPLES_FAST"] = "1"    # ~4x fewer trials and particles
    env["PYTHONWARNINGS"] = "ignore"
    env["MPLCONFIGDIR"] = str(tmp_path)  # keep the font cache out of $HOME
    # Running a script BY PATH puts the script's own directory on sys.path, not
    # the cwd, and pyproject's pythonpath setting only applies to the pytest
    # process. Put python/ on the child's path so a plain source checkout (no
    # `pip install -e .`) can still import realtimecoin.
    root = str(EXAMPLES_DIR.parent)
    env["PYTHONPATH"] = (
        root + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else root
    )

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(EXAMPLES_DIR.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, (
        "%s failed (exit %d)\n--- stdout ---\n%s\n--- stderr ---\n%s"
        % (script, result.returncode, result.stdout[-4000:], result.stderr[-4000:])
    )

    stdout = result.stdout
    # The setup cell must have reported fast mode, and the wrap-up cell must
    # have reported a non-zero figure count.
    assert "fast mode: True" in stdout, stdout[-2000:]
    stem = script[: -len(".py")]
    marker = "%s: done (" % stem
    assert marker in stdout, stdout[-2000:]
    n_figures = int(stdout.split(marker, 1)[1].split(" figures", 1)[0])
    assert n_figures > 5, "expected several figures, got %d" % n_figures


@requires_matplotlib
def test_example_figures_can_be_saved(tmp_path):
    """The helpers produce savable PNG figures from synthetic arrays.

    A fast structural check of ``viz`` that does not need a trained model: it
    catches shape and API mistakes in the helpers without paying for a run of
    the filter, so it is deliberately NOT marked slow.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    previous_backend = matplotlib.get_backend()
    plt.switch_backend("Agg")        # restored in the finally below
    try:
        _draw_and_save_every_helper(tmp_path, np)
    finally:
        plt.close("all")
        plt.switch_backend(previous_backend)


def _draw_and_save_every_helper(tmp_path, np):
    """Draw one figure per ``viz`` helper and write each to ``tmp_path``."""
    viz = _import_viz()

    n_trials, n_grid = 24, 41
    trials = np.arange(1, n_trials + 1)
    grid = np.linspace(-1.0, 1.0, n_grid)
    rng = np.random.default_rng(0)
    dens = np.abs(rng.random((n_grid, n_trials)))
    probs = rng.random((n_trials, 4))
    probs /= probs.sum(axis=1, keepdims=True)

    figures = [
        viz.state_trace(trials, rng.random((2, n_trials)),
                        rng.random((2, n_trials)), rng.random((2, n_trials)),
                        [8, 16], band=0.1 * np.ones((2, n_trials))),
        viz.density_lines(grid, {0: dens[:, 0], 1: dens[:, 1]},
                          novel_density=dens[:, 2]),
        viz.density_evolution(trials, grid, dens, true_line=np.zeros(n_trials),
                              pred_line=np.zeros(n_trials),
                              missing_trials=[5, 6], block_edges=[12]),
        viz.context_bars(probs, probs, novel_context=True),
        viz.context_bars(probs, None, mode="line"),
        viz.context_density_evolution(trials, grid, {0: dens, 2: dens[::-1]},
                                      novel_dens=0.1 * dens,
                                      true_line=np.zeros(n_trials),
                                      block_edges=[12]),
        viz.trajectory_2d(rng.random((2, n_trials)), rng.random((2, 3)),
                          observed_xy=rng.random((2, n_trials))),
    ]
    for i, fig in enumerate(figures):
        out = tmp_path / ("viz_%d.png" % i)
        fig.savefig(out, dpi=60)
        assert out.stat().st_size > 0


@requires_matplotlib
def test_context_bars_rejects_empty_input():
    """``context_bars`` needs at least one probability matrix."""
    viz = _import_viz()
    with pytest.raises(ValueError):
        viz.context_bars(None, None)
    with pytest.raises(ValueError):
        viz.context_bars([[1.0]], None, mode="bogus")

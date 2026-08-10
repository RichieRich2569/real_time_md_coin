"""Wall-clock benchmark of :class:`realtimecoin.RealTimeCOIN`.

Translated from ``validation/benchmark_realtimecoin.m``.

Times a fixed synthetic stream over a grid of particle counts. This is a
performance probe, not a scientific check: it reports timings and has NO pass
flag, exactly like the ``.m``.

Timing uses :func:`time.perf_counter` (MATLAB's ``tic``/``toc``), and a short
untimed warm-up run absorbs first-call import/allocation costs so they are not
charged to the first timed grid point.
"""

from __future__ import annotations

import time

import numpy as np

from realtimecoin import RealTimeCOIN

__all__ = ["run"]

_DEFAULTS = {
    "trials": 200,
    "particles": (50, 100, 250),
}


def run(seed=3001, **overrides):
    """Time the model over a particle-count grid.

    Parameters
    ----------
    seed : int, optional
        Seed for the synthetic stream and the models. Default 3001 (the MATLAB
        default).
    **overrides
        ``trials`` (200) and ``particles`` (``(50, 100, 250)``).

    Returns
    -------
    dict
        ``particles`` ``(G,)``, ``trials``, ``elapsed_seconds`` ``(G,)``,
        ``seconds_per_trial`` ``(G,)`` and ``config``. There is deliberately no
        ``passed`` key.
    """
    cfg = _config(seed, overrides)
    particles = cfg["particles"]
    trials = cfg["trials"]

    data_rng = np.random.default_rng(cfg["seed"])
    cues = 1 + (data_rng.random(trials) > 0.5).astype(int)
    signal = 0.3 * np.sin(np.arange(1, trials + 1) / 20.0)
    y = signal + 0.05 * data_rng.standard_normal(trials)

    # Warm-up (untimed): absorbs one-off allocation/JIT-like costs.
    warm_trials = min(trials, 10)
    warm = RealTimeCOIN(
        num_particles=int(particles[0]),
        max_contexts=5,
        # len(particles) + 1: distinct from every timed model below, and never
        # 0 - numpy's SeedSequence ignores trailing zero entropy words, so
        # [seed, 0] would collide with data_rng's plain `seed`.
        rng=np.random.default_rng([cfg["seed"], len(particles) + 1]),
    )
    for t in range(warm_trials):
        warm.observe_q(int(cues[t]))
        warm.observe_y(float(y[t]))

    elapsed = np.zeros(len(particles))
    for i, count in enumerate(particles):
        model = RealTimeCOIN(
            num_particles=int(count),
            max_contexts=5,
            rng=np.random.default_rng([cfg["seed"], i + 1]),
        )
        start = time.perf_counter()
        for t in range(trials):
            model.observe_q(int(cues[t]))
            model.observe_y(float(y[t]))
        elapsed[i] = time.perf_counter() - start
        print(
            "%d particles x %d trials: %.3f s" % (int(count), trials, elapsed[i])
        )

    return {
        "particles": np.asarray(particles, dtype=int),
        "trials": trials,
        "elapsed_seconds": elapsed,
        "seconds_per_trial": elapsed / trials,
        "config": cfg,
    }


def _config(seed, overrides):
    """Resolve the run configuration, rejecting unknown overrides.

    Parameters
    ----------
    seed : int
        Seed.
    overrides : dict
        Caller overrides; keys must be a subset of :data:`_DEFAULTS`.

    Returns
    -------
    dict
        The resolved configuration, with ``particles`` as a tuple of ints.

    Raises
    ------
    TypeError
        If an override key is not recognised.
    """
    unknown = set(overrides) - set(_DEFAULTS)
    if unknown:
        raise TypeError(
            "benchmark_realtimecoin.run() got unexpected keyword argument(s): "
            "%s." % ", ".join(sorted(unknown))
        )
    cfg = dict(_DEFAULTS)
    cfg.update(overrides)
    cfg["trials"] = int(cfg["trials"])
    cfg["particles"] = tuple(
        int(p) for p in np.atleast_1d(np.asarray(cfg["particles"])).reshape(-1)
    )
    cfg["seed"] = int(seed)
    return cfg


if __name__ == "__main__":       # pragma: no cover
    # Invoke as ``python -m validation.benchmark_realtimecoin`` from ``python/``:
    # this is a namespace package, so running the file by path would not put the
    # package root on sys.path.
    run()

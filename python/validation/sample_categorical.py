"""Draw one index from unnormalised categorical probabilities.

Translated from ``validation/validation_sample_categorical.m``.

Used by the validators' synthetic data generators to walk a context chain and
emit sensory cues. The MATLAB original draws from the global stream via
``rand``; here the generator is an explicit argument, because this package never
touches ``numpy.random``'s global state.

Returned labels are 0-BASED (MATLAB's are 1-based), matching the convention used
throughout the Python translation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sample_categorical"]


def sample_categorical(rng, weights):
    """Sample one 0-based index proportionally to ``weights``.

    Non-finite and negative weights are zeroed; if nothing positive is left the
    distribution degenerates to uniform, exactly as in MATLAB. The draw is
    ``find(u <= cumsum(p), 1)``, reproduced here with
    :func:`numpy.searchsorted` (``side="left"`` returns the first index whose
    cumulative mass reaches ``u``).

    Two edges deviate harmlessly from MATLAB, both by being safer:
    :meth:`numpy.random.Generator.random` can return exactly ``0.0`` where
    MATLAB's ``rand`` samples the open interval (with probability about
    ``2 ** -53`` this picks index 0 even when it has zero mass), and the result
    is clamped to the last index so float round-off in ``cumsum`` returns a
    valid label where MATLAB would return ``[]`` and error downstream.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random source; one uniform variate is consumed per call.
    weights : array_like
        Unnormalised probabilities; flattened before use.

    Returns
    -------
    int
        A 0-based index into the flattened ``weights``.
    """
    p = np.asarray(weights, dtype=float).reshape(-1)
    p = np.where(np.isfinite(p) & (p >= 0.0), p, 0.0)
    total = float(p.sum())
    if not total > 0.0:
        p = np.ones(p.size) / p.size
    else:
        p = p / total

    u = float(rng.random())
    cumulative = np.cumsum(p)
    # searchsorted(..., side="left") returns the first i with cumulative[i] >= u,
    # i.e. MATLAB's find(u <= cumsum(p), 1, 'first').
    idx = int(np.searchsorted(cumulative, u, side="left"))
    return min(idx, p.size - 1)   # guard against float round-off at the tail

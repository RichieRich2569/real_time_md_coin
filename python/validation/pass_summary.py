"""Aggregate named validation checks into one pass flag.

Translated from ``validation/validation_pass_summary.m``.

Every validator in this package builds a ``{name: value}`` mapping of named
checks and hands it to :func:`pass_summary`, which coerces each entry to a plain
``bool`` and ANDs them. Coercion is deliberately strict: a check that did not
produce an interpretable metric (``nan``, ``None``, a non-scalar) counts as a
FAILURE rather than being silently ignored, because a validator that could not
compute its metric has not demonstrated anything.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["pass_summary"]


def pass_summary(checks):
    """Coerce named checks to booleans and combine them.

    Parameters
    ----------
    checks : dict
        ``{check name: value}``. Values may be Python or numpy booleans, or
        numeric scalars (MATLAB's ``islogical`` / ``isnumeric`` branches).

    Returns
    -------
    passed : bool
        ``True`` only if every check coerced to ``True``.
    checks : dict
        A NEW dict with the same keys, in the same order, and plain ``bool``
        values. The input mapping is not mutated (MATLAB returns the coerced
        struct by value; this mirrors that).

    Notes
    -----
    Truth rule, transliterated from the ``.m``::

        logical scalar        -> its own value
        numeric scalar        -> isfinite(v) and v != 0
        anything else         -> False   (nan, None, arrays, strings, ...)

    Strings are rejected explicitly: ``isnumeric('1')`` is false in MATLAB, but
    ``numpy.asarray("1", dtype=float)`` would happily produce ``1.0``.

    Examples
    --------
    >>> pass_summary({"a": True, "b": 1.0, "c": float("nan")})
    (False, {'a': True, 'b': True, 'c': False})
    """
    coerced = {}
    passed = True
    for name, value in checks.items():
        ok = _as_pass_flag(value)
        coerced[name] = ok
        passed = passed and ok
    return bool(passed), coerced


def _as_pass_flag(value):
    """Coerce one check value to a plain ``bool``.

    Parameters
    ----------
    value : object
        The raw check value.

    Returns
    -------
    bool
        ``True`` only for a true logical scalar or a finite non-zero numeric
        scalar; ``False`` for everything else (see :func:`pass_summary`).
    """
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (str, bytes)):
        # MATLAB's isnumeric('1') is false, but np.asarray('1', dtype=float)
        # succeeds - reject text before it can be coerced into a passing check.
        return False
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        # Not numeric at all (a string, an object, ...): MATLAB's isnumeric
        # branch would reject it too.
        return False
    if array.ndim != 0:
        return False           # MATLAB requires isscalar
    number = float(array)
    if not math.isfinite(number):
        return False           # nan / inf: no interpretable metric
    return number != 0.0

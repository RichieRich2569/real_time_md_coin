"""Posterior predictive p-value validation (back-compatibility wrapper).

Translated from ``validation/validate_p_values.m``.

Generates scalar synthetic streams, scores each cue and feedback BEFORE it is
consumed, and reports Kolmogorov-Smirnov distances from a uniform distribution.
This is validation code, not a fast unit test.

Like the ``.m``, this is a thin compatibility shim: it delegates every argument
unchanged to :mod:`validation.validate_p_values_extended`, which keeps the
original feedback/cue fields and adds the state and parameter-rank diagnostics.
It is kept (not deleted) for the same reason the ``.m`` keeps it - it is a
documented public entry point - and it carries no independent logic that could
drift out of sync.
"""

from __future__ import annotations

from validation import validate_p_values_extended as _extended

__all__ = ["run"]


def run(seed=1201, strict=False, **overrides):
    """Delegate to :func:`validation.validate_p_values_extended.run`.

    Parameters
    ----------
    seed : int, optional
        Base seed. Default 1201.
    strict : bool, optional
        Raise instead of returning ``passed=False``. Default ``False``.
    **overrides
        Forwarded unchanged; see the extended validator for the accepted keys.

    Returns
    -------
    dict
        The extended validator's results struct.
    """
    return _extended.run(seed=seed, strict=strict, **overrides)


if __name__ == "__main__":       # pragma: no cover
    # Invoke as ``python -m validation.validate_p_values`` from the repo root:
    # this is a namespace package, so running the file by path would not put the
    # package root on sys.path.
    run()

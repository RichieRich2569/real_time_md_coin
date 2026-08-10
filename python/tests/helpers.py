"""Assertion and integration helpers shared by the test-suite.

Python counterpart of the MATLAB ``tests/+testutil`` package: ``assertClose``,
``assertInRange``, ``mustError``, ``integrate2d`` and ``assertDensityPeaks``.

The MATLAB helpers exist because plain ``assert`` gives an unhelpful message on
an array comparison; the Python ones exist for the same reason and keep the same
failure-message content (in particular the observed maximum absolute error).

Not ported here: ``alignmentFixture`` and ``largeAssignmentFixture``, which
build hand-crafted aligned particle states - unit B3 owns those, alongside the
alignment implementation they exercise.

For ordinary numeric comparisons prefer :func:`numpy.testing.assert_allclose`
directly; :func:`assert_close` is for the max-abs-difference contract the
MATLAB tests were written against.
"""

from __future__ import annotations

import numpy as np

#: ``numpy.trapz`` was renamed ``numpy.trapezoid`` in numpy 2.0; the package
#: supports numpy >= 1.24, so bind whichever exists.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "assert_close",
    "assert_in_range",
    "must_error",
    "integrate_2d",
    "assert_density_peaks",
]


def assert_close(actual, expected, tol, msg=""):
    """Assert two arrays match element-wise within an absolute tolerance.

    Fails when the largest absolute difference exceeds ``tol``; the failure
    message reports that observed maximum, which is what makes a near-miss
    diagnosable.

    Parameters
    ----------
    actual, expected : array_like
        Values to compare. Broadcast against each other, then flattened.
    tol : float
        Maximum permitted absolute difference.
    msg : str, optional
        Prefix for the failure message.

    Returns
    -------
    float
        The observed maximum absolute error.

    Raises
    ------
    AssertionError
        If the maximum absolute difference exceeds ``tol``.
    """
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    err = float(np.max(np.abs(a - b))) if a.size or b.size else 0.0
    assert err <= tol, "%s (max abs err %.3g, tol %.3g)" % (msg, err, tol)
    return err


def assert_in_range(name, v, lo, hi):
    """Assert every element of ``v`` lies in ``[lo, hi]``.

    Both bounds are relaxed by ``1e-12`` so exact-boundary values computed in
    floating point still pass.

    Parameters
    ----------
    name : str
        Name reported in the failure message.
    v : array_like
        Values to check.
    lo, hi : float
        Inclusive bounds.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any element falls outside the (relaxed) bounds.
    """
    arr = np.asarray(v, dtype=float)
    below = arr < lo - 1e-12
    above = arr > hi + 1e-12
    if np.any(below) or np.any(above):
        raise AssertionError(
            "FAILED: %s outside [%g, %g] (min %.6g, max %.6g)"
            % (name, lo, hi, float(np.min(arr)), float(np.max(arr)))
        )


def must_error(name, fn, expected_id):
    """Assert a callable raises an error whose message carries an identifier.

    The Python analogue of MATLAB's ``testutil.mustError``: the exception
    classes in :mod:`realtimecoin.exceptions` prefix their message with the
    MATLAB identifier, and the plain ``ValueError``s raised by the constructor
    validators do the same, so a substring test on ``str(exc)`` reproduces
    MATLAB's ``err.identifier`` comparison.

    Equivalent to ``pytest.raises(Exception, match=re.escape(expected_id))``;
    use whichever reads better at the call site.

    Parameters
    ----------
    name : str
        Name reported in the failure message.
    fn : callable
        Zero-argument callable expected to raise.
    expected_id : str
        Identifier that must appear in the exception message.

    Returns
    -------
    Exception
        The exception that was raised, for further inspection.

    Raises
    ------
    AssertionError
        If ``fn`` does not raise, or raises without ``expected_id`` in its
        message.
    """
    try:
        fn()
    except Exception as exc:   # noqa: BLE001 - any exception type is admissible
        if expected_id not in str(exc):
            raise AssertionError(
                "FAILED: %s raised %s(%s), expected identifier %r"
                % (name, type(exc).__name__, exc, expected_id)
            ) from exc
        return exc
    raise AssertionError("FAILED: %s did not raise" % (name,))


def integrate_2d(dens_fun, center, half_std):
    """Tensor-product trapezoidal integral of a 2-D density.

    Integrates over a ``121 x 121`` grid spanning ``center +/- 6 * half_std`` on
    each axis - wide enough that a Gaussian-like density integrates to one to
    within the trapezoid error, which is what the density tests check.

    Parameters
    ----------
    dens_fun : callable
        Maps a ``(K, 2)`` array of query points (one point per ROW, this
        package's convention) to a ``(K,)`` density.
    center : sequence of float
        Length-2 grid centre.
    half_std : sequence of float
        Length-2 per-axis scale; the grid spans six of these either side.

    Returns
    -------
    integral : float
        The trapezoidal integral of the density over the grid.
    vals : numpy.ndarray
        ``(n * n,)`` density evaluated on the flattened grid.
    """
    n = 121
    span = 6
    ax1 = np.linspace(center[0] - span * half_std[0], center[0] + span * half_std[0], n)
    ax2 = np.linspace(center[1] - span * half_std[1], center[1] + span * half_std[1], n)
    # meshgrid with 'xy' indexing so g1 varies along the columns, matching the
    # MATLAB helper's grid orientation.
    g1, g2 = np.meshgrid(ax1, ax2)
    pts = np.column_stack([g1.reshape(-1), g2.reshape(-1)])     # (n * n, 2)
    vals = np.asarray(dens_fun(pts), dtype=float).reshape(-1)
    z = vals.reshape(g1.shape)
    # Integrate along axis 1 (ax1) first, then along ax2, as MATLAB does.
    integral = float(_trapezoid(_trapezoid(z, ax1, axis=1), ax2))
    return integral, vals


def assert_density_peaks(name, dmap, grid, means):
    """Assert each per-context density peaks at that context's prototype mean.

    For every context key in ``dmap`` the density must peak at the grid point
    nearest ``means[context]``, within two grid spacings.

    Parameters
    ----------
    name : str
        Name reported in the failure message.
    dmap : dict
        ``{context_label: (K,) density}``, as returned by the
        ``*_given_context_probability`` queries.
    grid : array_like
        ``(K,)`` query grid the densities were evaluated on; assumed uniform.
    means : dict or array_like
        Expected mean per context label; indexed by the keys of ``dmap``.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any density peaks more than two grid spacings from its mean.
    """
    grid = np.asarray(grid, dtype=float)
    dx = grid[1] - grid[0]
    for c, density in dmap.items():
        i_peak = int(np.argmax(np.asarray(density, dtype=float)))
        expected = float(means[c])
        if abs(grid[i_peak] - expected) > 2 * dx:
            raise AssertionError(
                "FAILED: %s context %s peak at %.5g, expected mean %.5g"
                % (name, c, grid[i_peak], expected)
            )

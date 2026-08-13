"""Shared pytest fixtures.

Everything random in this package flows through an explicit
:class:`numpy.random.Generator`, so tests must never rely on the global
``numpy.random`` state. These fixtures make the seeded generator and the seeded
model factory the path of least resistance.
"""

from __future__ import annotations

import numpy as np
import pytest

from realtimecoin import RealTimeCOIN

#: Default seed for the whole suite. A fixed integer, not a random one, so a
#: failure is reproducible from the test name alone.
DEFAULT_SEED = 12345


@pytest.fixture
def rng():
    """Seeded random generator.

    Returns
    -------
    numpy.random.Generator
        A generator seeded with :data:`DEFAULT_SEED`, fresh for each test.
    """
    return np.random.default_rng(DEFAULT_SEED)


@pytest.fixture
def make_model():
    """Factory building seeded :class:`~realtimecoin.RealTimeCOIN` models.

    The returned callable forwards every keyword to the constructor and
    defaults ``rng`` to :data:`DEFAULT_SEED`, so a test that does not care about
    the seed still gets a reproducible model. Tests that need two models with
    the same explicit seed (or an unusual one) construct ``RealTimeCOIN``
    directly with ``rng=...`` instead.

    Returns
    -------
    callable
        ``make_model(**kwargs) -> RealTimeCOIN``. Pass ``rng=<seed or
        Generator>`` to override the default seed.

    Examples
    --------
    >>> def test_something(make_model):
    ...     model = make_model(num_particles=8, max_contexts=3)
    ...     assert model.Trial == 0
    """

    def _make(**kwargs):
        kwargs.setdefault("rng", DEFAULT_SEED)
        return RealTimeCOIN(**kwargs)

    return _make

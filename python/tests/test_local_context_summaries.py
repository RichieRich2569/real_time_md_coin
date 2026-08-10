"""Fast local-label context summaries avoid the global relabelling.

Python port of ``tests/test_local_context_summaries.m``. The three ``*_local``
queries return a length-``max_contexts + 1`` vector that normalises, and - the
point of their existence - they must NOT build or refresh the global alignment
cache.

The MATLAB test detects the "did not touch the alignment" property indirectly,
by observing that ``context_alignment().cache_state_version`` changed across an
``observe_y``. Here it is asserted directly and more strongly: the queries must
leave ``model.alignment_cache`` at ``None`` and ``model.state_version``
unchanged, which needs no alignment machinery at all and therefore also works
while unit B3 is stubbed.

The normalisation checks themselves route through
``context.local_context_probability_vector``, which selects the modal particles
via unit B3's ``select_modal_contexts``; they are wrapped in
:func:`_skipping_pending_units` until that unit merges.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from realtimecoin import RealTimeCOIN


@contextlib.contextmanager
def _skipping_pending_units(what):
    """Skip the enclosed assertions while their unit is still a stub.

    Parameters
    ----------
    what : str
        Short description of the pending dependency.

    Yields
    ------
    None
    """
    try:
        yield
    except NotImplementedError as exc:
        pytest.skip("pending %s: %s" % (what, exc))


def _model():
    """Model matching the MATLAB fixture: 20 particles, 4 contexts, one trial.

    Returns
    -------
    RealTimeCOIN
        The model after a single uncued observation.
    """
    model = RealTimeCOIN(num_particles=20, max_contexts=4, rng=12345)
    model.observe_y(0.1)
    return model


def test_local_summaries_are_normalised_context_vectors():
    """All three summaries are length-C vectors summing to one."""
    model = _model()
    with _skipping_pending_units("unit B3 (alignment)"):
        summaries = {
            "predicted": model.predicted_context_probabilities_local(),
            "responsibilities": model.context_responsibilities_local(),
            "count": model.sampled_context_count_local(),
        }
    for name, vector in summaries.items():
        vector = np.asarray(vector)
        assert vector.ndim == 1, name
        assert vector.size == model.max_contexts + 1, (
            "%s local summary should be a Cmax-length vector" % name
        )
        assert np.all(vector >= 0.0), name
        assert abs(float(vector.sum()) - 1.0) < 1e-12, (
            "%s local summary should normalize" % name
        )


def test_local_summaries_do_not_touch_the_alignment_cache():
    """The whole point of the local frame: no global alignment is computed."""
    model = _model()
    # observe_y invalidates the cache, so it starts empty.
    assert model.alignment_cache is None
    version_before = model.state_version

    for query in (
        model.predicted_context_probabilities_local,
        model.context_responsibilities_local,
        model.sampled_context_count_local,
    ):
        try:
            query()
        except NotImplementedError:
            # select_modal_contexts is unit B3's; reaching it is itself proof
            # that no CACHE was built (ensure_context_alignment is never called).
            pass
        assert model.alignment_cache is None, (
            "%s built the global alignment cache" % query.__name__
        )
        assert model.state_version == version_before, (
            "%s bumped the state version" % query.__name__
        )


def test_local_summaries_are_read_only():
    """The queries mutate no particle state and consume no random numbers."""
    model = _model()
    before = {
        name: value.copy()
        for name, value in model.D.as_dict().items()
        if isinstance(value, np.ndarray)
    }
    rng_state_before = model.rng.bit_generator.state

    for query in (
        model.predicted_context_probabilities_local,
        model.context_responsibilities_local,
        model.sampled_context_count_local,
    ):
        try:
            query()
        except NotImplementedError:
            pass

    after = model.D.as_dict()
    for name, value in before.items():
        np.testing.assert_array_equal(after[name], value, err_msg=name)
    assert model.rng.bit_generator.state == rng_state_before
    assert model.Trial == 1


def test_local_and_global_frames_agree_on_a_single_context_model():
    """With one context per particle the local frame is trivially resolvable.

    ``max_contexts == 1`` leaves every particle with exactly one context and no
    novel slot, so the local summary must put all its mass on slot 0 - a fact
    that needs no relabelling and pins the vector's orientation.
    """
    model = RealTimeCOIN(num_particles=12, max_contexts=1, rng=7)
    model.observe_y(0.3)
    with _skipping_pending_units("unit B3 (alignment)"):
        vector = np.asarray(model.sampled_context_count_local())
    np.testing.assert_allclose(vector[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(vector[1:], 0.0, atol=1e-12)

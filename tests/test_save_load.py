"""Persistence: snapshot / load_snapshot and save_model / load_model.

Python counterpart of ``tests/test_save_load.m``, strengthened where the Python
implementation can promise more than the MATLAB one. MATLAB draws from the
global random stream, so a restored model there can only be checked for
*posterior* equality; here the snapshot carries ``model.rng``'s bit-generator
state, so the far stronger property holds:

    saving mid-run, then streaming the remaining observations into a model
    loaded from the file, reproduces the uninterrupted run BIT FOR BIT.

That "resume == uninterrupted" test is the centre of this file; it is run for
both the scalar and the multi-dimensional pipeline, over both the on-disk and
the in-memory path.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import fields as dataclass_fields

import numpy as np
import pytest

from realtimecoin import ModelFormatError, ParticleState, RealTimeCOIN
from realtimecoin.model import PROPERTY_NAMES
from realtimecoin.persist import (
    FORMAT_VERSION,
    _META_KEY,
    _from_jsonable,
    _to_jsonable,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _observation(state_dim, t):
    """Deterministic feedback for trial ``t`` (scalar or ``(N,)`` vector)."""
    if state_dim == 1:
        return 0.03 * t
    return np.array([0.03 * t * (1 + 0.1 * i) for i in range(state_dim)])


def _stream(model, state_dim, trials):
    """Feed ``trials`` cue/feedback pairs and return the motor-output trace.

    Returns
    -------
    numpy.ndarray
        ``(len(trials), N)`` motor outputs, one row per trial.
    """
    out = []
    for t in trials:
        model.observe_q(10 + (t % 3))
        model.observe_y(_observation(state_dim, t))
        out.append(np.atleast_1d(np.asarray(model.motor_output(), dtype=float)))
    return np.array(out)


def _trained(state_dim=1, trials=6, **kwargs):
    """Build a seeded model and run ``trials`` trials into it."""
    kwargs.setdefault("num_particles", 20)
    kwargs.setdefault("max_contexts", 3)
    kwargs.setdefault("rng", 3)
    model = RealTimeCOIN(state_dim=state_dim, **kwargs)
    _stream(model, state_dim, range(1, trials + 1))
    return model


def _rewrite_meta(path, mutate):
    """Rewrite a saved model's metadata block through ``mutate(meta) -> meta``.

    Used to forge otherwise-valid files that fail one specific validation.
    """
    with np.load(path, allow_pickle=False) as archive:
        entries = {name: np.array(archive[name]) for name in archive.files}
    meta = json.loads(bytes(entries.pop(_META_KEY)).decode("utf-8"))
    blob = json.dumps(mutate(meta)).encode("utf-8")
    entries[_META_KEY] = np.frombuffer(blob, dtype=np.uint8)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **entries)


# ----------------------------------------------------------------------
# Translation of tests/test_save_load.m
# ----------------------------------------------------------------------


def test_stationary_save_resets_trial_but_keeps_what_was_learned(tmp_path):
    """Stationary save: Trial rewinds, counts / cue columns / cues survive.

    The MATLAB assertions of ``test_save_load.m``, one for one.
    """
    model = RealTimeCOIN(num_particles=20, max_contexts=3, infer_bias=True, rng=3)
    model.observe_q(10)
    model.observe_y(0.3)
    model.observe_q(20)
    model.observe_y(0.1)

    count_mass_before = float(np.sum(model.D.n_context))
    cue_columns_before = model.D.n_cue.shape[2]

    path = tmp_path / "stationary.rtcoin"
    model.save_model(path, set_stationary=True)

    loaded = RealTimeCOIN(num_particles=1)   # dummy initialisation, as MATLAB
    loaded.load_model(path)

    assert loaded.Trial == 0, "Trial should reset to zero after a stationary save"
    assert float(np.sum(loaded.D.n_context)) == count_mass_before, (
        "Stationary save should preserve learned transition counts"
    )
    assert loaded.D.n_cue.shape[2] == cue_columns_before, (
        "Stationary save should preserve learned cue columns"
    )
    assert loaded.cue_values == [10.0, 20.0], (
        "Stationary save should preserve the raw cue-value mapping"
    )
    assert loaded.D.n_cues == model.D.n_cues


def test_non_stationary_save_round_trips_trial_and_responsibilities(tmp_path):
    """Plain save: the trial counter and the posterior come back unchanged."""
    model = RealTimeCOIN(num_particles=20, max_contexts=3, infer_bias=True, rng=3)
    for value in (0.3, 0.1, 0.1):
        model.observe_q(10)
        model.observe_y(value)

    path = tmp_path / "verbatim.rtcoin"
    model.save_model(path, set_stationary=False)

    loaded = RealTimeCOIN(num_particles=1)
    loaded.load_model(path)

    assert loaded.Trial == model.Trial

    before = model.responsibilities_map()
    after = loaded.responsibilities_map()
    assert set(before) == set(after), "Context key set mismatch after save/load"
    for key in before:
        # MATLAB allowed 1e-6 here; an npz round trip is bit-exact and the query
        # is deterministic, so anything short of equality would be a real defect.
        assert after[key] == before[key], key


# ----------------------------------------------------------------------
# Resume equivalence: the property MATLAB cannot state
# ----------------------------------------------------------------------


@pytest.mark.parametrize("state_dim", [1, 2])
def test_file_resume_is_bit_identical_to_an_uninterrupted_run(tmp_path, state_dim):
    """Save mid-run, reload, stream on: the trace matches exactly.

    Saving must capture the random stream as well as the posterior, otherwise
    the resumed run diverges from the uninterrupted one at the first draw.
    """
    n_pre, n_post = 8, 8
    model = _trained(state_dim=state_dim, trials=n_pre, num_particles=15)

    path = tmp_path / "midrun.rtcoin"
    model.save_model(path, set_stationary=False)

    tail = range(n_pre + 1, n_pre + n_post + 1)
    uninterrupted = _stream(model, state_dim, tail)

    resumed_model = RealTimeCOIN(num_particles=1, rng=999)
    resumed_model.load_model(path)
    resumed = _stream(resumed_model, state_dim, tail)

    assert resumed.shape == uninterrupted.shape
    assert np.max(np.abs(resumed - uninterrupted)) == 0.0, (
        "A resumed run must be bit-identical to the uninterrupted one"
    )
    assert resumed_model.Trial == model.Trial


@pytest.mark.parametrize("state_dim", [1, 2])
def test_snapshot_resume_is_bit_identical_to_an_uninterrupted_run(state_dim):
    """The in-memory path carries the same resume guarantee as the file."""
    n_pre, n_post = 8, 8
    model = _trained(state_dim=state_dim, trials=n_pre, num_particles=15)
    payload = model.snapshot()

    tail = range(n_pre + 1, n_pre + n_post + 1)
    uninterrupted = _stream(model, state_dim, tail)

    resumed_model = RealTimeCOIN(num_particles=1, rng=999)
    resumed_model.load_snapshot(payload)
    resumed = _stream(resumed_model, state_dim, tail)

    assert np.max(np.abs(resumed - uninterrupted)) == 0.0


def test_one_snapshot_restores_into_several_models():
    """A snapshot is reusable: loading it twice gives two identical models.

    Guards the deep copy on the way IN - if the restore aliased the payload,
    the first model's subsequent trials would corrupt the second restore.
    """
    model = _trained(trials=5)
    payload = model.snapshot()

    first = RealTimeCOIN(num_particles=1)
    first.load_snapshot(payload)
    _stream(first, 1, range(6, 10))          # mutate the first restore

    second = RealTimeCOIN(num_particles=1)
    second.load_snapshot(payload)

    assert second.Trial == model.Trial
    np.testing.assert_array_equal(second.D.n_context, model.D.n_context)
    np.testing.assert_array_equal(
        second.D.responsibilities, model.D.responsibilities
    )


def test_snapshot_shares_no_buffer_with_the_model():
    """Mutating a snapshot's arrays cannot reach back into the live model."""
    model = _trained(trials=3)
    payload = model.snapshot()
    original = model.D.n_context.copy()

    payload["D"].n_context[...] = -1.0
    payload["cue_values"].append(12345.0)

    np.testing.assert_array_equal(model.D.n_context, original)
    assert 12345.0 not in model.cue_values


# ----------------------------------------------------------------------
# Completeness of the serialisation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("state_dim", [1, 2])
def test_every_non_none_particle_field_round_trips(tmp_path, state_dim):
    """Every populated ParticleState field survives a save/load unchanged.

    Driven by ``dataclasses.fields`` rather than a hand-written list, so a field
    added to :class:`~realtimecoin.state.ParticleState` cannot escape
    persistence unnoticed.
    """
    model = _trained(state_dim=state_dim, trials=5, infer_bias=True)
    path = tmp_path / "complete.rtcoin"
    model.save_model(path, set_stationary=False)

    loaded = RealTimeCOIN(num_particles=1)
    loaded.load_model(path)

    populated = 0
    for field in dataclass_fields(ParticleState):
        before = getattr(model.D, field.name)
        after = getattr(loaded.D, field.name)
        if before is None:
            assert after is None, "%s should have stayed None" % (field.name,)
            continue
        populated += 1
        if isinstance(before, np.ndarray):
            assert after.dtype == before.dtype, (
                "%s changed dtype across the round trip" % (field.name,)
            )
            np.testing.assert_array_equal(after, before, err_msg=field.name)
        else:
            assert after == before, field.name
    assert populated > 20, "sanity: most fields should be populated after a run"


def test_no_model_attribute_escapes_the_snapshot():
    """Every attribute a live model carries is saved, or knowingly excluded.

    The companion to the ParticleState field sweep: it guards the OTHER half of
    the state. If a later unit adds an attribute in ``RealTimeCOIN.__init__``
    and forgets to serialise it, this fails and names it, instead of the loss
    showing up as a subtly diverging resumed run.
    """
    model = _trained(trials=4)
    model.context_alignment()          # populate the lazily-built cache
    assert model.alignment_cache is not None

    accounted_for = set(PROPERTY_NAMES) | {
        "D",
        "pending_q",
        "trial",
        "cue_values",
        "state_version",
        "alignment_seed",
        # Saved as payload["rng_state"] rather than under its own name.
        "rng",
        # Deliberately NOT saved: a derived cache, rebuilt on first use after
        # the restore invalidates it.
        "alignment_cache",
    }
    escaped = sorted(set(vars(model)) - accounted_for)
    assert not escaped, (
        "model attribute(s) %s are not covered by the persistence layer; add "
        "them to serializable_state/restore_serializable_state (or to this "
        "test's exclusion list, with a reason)" % (escaped,)
    )

    payload = model.snapshot()
    assert set(payload) == {
        "format",
        "properties",
        "D",
        "pending_q",
        "trial",
        "cue_values",
        "state_version",
        "alignment_seed",
        "rng_state",
    }


def test_every_constructor_property_round_trips(tmp_path):
    """All 20 public properties come back, including the covariance overrides."""
    q = np.array([[0.002, 0.0005], [0.0005, 0.003]])
    r = np.array([[0.01, 0.0], [0.0, 0.02]])
    model = RealTimeCOIN(
        num_particles=8,
        max_contexts=3,
        state_dim=2,
        gamma_context=0.2,
        alpha_context=7.5,
        rho_context=0.3,
        gamma_cue=0.15,
        alpha_cue=20.0,
        prior_mean_retention=0.9,
        prior_mean_drift=0.01,
        prior_mean_bias=0.02,
        prior_precision_retention=100.0,
        prior_precision_drift=200.0,
        prior_precision_bias=300.0,
        sigma_process_noise=0.01,
        sigma_sensory_noise=0.04,
        sigma_motor_noise=0.005,
        process_noise_covariance=q,
        observation_noise_covariance=r,
        infer_bias=True,
        rng=5,
    )
    path = tmp_path / "props.rtcoin"
    model.save_model(path, set_stationary=False)

    loaded = RealTimeCOIN(num_particles=1)
    loaded.load_model(path)

    for name in PROPERTY_NAMES:
        before = getattr(model, name)
        after = getattr(loaded, name)
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(after, before, err_msg=name)
        else:
            assert after == before, name
            assert type(after) is type(before), name


def test_bookkeeping_round_trips(tmp_path):
    """Trial, cue registry, staged cue and alignment seed all survive."""
    model = _trained(trials=4)
    model.context_alignment()          # populate model.alignment_seed
    model.observe_q(42)                # leave a cue staged
    assert model.alignment_seed is not None

    path = tmp_path / "book.rtcoin"
    model.save_model(path, set_stationary=False)
    loaded = RealTimeCOIN(num_particles=1)
    loaded.load_model(path)

    assert loaded.trial == model.trial
    assert loaded.pending_q == model.pending_q == 42.0
    assert loaded.cue_values == model.cue_values
    assert loaded.alignment_seed is not None
    assert set(loaded.alignment_seed) == set(model.alignment_seed)
    for key, expected in model.alignment_seed.items():
        actual = loaded.alignment_seed[key]
        if isinstance(expected, np.ndarray):
            assert actual.dtype == expected.dtype, key
            np.testing.assert_array_equal(actual, expected, err_msg=key)
        elif isinstance(expected, dict):
            # global_contexts: the nested dict-of-arrays that exercises the
            # recursive encoder, and where a lost dtype would hide best.
            assert set(actual) == set(expected), key
            for name, proto in expected.items():
                assert actual[name].dtype == proto.dtype, "%s/%s" % (key, name)
                assert actual[name].shape == proto.shape, "%s/%s" % (key, name)
                np.testing.assert_array_equal(actual[name], proto, err_msg=name)
        else:
            assert actual == expected, key
            assert type(actual) is type(expected), key
    # The restore invalidates the cache, so the version advances by exactly one.
    assert loaded.state_version == model.state_version + 1
    assert loaded.alignment_cache is None


def test_the_file_is_pickle_free(tmp_path):
    """The archive loads with ``allow_pickle=False`` and holds no object arrays.

    A model file is data, never code: reading one must not be able to execute
    anything.
    """
    model = _trained(trials=3)
    path = tmp_path / "safe.rtcoin"
    model.save_model(path, set_stationary=False)

    # Reading every member with allow_pickle=False IS the check: numpy raises
    # ValueError on an object array here, so reaching the end proves there is
    # no pickled payload anywhere in the archive.
    with np.load(path, allow_pickle=False) as archive:
        assert _META_KEY in archive.files
        for name in archive.files:
            assert archive[name] is not None

    # And the raw bytes carry no pickle opcodes from a numpy object array.
    assert b"numpy.core.multiarray._reconstruct" not in path.read_bytes()


def test_save_writes_the_exact_filename(tmp_path):
    """No ``.npz`` suffix is appended behind the caller's back."""
    path = tmp_path / "model.checkpoint"
    _trained(trials=2).save_model(path, set_stationary=False)
    assert path.exists()
    assert not (tmp_path / "model.checkpoint.npz").exists()


# ----------------------------------------------------------------------
# Saving must not disturb the live model
# ----------------------------------------------------------------------


@pytest.mark.parametrize("stationary", [False, True])
def test_saving_never_mutates_the_live_model(tmp_path, stationary):
    """Neither save mode touches the live posterior, counter or random stream.

    MATLAB's ``saveModel(file, true)`` rewinds the object and restores it
    afterwards; here the stationary reset runs on a clone, so nothing about the
    live model - including where its random stream stands - changes at all.
    """
    model = _trained(trials=5, infer_bias=True, state_dim=2,
                     process_noise_covariance=np.array([[2e-4, 1e-5],
                                                        [1e-5, 3e-4]]))
    model.context_alignment()          # populate alignment_seed and the cache
    before = model.snapshot()
    rng_before = model.rng.bit_generator.state
    seed_before = model.alignment_seed
    cache_before = model.alignment_cache
    q_before = model.process_noise_covariance.copy()

    model.save_model(tmp_path / "probe.rtcoin", set_stationary=stationary)

    assert model.trial == before["trial"]
    assert model.pending_q == before["pending_q"]
    assert model.cue_values == before["cue_values"]
    assert model.rng.bit_generator.state == rng_before
    assert model.state_version == before["state_version"]
    # The alignment seed and cache must survive by IDENTITY: a save that
    # recomputed or dropped them would silently cost the next query its warm
    # start.
    assert model.alignment_seed is seed_before
    assert model.alignment_cache is cache_before
    np.testing.assert_array_equal(model.process_noise_covariance, q_before)
    for field in dataclass_fields(ParticleState):
        old = getattr(before["D"], field.name)
        new = getattr(model.D, field.name)
        if isinstance(old, np.ndarray):
            np.testing.assert_array_equal(new, old, err_msg=field.name)
        else:
            assert new == old, field.name


def test_a_stationary_save_leaves_the_run_reproducible(tmp_path):
    """A stationary save mid-run does not perturb the continuing trace."""
    model = _trained(trials=6, num_particles=15)
    reference = model.snapshot()

    model.save_model(tmp_path / "stat.rtcoin", set_stationary=True)
    after_save = _stream(model, 1, range(7, 13))

    twin = RealTimeCOIN(num_particles=1)
    twin.load_snapshot(reference)
    without_save = _stream(twin, 1, range(7, 13))

    assert np.max(np.abs(after_save - without_save)) == 0.0


# ----------------------------------------------------------------------
# Rejecting bad files
# ----------------------------------------------------------------------


def test_unknown_format_version_raises_model_format_error(tmp_path):
    """A file from a future (or forged) schema is refused, not guessed at."""
    path = tmp_path / "future.rtcoin"
    _trained(trials=2).save_model(path, set_stationary=False)

    def bump(meta):
        meta["format"] = "rtcoin-model-v99"
        return meta

    _rewrite_meta(path, bump)

    model = RealTimeCOIN(num_particles=1)
    with pytest.raises(ModelFormatError, match="RealTimeCOIN:ModelFormat"):
        model.load_model(path)
    with pytest.raises(ModelFormatError, match="rtcoin-model-v99"):
        model.load_model(path)


def test_archive_without_metadata_raises_model_format_error(tmp_path):
    """A plain npz that is not a model save is refused."""
    path = tmp_path / "notamodel.npz"
    with open(path, "wb") as handle:
        np.savez_compressed(handle, some_array=np.arange(4))

    with pytest.raises(ModelFormatError, match=_META_KEY):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_non_archive_file_raises_model_format_error(tmp_path):
    """Arbitrary bytes produce a clear format error, not a decode traceback."""
    path = tmp_path / "garbage.rtcoin"
    path.write_bytes(b"this is definitely not an npz archive")

    with pytest.raises(ModelFormatError, match="not a readable"):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_missing_particle_array_raises_model_format_error(tmp_path):
    """A truncated archive is refused rather than loaded with holes."""
    path = tmp_path / "truncated.rtcoin"
    _trained(trials=3).save_model(path, set_stationary=False)

    with np.load(path, allow_pickle=False) as archive:
        entries = {
            name: np.array(archive[name])
            for name in archive.files
            if name != "D.n_context"
        }
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **entries)

    with pytest.raises(ModelFormatError, match="missing the array entry"):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_field_list_mismatch_raises_model_format_error(tmp_path):
    """Metadata that does not describe THIS ParticleState is refused."""
    path = tmp_path / "otherfields.rtcoin"
    _trained(trials=3).save_model(path, set_stationary=False)

    def drop_a_field(meta):
        meta["d_none_fields"] = [
            name for name in meta["d_none_fields"] if name != "retention"
        ]
        meta["d_none_fields"].append("a_field_that_does_not_exist")
        return meta

    _rewrite_meta(path, drop_a_field)

    with pytest.raises(ModelFormatError, match="different particle state"):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_corrupt_metadata_blob_raises_model_format_error(tmp_path):
    """Damaged JSON in the metadata entry is reported as a format error."""
    path = tmp_path / "badmeta.rtcoin"
    _trained(trials=2).save_model(path, set_stationary=False)

    with np.load(path, allow_pickle=False) as archive:
        entries = {name: np.array(archive[name]) for name in archive.files}
    entries[_META_KEY] = np.frombuffer(b"{not json at all", dtype=np.uint8)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **entries)

    with pytest.raises(ModelFormatError, match="unreadable metadata"):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_unknown_property_name_is_refused_not_injected(tmp_path):
    """A file cannot set arbitrary attributes on the model it is loaded into.

    The property names come straight out of the archive; without a whitelist a
    hand-edited file could overwrite ``rng``, ``D``, or shadow a bound method.
    """
    path = tmp_path / "injected.rtcoin"
    _trained(trials=2).save_model(path, set_stationary=False)

    def inject(meta):
        meta["properties"]["motor_output"] = 1.0
        return meta

    _rewrite_meta(path, inject)

    model = RealTimeCOIN(num_particles=1)
    with pytest.raises(ModelFormatError, match="unknown model propert"):
        model.load_model(path)
    # The failed load must not have left the attribute behind.
    assert callable(model.motor_output)


def test_missing_shape_properties_are_refused(tmp_path):
    """Properties the particle shapes depend on may not be dropped."""
    path = tmp_path / "noprops.rtcoin"
    _trained(trials=2).save_model(path, set_stationary=False)

    def strip(meta):
        meta["properties"] = {}
        return meta

    _rewrite_meta(path, strip)

    with pytest.raises(ModelFormatError, match="num_particles"):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_particle_shapes_must_match_the_declared_properties(tmp_path):
    """Properties and particle arrays from different models cannot be paired."""
    path = tmp_path / "mismatch.rtcoin"
    _trained(trials=2, num_particles=20).save_model(path, set_stationary=False)

    def relabel(meta):
        meta["properties"]["num_particles"] = 19
        return meta

    _rewrite_meta(path, relabel)

    with pytest.raises(ModelFormatError, match="num_particles=19"):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_bare_npy_file_raises_model_format_error(tmp_path):
    """``np.load`` returns an array for a .npy; that is not a model archive."""
    path = tmp_path / "single.npy"
    with open(path, "wb") as handle:
        np.save(handle, np.arange(5))

    with pytest.raises(ModelFormatError, match="single numpy array"):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_corrupt_archive_member_raises_model_format_error(tmp_path):
    """A damaged compressed member is a format error, not a raw zlib error.

    ``np.load`` is lazy - it validates only the zip directory - so this failure
    surfaces later, when the member is actually decompressed.
    """
    path = tmp_path / "corrupt.rtcoin"
    _trained(trials=3).save_model(path, set_stationary=False)

    raw = bytearray(path.read_bytes())
    # Corrupt the tail of the payload, past the zip directory header, so the
    # archive still opens but one member fails to inflate.
    for offset in range(len(raw) // 3, len(raw) // 3 + 64):
        raw[offset] ^= 0xFF
    path.write_bytes(bytes(raw))

    with pytest.raises(ModelFormatError, match="RealTimeCOIN:ModelFormat"):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_malformed_rng_state_raises_model_format_error(tmp_path):
    """A damaged random-stream block is reported, not raised as a KeyError."""
    path = tmp_path / "badrng.rtcoin"
    _trained(trials=2).save_model(path, set_stationary=False)

    def damage(meta):
        meta["rng_state"] = {"bit_generator": "PCG64", "state": {"state": 1}}
        return meta

    _rewrite_meta(path, damage)

    with pytest.raises(ModelFormatError, match="random stream"):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_unknown_bit_generator_name_is_refused(tmp_path):
    """The bit-generator name is untrusted: only real BitGenerators resolve."""
    path = tmp_path / "badbg.rtcoin"
    _trained(trials=2).save_model(path, set_stationary=False)

    def forge(meta):
        # numpy.random.seed exists, but it is a function, not a BitGenerator.
        meta["rng_state"] = {"bit_generator": "seed", "state": {}}
        return meta

    _rewrite_meta(path, forge)

    with pytest.raises(ModelFormatError, match="not an available numpy bit generator"):
        RealTimeCOIN(num_particles=1).load_model(path)


def test_a_failed_save_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """A crash mid-write must not destroy a valid existing file."""
    path = tmp_path / "checkpoint.rtcoin"
    model = _trained(trials=4)
    model.save_model(path, set_stationary=False)
    good = path.read_bytes()

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        model.save_model(path, set_stationary=False)

    assert path.read_bytes() == good, "the old checkpoint was clobbered"
    assert not (tmp_path / "checkpoint.rtcoin.tmp").exists(), "temp file left behind"
    RealTimeCOIN(num_particles=1).load_model(path)   # still loadable


def test_missing_file_raises_file_not_found(tmp_path):
    """An absent path is an ordinary FileNotFoundError, not a format error."""
    with pytest.raises(FileNotFoundError):
        RealTimeCOIN(num_particles=1).load_model(tmp_path / "nope.rtcoin")


def test_load_snapshot_rejects_a_non_snapshot():
    """load_snapshot validates its argument instead of failing obscurely."""
    model = RealTimeCOIN(num_particles=1)
    with pytest.raises(ModelFormatError):
        model.load_snapshot(["not", "a", "mapping"])
    with pytest.raises(ModelFormatError, match="properties"):
        model.load_snapshot({"trial": 3})


def test_directory_path_raises_the_os_error_not_a_format_error(tmp_path):
    """A path problem is reported as a path problem."""
    directory = tmp_path / "adir"
    directory.mkdir()
    with pytest.raises(OSError) as excinfo:
        RealTimeCOIN(num_particles=1).load_model(directory)
    assert not isinstance(excinfo.value, ModelFormatError)


# ----------------------------------------------------------------------
# The JSON tagging layer
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        3,
        -2.5,
        float("inf"),
        float("-inf"),
        "text",
        [1, "two", None],
        (1, 2, 3),
        {"a": 1, "b": [2, 3]},
        {1: "int key", (2, 3): "tuple key"},
        np.arange(6, dtype=np.int64).reshape(2, 3),
        # Zero-size: tolist() flattens every empty array to [], so only the
        # recorded shape can tell (0, 2) from (0,) and from (2, 0).
        np.zeros((0, 2)),
        np.zeros((2, 0), dtype=np.int32),
        np.array([[1.5, np.inf], [-0.5, 2.0]]),
        np.array([True, False]),
        {"nested": {"arr": np.zeros((2, 2), dtype=np.int32), "t": (1.0, "x")}},
    ],
)
def test_json_tagging_round_trips(value):
    """Every value type the metadata block can hold survives JSON encoding."""
    decoded = _from_jsonable(json.loads(json.dumps(_to_jsonable(value))))
    _assert_same(decoded, value)


def test_json_tagging_round_trips_nan():
    """NaN survives too - it just cannot be compared with ``==``."""
    decoded = _from_jsonable(json.loads(json.dumps(_to_jsonable(float("nan")))))
    assert np.isnan(decoded)
    arr = _from_jsonable(
        json.loads(json.dumps(_to_jsonable(np.array([np.nan, 1.0]))))
    )
    np.testing.assert_array_equal(arr, np.array([np.nan, 1.0]))


def test_json_tagging_rejects_an_unsupported_type():
    """An unserialisable value fails loudly at save time, not silently."""
    with pytest.raises(TypeError, match="Cannot serialise"):
        _to_jsonable(object())


def _assert_same(actual, expected):
    """Assert equality while distinguishing arrays, tuples and lists."""
    assert type(actual) is type(expected), (type(actual), type(expected))
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_same(a, e)
    else:
        assert actual == expected


def test_restore_accepts_a_particle_state_given_as_a_mapping():
    """``D`` may arrive as a field mapping, as ``from_state`` accepts."""
    model = _trained(trials=3)
    payload = model.snapshot()
    payload["D"] = {
        name: value
        for name, value in payload["D"].as_dict().items()
        if value is not None
    }

    other = RealTimeCOIN(num_particles=1)
    other.load_snapshot(payload)
    np.testing.assert_array_equal(other.D.n_context, model.D.n_context)
    assert other.D.n_cues == model.D.n_cues


def test_restore_reproduces_the_stream_across_bit_generators():
    """A snapshot taken from a non-default bit generator restores exactly."""
    model = RealTimeCOIN(
        num_particles=10,
        max_contexts=3,
        rng=np.random.Generator(np.random.Philox(2024)),
    )
    _stream(model, 1, range(1, 5))
    payload = model.snapshot()

    uninterrupted = _stream(model, 1, range(5, 9))

    other = RealTimeCOIN(num_particles=1, rng=7)      # PCG64: a different kind
    other.load_snapshot(payload)
    assert type(other.rng.bit_generator).__name__ == "Philox"
    assert np.max(np.abs(_stream(other, 1, range(5, 9)) - uninterrupted)) == 0.0


def test_saved_metadata_declares_the_pinned_schema(tmp_path):
    """The on-disk schema name and layout are part of the contract."""
    path = tmp_path / "schema.rtcoin"
    _trained(trials=2).save_model(path, set_stationary=False)

    with zipfile.ZipFile(path) as zf:
        assert _META_KEY + ".npy" in zf.namelist()

    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(bytes(archive[_META_KEY]).decode("utf-8"))
        array_names = {n[len("D."):] for n in archive.files if n.startswith("D.")}

    assert meta["format"] == FORMAT_VERSION == "rtcoin-model-v1"
    assert set(meta["properties"]) == set(PROPERTY_NAMES)
    assert meta["rng_state"]["bit_generator"] == "PCG64"
    assert set(meta["d_array_fields"]) == array_names

    declared = {f.name for f in dataclass_fields(ParticleState)}
    partition = (
        set(meta["d_array_fields"])
        | set(meta["d_scalar_fields"])
        | set(meta["d_none_fields"])
    )
    assert partition == declared, "the three field lists must partition the state"

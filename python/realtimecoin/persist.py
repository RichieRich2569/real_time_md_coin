"""Snapshot / restore and file persistence for the model.

Translated from the public methods ``snapshot``, ``loadSnapshot``,
``saveModel``, ``loadModel`` and ``set_stationary``, plus the private helpers
``serializableState`` and ``restoreSerializableState``.

A snapshot is a plain, value-semantics mapping holding everything needed to
reconstruct the model: every public property under ``"properties"``, plus the
particle state, the staged cue, the trial counter, the cue registry and the
alignment bookkeeping (``state_version``, ``alignment_seed``). It is a true
deep copy, so it is safe to hand across process boundaries.

Note that :meth:`realtimecoin.model.RealTimeCOIN.from_state` already provides a
minimal restore path (used by the test fixtures); this module formalises it and
adds the on-disk format.


Two deliberate departures from the MATLAB original
--------------------------------------------------
1. **The random stream travels with the state.** MATLAB's ``serializableState``
   captures no RNG, because MATLAB draws from the global stream; here every
   draw goes through ``model.rng``, so the snapshot carries
   ``model.rng.bit_generator.state`` under ``"rng_state"``. A restored model
   therefore resumes the *exact* stream, and streaming the same observations
   into it reproduces the original run bit for bit.
2. **A stationary save never touches the live model.** MATLAB's
   ``saveModel(file, true)`` mutates the object with ``set_stationary``, writes,
   then restores. Here the stationarised state is computed on a private clone
   (see :func:`_clone_for_save`); the live model is not mutated at any point, so
   an interrupted save cannot leave it rewound and the live random stream is not
   advanced.


On-disk format: ``rtcoin-model-v1``
-----------------------------------
A ``numpy`` ``.npz`` archive (written with :func:`numpy.savez_compressed`, read
with ``allow_pickle=False`` - the format holds no pickles and loading it can
never execute code) containing:

``__meta__``
    A ``uint8`` array holding UTF-8 JSON with the keys

    ==================== =================================================
    ``format``           ``"rtcoin-model-v1"``
    ``properties``       the 20 constructor properties of
                         :data:`realtimecoin.model.PROPERTY_NAMES`
    ``trial``            trial counter
    ``pending_q``        staged raw cue, or ``null``
    ``cue_values``       raw cue values in label order
    ``state_version``    alignment-cache version counter
    ``alignment_seed``   warm-start seed (nested, arrays tagged)
    ``rng_state``        ``model.rng.bit_generator.state``
    ``d_array_fields``   ParticleState fields stored as ``D.*`` arrays
    ``d_scalar_fields``  non-array ParticleState fields, by value
    ``d_none_fields``    ParticleState fields that are ``None``
    ==================== =================================================

``D.<field>``
    One entry per non-``None`` array field of
    :class:`realtimecoin.state.ParticleState`, with its dtype and shape
    preserved by ``npz`` itself.

The three ``d_*_fields`` lists partition ``dataclasses.fields(ParticleState)``,
and :func:`load_model` checks that they do: a field that is added to the
dataclass but not written cannot slip through unnoticed, and a truncated or
hand-edited archive is rejected rather than silently loaded as a model with
``None`` where an array belongs.

Nested values that JSON cannot express natively (``numpy`` arrays and scalars,
tuples, dictionaries with non-string keys) are tagged by :func:`_to_jsonable`
and reconstructed exactly by :func:`_from_jsonable`.

NOT ported: MATLAB's ``restoreMDBiasStatisticsCompatibility``, the shim that
back-fills ``bias_info_ss`` / ``bias_precision_ss`` for multi-dimensional models
saved before those fields existed. There is no pre-``v1`` Python save format for
it to migrate from, and the ``d_*_fields`` partition check above already rejects
any file that does not describe the current
:class:`~realtimecoin.state.ParticleState`. If a ``v2`` ever needs to read ``v1``
files, that is where the migration hook belongs.
"""

from __future__ import annotations

import copy as _copy
import json
import os
import zipfile
import zlib
from dataclasses import fields as _dataclass_fields

import numpy as np

from . import alignment as _alignment
from . import context as _context
from . import numerics as _numerics
from . import pipeline_md as _pipeline_md
from . import pipeline_scalar as _pipeline_scalar
from . import state as _state
from . import statics as _statics
from .exceptions import ModelFormatError
from .state import ParticleState, copy_state

__all__ = [
    "snapshot",
    "load_snapshot",
    "save_model",
    "load_model",
    "set_stationary",
    "serializable_state",
    "restore_serializable_state",
    "FORMAT_VERSION",
]

#: Schema name written into (and demanded by) every model file.
FORMAT_VERSION = "rtcoin-model-v1"

#: ``npz`` entry holding the UTF-8 JSON metadata blob.
_META_KEY = "__meta__"

#: Prefix of the ``npz`` entry backing each ParticleState array field.
_D_PREFIX = "D."


# --------------------------------------------------------------------------
# JSON encoding of the non-array bookkeeping
# --------------------------------------------------------------------------


def _to_jsonable(value):
    """Recursively convert ``value`` into something :mod:`json` can encode.

    Values JSON handles natively (``None``, ``bool``, ``int``, ``float``,
    ``str``, and lists/dicts of those) pass through unchanged. Everything else
    is wrapped in a single-key tag dict that :func:`_from_jsonable` inverts
    exactly:

    * ``numpy.ndarray`` ->
      ``{"__ndarray__": nested lists, "dtype": str, "shape": [...]}``
    * ``numpy`` scalar -> its Python equivalent via ``.item()``
    * ``tuple`` -> ``{"__tuple__": [...]}``
    * ``dict`` with any non-string key -> ``{"__items__": [[k, v], ...]}``

    Parameters
    ----------
    value : object
        Value to encode.

    Returns
    -------
    object
        A JSON-encodable structure.

    Raises
    ------
    TypeError
        If ``value`` is of a type this encoder does not handle.
    """
    if value is None:
        return value
    if isinstance(value, np.ndarray):
        # tolist() already yields JSON scalars; dtype is carried separately so
        # an integer index array does not come back as float, and shape because
        # tolist() flattens any zero-size array to a bare [] - (0, 2) and (0,)
        # are indistinguishable without it.
        return {
            "__ndarray__": value.tolist(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    # Before the plain-scalar branch: numpy.float64 IS a float subclass, so it
    # would otherwise slip through unconverted and round-trip as a plain float
    # anyway. Demoting it here makes that conversion explicit and total.
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_to_jsonable(v) for v in value]}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: _to_jsonable(v) for k, v in value.items()}
        # Non-string keys are legal in Python but not in JSON objects; fall
        # back to an association list so they survive verbatim.
        return {
            "__items__": [[_to_jsonable(k), _to_jsonable(v)]
                          for k, v in value.items()]
        }
    raise TypeError(
        "Cannot serialise value of type %r into the model format."
        % (type(value).__name__,)
    )


def _from_jsonable(value):
    """Invert :func:`_to_jsonable`.

    Parameters
    ----------
    value : object
        Structure decoded from the JSON metadata blob.

    Returns
    -------
    object
        The original value, with ``numpy`` arrays, tuples and non-string-keyed
        dictionaries reconstructed.
    """
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    if isinstance(value, dict):
        # Only a dict with EXACTLY a tag's keys is treated as that tag, so an
        # ordinary mapping that happens to contain a "__tuple__" key decodes as
        # the mapping it is.
        keys = set(value)
        if keys == {"__ndarray__", "dtype", "shape"}:
            return np.array(
                value["__ndarray__"], dtype=np.dtype(value["dtype"])
            ).reshape(value["shape"])
        if keys == {"__tuple__"}:
            return tuple(_from_jsonable(v) for v in value["__tuple__"])
        if keys == {"__items__"}:
            return {
                _hashable(_from_jsonable(k)): _from_jsonable(v)
                for k, v in value["__items__"]
            }
        return {k: _from_jsonable(v) for k, v in value.items()}
    return value


def _hashable(key):
    """Make a decoded dictionary key hashable again.

    A JSON array nested inside an ``__items__`` pair decodes to a list, and a
    tagged array decodes to an ``ndarray``; neither can be a dict key, so both
    become tuples.
    """
    if isinstance(key, np.ndarray):
        return tuple(_hashable(k) for k in key.tolist())
    if isinstance(key, list):
        return tuple(_hashable(k) for k in key)
    return key


def _deep_copy_value(value):
    """Value-copy a property / bookkeeping value (``numpy``-aware)."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return _copy.deepcopy(value)


# --------------------------------------------------------------------------
# In-memory snapshot / restore
# --------------------------------------------------------------------------


def serializable_state(model):
    """Collect the model state into a plain mapping (the snapshot payload).

    Every value is deep-copied, so the result shares no buffer with ``model``
    and later mutation of either side cannot leak into the other.

    Parameters
    ----------
    model : RealTimeCOIN
        Model to capture. Not mutated.

    Returns
    -------
    dict
        Keys ``"properties"``, ``"D"``, ``"pending_q"``, ``"trial"``,
        ``"cue_values"``, ``"state_version"`` and ``"alignment_seed"``, plus
        ``"rng_state"`` (the ``model.rng`` bit-generator state, which MATLAB has
        no counterpart for) and ``"format"``.
    """
    from .model import PROPERTY_NAMES   # local import: model imports persist

    return {
        "format": FORMAT_VERSION,
        "properties": {
            name: _deep_copy_value(getattr(model, name)) for name in PROPERTY_NAMES
        },
        "D": copy_state(model.D) if model.D is not None else None,
        "pending_q": model.pending_q,
        "trial": int(model.trial),
        "cue_values": list(model.cue_values),
        "state_version": int(model.state_version),
        "alignment_seed": _copy.deepcopy(model.alignment_seed),
        "rng_state": _copy.deepcopy(model.rng.bit_generator.state),
    }


def restore_serializable_state(model, state):
    """Rebuild the model in place from a :func:`serializable_state` mapping.

    The inverse of :func:`serializable_state`. Missing ``state_version``
    defaults to ``trial`` and missing ``alignment_seed`` to ``None``, so older
    payloads still load. ``rng_state`` is likewise optional: when it is absent
    the model keeps its current random stream. The cached context alignment is
    invalidated.

    Everything is deep-copied on the way in, so ``state`` stays reusable - the
    same snapshot can be loaded into several models.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.
    state : dict
        Mapping produced by :func:`serializable_state`.

    Returns
    -------
    None

    Raises
    ------
    ModelFormatError
        If ``state`` is not a mapping or lacks the required keys.
    """
    from .model import PROPERTY_NAMES   # local import: model imports persist

    if not isinstance(state, dict):
        raise ModelFormatError(
            "Snapshot must be a mapping, got %r." % (type(state).__name__,)
        )
    for required in ("properties", "D", "trial"):
        if required not in state:
            raise ModelFormatError(
                "Snapshot is missing the required key %r." % (required,)
            )

    properties = state["properties"]
    if not isinstance(properties, dict):
        raise ModelFormatError(
            "Snapshot field 'properties' must be a mapping, got %r."
            % (type(properties).__name__,)
        )
    # The property names decide what gets written onto the model, and in a file
    # they are untrusted input. Without this check a hand-edited archive could
    # name ANY attribute - 'rng', 'D', or a bound method - and have it silently
    # overwritten, surfacing as a failure far from the load.
    unknown = sorted(set(properties) - set(PROPERTY_NAMES))
    if unknown:
        raise ModelFormatError(
            "Snapshot sets unknown model propert%s %s; expected a subset of %s."
            % ("y" if len(unknown) == 1 else "ies", unknown, list(PROPERTY_NAMES))
        )
    missing = sorted({"num_particles", "max_contexts", "state_dim"} - set(properties))
    if missing:
        raise ModelFormatError(
            "Snapshot is missing the propert%s %s that the particle-state "
            "shapes depend on." % ("y" if len(missing) == 1 else "ies", missing)
        )

    for name, value in properties.items():
        setattr(model, name, _deep_copy_value(value))

    particles = state["D"]
    if isinstance(particles, ParticleState):
        model.D = copy_state(particles)
    elif isinstance(particles, dict):
        try:
            model.D = copy_state(ParticleState(**particles))
        except TypeError as exc:
            raise ModelFormatError(
                "Snapshot field 'D' names a field that is not part of the "
                "particle state (%s)." % (exc,)
            ) from exc
    elif particles is None:
        model.D = None
    else:
        raise ModelFormatError(
            "Snapshot field 'D' must be a ParticleState or a mapping, got %r."
            % (type(particles).__name__,)
        )

    _validate_particle_shapes(model)

    model.pending_q = state.get("pending_q")
    model.trial = int(state["trial"])
    model.cue_values = list(state.get("cue_values", []))
    model.state_version = int(state.get("state_version", model.trial))
    model.alignment_seed = _copy.deepcopy(state.get("alignment_seed"))

    rng_state = state.get("rng_state")
    if rng_state is not None:
        _restore_rng_state(model, _copy.deepcopy(rng_state))

    _alignment.invalidate_context_alignment(model)


def _validate_particle_shapes(model):
    """Check the restored particle arrays against the restored properties.

    ``num_particles``, ``max_contexts`` and ``state_dim`` come from the same
    payload as ``D``, but nothing forces them to agree - a truncated or
    hand-edited file can pair one model's properties with another's particles.
    Checking three anchor arrays here turns that into a clear error at load time
    instead of a broadcasting failure several trials into the run.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose freshly restored ``D`` and properties are compared.

    Returns
    -------
    None

    Raises
    ------
    ModelFormatError
        If an anchor array does not match the declared dimensions.
    """
    d = model.D
    if d is None:
        return
    p_count = model.num_particles
    c_slots = model.max_contexts + 1
    anchors = (
        ("n_active", (p_count,)),
        ("context", (p_count,)),
        ("n_context", (p_count, c_slots, c_slots)),
        ("prior_probabilities", (p_count, c_slots)),
    )
    for name, expected in anchors:
        value = getattr(d, name)
        if value is None:
            continue
        if value.shape != expected:
            raise ModelFormatError(
                "Restored particle field %r has shape %s, but the restored "
                "properties (num_particles=%d, max_contexts=%d) require %s."
                % (name, value.shape, p_count, model.max_contexts, expected)
            )


def _restore_rng_state(model, rng_state):
    """Point ``model.rng`` at the saved bit-generator state.

    Assigning the state in place keeps the ``Generator`` object identity when
    the bit generator matches (the common case). When it does not - a snapshot
    from a model built on a different bit generator - a fresh generator of the
    saved kind is constructed instead, so the stream is still reproduced
    exactly.

    Parameters
    ----------
    model : RealTimeCOIN
        Model whose ``rng`` is repositioned.
    rng_state : dict
        A ``numpy.random.BitGenerator.state`` mapping.

    Returns
    -------
    None

    Raises
    ------
    ModelFormatError
        If the saved bit generator is not available in this ``numpy`` build.
    """
    name = rng_state.get("bit_generator") if isinstance(rng_state, dict) else None
    current = type(model.rng.bit_generator).__name__
    if name == current:
        # The inner state dict is untrusted too: numpy raises a bare KeyError /
        # ValueError for a malformed one, which would escape load_model unlabelled.
        try:
            model.rng.bit_generator.state = rng_state
        except (KeyError, ValueError, TypeError) as exc:
            raise ModelFormatError(
                "Saved random stream for %s is malformed (%s)." % (name, exc)
            ) from exc
        return
    factory = getattr(np.random, name, None) if isinstance(name, str) else None
    # The name comes out of the file, so it is untrusted: resolve it only to an
    # actual BitGenerator class. Without this check a forged file could name any
    # attribute of numpy.random (``seed``, ``shuffle``, ...) and get it called.
    if not (isinstance(factory, type) and issubclass(factory, np.random.BitGenerator)):
        raise ModelFormatError(
            "Saved random stream names bit generator %r, which is not an "
            "available numpy bit generator." % (name,)
        )
    bit_generator = factory()
    try:
        bit_generator.state = rng_state
    except (KeyError, ValueError, TypeError) as exc:
        raise ModelFormatError(
            "Saved random stream for %s is malformed (%s)." % (name, exc)
        ) from exc
    model.rng = np.random.Generator(bit_generator)


def snapshot(model):
    """Capture the full model state as a plain, serialisable mapping.

    The in-memory counterpart of :func:`save_model` / :func:`load_model`: use it
    to checkpoint a model and later restore it with :func:`load_snapshot`, or to
    hand model state to and from parallel workers without disk I/O.

    Round-trip guarantee: after ``load_snapshot(other, snapshot(model))``,
    ``other`` produces identical outputs from every query method as ``model``.
    Because the snapshot carries the random-stream position, ``other`` also
    continues the run identically observation for observation.

    Parameters
    ----------
    model : RealTimeCOIN
        Model to capture. Not mutated.

    Returns
    -------
    dict
        Deep copy of the model state; shares no array buffer with ``model``.
    """
    return serializable_state(model)


def load_snapshot(model, s):
    """Restore full model state in place from a snapshot mapping.

    Overwrites every public property, the particle state and the trial / cue /
    alignment bookkeeping. The cached context alignment is invalidated so it is
    recomputed on the next query.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.
    s : dict
        Mapping produced by :func:`snapshot`.

    Returns
    -------
    None
    """
    restore_serializable_state(model, s)


# --------------------------------------------------------------------------
# File persistence
# --------------------------------------------------------------------------


def _clone_for_save(model):
    """Build a throwaway copy of ``model`` that shares no mutable state.

    Used so a stationary save can compute the stationarised state without
    touching the live object. The clone gets its own particle state, its own
    cue registry and its own random generator positioned at the live stream's
    current point - so the saved file is a deterministic function of the live
    model, while the live model's stream stays exactly where it was.

    Parameters
    ----------
    model : RealTimeCOIN
        Model to clone. Not mutated.

    Returns
    -------
    RealTimeCOIN
        The clone.
    """
    clone = _copy.copy(model)          # new instance, shallow attribute copy
    clone.D = copy_state(model.D) if model.D is not None else None
    clone.cue_values = list(model.cue_values)
    clone.alignment_cache = None
    clone.alignment_seed = _copy.deepcopy(model.alignment_seed)
    # The only other mutable attributes a shallow copy would share. Nothing
    # currently writes into them in place, but the point of the clone is that
    # it does not DEPEND on that staying true.
    for name in ("process_noise_covariance", "observation_noise_covariance"):
        value = getattr(model, name)
        setattr(clone, name, None if value is None else value.copy())
    # Fork the stream at its current point: deterministic given the live model,
    # and it leaves the live generator exactly where it was.
    bit_generator = type(model.rng.bit_generator)()
    bit_generator.state = _copy.deepcopy(model.rng.bit_generator.state)
    clone.rng = np.random.Generator(bit_generator)
    return clone


def save_model(model, filename, set_stationary_first=True):
    """Serialise the model state to a file.

    By default the state written out is first placed at its stationary prior via
    :func:`set_stationary`, so the saved snapshot is contingency-independent.

    The live object is never modified, in either mode. MATLAB gets the same
    effect by mutating the object and restoring it afterwards; here the
    stationary reset runs on a private clone (:func:`_clone_for_save`), so an
    interrupted save cannot leave the model rewound, and ``model.rng`` is not
    advanced - saving does not perturb an in-progress run.

    Parameters
    ----------
    model : RealTimeCOIN
        Model to serialise. Not mutated.
    filename : str or os.PathLike
        Destination path. Written verbatim - no ``.npz`` suffix is appended.
        The write goes to a sibling temporary file that is renamed into place,
        so a failure part-way through cannot destroy an existing checkpoint.
    set_stationary_first : bool, optional
        When false the current (in-progress) state is saved verbatim.
        Defaults to true. This is MATLAB's third positional ``setStationary``
        argument; it is named out to keep the facade's ``set_stationary=True``
        keyword unambiguous against the :func:`set_stationary` function.

    Returns
    -------
    None
    """
    source = model
    if set_stationary_first:
        source = _clone_for_save(model)
        set_stationary(source)

    payload = serializable_state(source)
    arrays, meta = _split_payload(payload)

    # Write to a sibling temporary file and rename it into place, so a failure
    # mid-write leaves any existing checkpoint intact rather than truncated.
    target = os.fspath(filename)
    temporary = target + ".tmp"
    try:
        # savez appends ".npz" to a *name* but not to an open file object;
        # writing through the handle keeps the caller's filename exactly as
        # given.
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays, **{_META_KEY: _encode_meta(meta)})
        os.replace(temporary, target)   # atomic on POSIX and on Windows
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _encode_meta(meta):
    """Encode the metadata mapping as a ``uint8`` array of UTF-8 JSON."""
    blob = json.dumps(meta, allow_nan=True).encode("utf-8")
    return np.frombuffer(blob, dtype=np.uint8)


def _split_payload(payload):
    """Split a snapshot payload into ``npz`` arrays and a JSON metadata dict.

    Driven by :func:`dataclasses.fields` over
    :class:`~realtimecoin.state.ParticleState`, so a field added to the
    dataclass is picked up automatically and lands in exactly one of the three
    ``d_*_fields`` lists.

    Parameters
    ----------
    payload : dict
        Mapping from :func:`serializable_state`.

    Returns
    -------
    arrays : dict
        ``{"D.<field>": ndarray}`` for every array-valued particle field.
    meta : dict
        JSON-encodable metadata.
    """
    particles = payload["D"]
    arrays = {}
    scalar_fields = {}
    none_fields = []
    for field in _dataclass_fields(ParticleState):
        value = None if particles is None else getattr(particles, field.name)
        if value is None:
            none_fields.append(field.name)
        elif isinstance(value, np.ndarray):
            arrays[_D_PREFIX + field.name] = value
        else:
            scalar_fields[field.name] = _to_jsonable(value)

    meta = {
        "format": FORMAT_VERSION,
        "properties": {
            name: _to_jsonable(value)
            for name, value in payload["properties"].items()
        },
        "trial": payload["trial"],
        "pending_q": _to_jsonable(payload["pending_q"]),
        "cue_values": _to_jsonable(payload["cue_values"]),
        "state_version": payload["state_version"],
        "alignment_seed": _to_jsonable(payload["alignment_seed"]),
        "rng_state": _to_jsonable(payload["rng_state"]),
        "d_array_fields": [key[len(_D_PREFIX):] for key in arrays],
        "d_scalar_fields": scalar_fields,
        "d_none_fields": none_fields,
        "d_is_none": particles is None,
    }
    return arrays, meta


def load_model(model, filename):
    """Restore model state in place from a file written by :func:`save_model`.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place; it continues from the saved trial / posterior.
    filename : str or os.PathLike
        Source path.

    Returns
    -------
    None

    Raises
    ------
    ModelFormatError
        If the file does not hold a recognisable model snapshot: it is not a
        readable ``npz`` archive, carries no metadata blob, declares an
        unknown format version, or is missing an array the metadata promises.
    """
    restore_serializable_state(model, _read_payload(filename))


def _read_payload(filename):
    """Read and validate a model file, returning a snapshot payload.

    Parameters
    ----------
    filename : str or os.PathLike
        Source path.

    Returns
    -------
    dict
        A payload in the shape :func:`restore_serializable_state` expects.

    Raises
    ------
    ModelFormatError
        On any structural problem with the file.
    """
    try:
        archive = np.load(filename, allow_pickle=False)
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        # Problems with the PATH, not with the CONTENT: let the caller see the
        # real OS error rather than a misleading "bad format" diagnosis.
        raise
    except Exception as exc:   # noqa: BLE001 - any decode failure is a format error
        raise ModelFormatError(
            "File %r is not a readable RealTimeCOIN model archive (%s)."
            % (str(filename), exc)
        ) from exc

    # np.load happily returns a bare ndarray for a .npy file, which is not a
    # context manager and has no .files - check before touching either.
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ModelFormatError(
            "File %r holds a single numpy array, not a RealTimeCOIN model "
            "archive." % (str(filename),)
        )

    with archive:
        names = set(archive.files)
        if _META_KEY not in names:
            raise ModelFormatError(
                "File %r is not a RealTimeCOIN model save: it has no %r entry."
                % (str(filename), _META_KEY)
            )
        try:
            meta = json.loads(bytes(archive[_META_KEY]).decode("utf-8"))
        except Exception as exc:   # noqa: BLE001 - corrupt blob
            raise ModelFormatError(
                "File %r has an unreadable metadata block (%s)."
                % (str(filename), exc)
            ) from exc
        if not isinstance(meta, dict):
            raise ModelFormatError(
                "File %r has a malformed metadata block." % (str(filename),)
            )

        version = meta.get("format")
        if version != FORMAT_VERSION:
            raise ModelFormatError(
                "File %r has model format %r; this build reads %r."
                % (str(filename), version, FORMAT_VERSION)
            )

        particles = _read_particles(filename, archive, names, meta)

    return {
        "format": version,
        "properties": {
            name: _from_jsonable(value)
            for name, value in meta.get("properties", {}).items()
        },
        "D": particles,
        "pending_q": _from_jsonable(meta.get("pending_q")),
        "trial": meta.get("trial", 0),
        "cue_values": _from_jsonable(meta.get("cue_values", [])),
        "state_version": meta.get("state_version", meta.get("trial", 0)),
        "alignment_seed": _from_jsonable(meta.get("alignment_seed")),
        "rng_state": _from_jsonable(meta.get("rng_state")),
    }


def _read_particles(filename, archive, names, meta):
    """Rebuild the :class:`~realtimecoin.state.ParticleState` from an archive.

    Checks that the three ``d_*_fields`` lists in ``meta`` partition the
    dataclass fields exactly and that every promised array is present, so a
    truncated archive or a state written by a build with different fields is
    rejected instead of loading as a silently incomplete model.

    Parameters
    ----------
    filename : str or os.PathLike
        Source path, used only in error messages.
    archive : numpy.lib.npyio.NpzFile
        The open archive.
    names : set of str
        ``archive.files`` as a set.
    meta : dict
        Decoded metadata block.

    Returns
    -------
    ParticleState or None
        The reconstructed state (``None`` only if the save held none).

    Raises
    ------
    ModelFormatError
        If the field lists or the array entries are inconsistent.
    """
    if meta.get("d_is_none"):
        return None

    array_fields = list(meta.get("d_array_fields", []))
    scalar_fields = dict(meta.get("d_scalar_fields", {}))
    none_fields = list(meta.get("d_none_fields", []))

    declared = [f.name for f in _dataclass_fields(ParticleState)]
    saved = array_fields + list(scalar_fields) + none_fields
    # Duplicates first: they also break the partition check below, but with a
    # confusing "missing [], unknown []" message.
    if len(saved) != len(set(saved)):
        repeated = sorted({name for name in saved if saved.count(name) > 1})
        raise ModelFormatError(
            "File %r lists particle-state field(s) %s more than once."
            % (str(filename), repeated)
        )
    if sorted(saved) != sorted(declared):
        missing = sorted(set(declared) - set(saved))
        unknown = sorted(set(saved) - set(declared))
        raise ModelFormatError(
            "File %r describes a different particle state: missing field(s) %s, "
            "unknown field(s) %s." % (str(filename), missing, unknown)
        )

    particles = ParticleState()
    for name in array_fields:
        key = _D_PREFIX + name
        if key not in names:
            raise ModelFormatError(
                "File %r is missing the array entry %r for particle field %r."
                % (str(filename), key, name)
            )
        # np.load is lazy: THIS is where a member is actually decompressed, so
        # a corrupt deflate stream (zlib.error) or a bad CRC-32
        # (zipfile.BadZipFile) surfaces here and must be reported as a format
        # problem rather than escaping as a raw compression error.
        try:
            value = np.array(archive[key])
        except (zipfile.BadZipFile, zlib.error, EOFError, ValueError, OSError) as exc:
            raise ModelFormatError(
                "File %r has a corrupt entry %r (%s)." % (str(filename), key, exc)
            ) from exc
        # np.array() copies out of the lazily-decompressed archive member, so
        # the state stays valid after the NpzFile is closed.
        setattr(particles, name, value)
    for name, value in scalar_fields.items():
        setattr(particles, name, _from_jsonable(value))
    return particles


# --------------------------------------------------------------------------
# Stationarisation
# --------------------------------------------------------------------------


def set_stationary(model):
    """Reset the particle filter to its stationary prior.

    Re-initialises the context and state beliefs of every particle to the
    stationary distribution implied by the current hyperparameters, rewinds the
    trial counter to 0 and clears any pending cue. For each particle the
    stationary context distribution is derived from its local transition matrix,
    a context is sampled from it, and the per-context state moments are set to
    their stationary Kalman values. Both the scalar and the multi-dimensional
    branches are handled.

    What is DELIBERATELY kept: everything learned. The transition and cue
    counts, the global transition / cue probabilities, the sampled dynamics
    (``retention`` / ``drift`` / ``Theta``), the biases, the per-particle
    context counts ``n_active`` and the cue registry (``cue_values``,
    ``n_cues``) all survive untouched. Only the *beliefs that depend on where in
    the experiment we are* - the context probabilities, the sampled context, the
    state moments, the trial counter and the staged cue - are rewound.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place. Uses ``model.rng``.

    Returns
    -------
    None
    """
    # The local matrices are what the stationary distribution is read off, so
    # rebuild them from the current counts first.
    _context.update_local_transition_matrix(model)
    _context.update_local_cue_matrix(model)

    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    d = model.D

    d.prior_probabilities = np.zeros((p_count, c_slots))          # (P, C)
    d.predicted_probabilities = np.zeros((p_count, c_slots))      # (P, C)
    d.responsibilities = np.zeros((p_count, c_slots))             # (P, C)
    d.probability_cue = np.ones((p_count, c_slots))               # (P, C)
    d.probability_state_feedback = np.ones((p_count, c_slots))    # (P, C)
    d.i_resampled = np.arange(p_count)                            # (P,)

    context = np.zeros(p_count, dtype=int)
    for p in range(p_count):
        k_active = int(d.n_active[p])
        # Valid slots: the instantiated contexts, plus the novel one when the
        # particle has not yet used its full budget.
        valid = np.zeros(c_slots, dtype=bool)
        valid[:k_active] = True
        if k_active < model.max_contexts:
            valid[k_active] = True
        valid_idx = np.flatnonzero(valid)

        transition = d.local_transition_matrix[p][np.ix_(valid_idx, valid_idx)]
        stationary_context_prob = _statics.stationary_distribution(transition)

        d.prior_probabilities[p, valid_idx] = stationary_context_prob
        d.predicted_probabilities[p, valid_idx] = stationary_context_prob
        d.responsibilities[p, valid_idx] = stationary_context_prob

        # Inverse-CDF draw over the valid slots. The last bin is pinned to 1 so
        # round-off in the cumulative sum can never leave the draw unmatched.
        cumulative = np.cumsum(stationary_context_prob)
        cumulative[-1] = 1.0
        context[p] = valid_idx[int(np.argmax(model.rng.random() <= cumulative))]

    d.context = context
    d.previous_context = context.copy()

    if model.state_dim > 1:
        _set_stationary_states_md(model)
        _pipeline_md.predict_state_feedback_md(model)
    else:
        _set_stationary_states_scalar(model)
        _pipeline_scalar.predict_state_feedback(model)

    model.pending_q = None
    model.trial = 0
    _alignment.invalidate_context_alignment(model)


def _set_stationary_states_scalar(model):
    """Set every scalar state moment to its stationary value.

    ``mean = d / (1 - a)`` and ``var = sigma_process^2 / (1 - a^2)``, evaluated
    per particle and context from the currently sampled ``retention`` and
    ``drift`` (entries with no finite stationary value are left at zero by the
    numerics helpers).

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.

    Returns
    -------
    None
    """
    d = model.D
    d.state_filtered_mean = _numerics.stationary_state_mean(d.retention, d.drift)
    d.state_filtered_var = _numerics.stationary_state_var(
        d.retention, model.sigma_process_noise
    )
    # Copies, not aliases: MATLAB struct assignment is a value copy, and the
    # pipeline updates these four arrays independently.
    d.previous_state_filtered_mean = d.state_filtered_mean.copy()
    d.previous_state_filtered_var = d.state_filtered_var.copy()
    d.state_mean = d.state_filtered_mean.copy()
    d.state_var = d.state_filtered_var.copy()


def _set_stationary_states_md(model):
    """Set every multi-dimensional state moment to its stationary value.

    Per particle and context, the mean solves ``m = A m + d`` and the covariance
    solves the discrete Lyapunov equation ``V = A V A' + Q``, with ``[A | d]``
    read off the sampled ``Theta`` and ``Q`` the process-noise covariance.

    Parameters
    ----------
    model : RealTimeCOIN
        Model mutated in place.

    Returns
    -------
    None
    """
    n = model.state_dim
    c_slots = model.max_contexts + 1
    p_count = model.num_particles
    q_cov = _state.process_noise_cov(model)                      # (N, N)
    d = model.D

    d.state_filtered_mean = np.zeros((p_count, c_slots, n))          # (P, C, N)
    d.state_filtered_cov = np.zeros((p_count, c_slots, n, n))        # (P, C, N, N)
    for p in range(p_count):
        for c in range(c_slots):
            a_matrix = d.Theta[p, c, :, :n]     # (N, N)
            drift = d.Theta[p, c, :, n]         # (N,)
            d.state_filtered_mean[p, c] = _numerics.stationary_state_mean_md(
                a_matrix, drift
            )
            d.state_filtered_cov[p, c] = _numerics.stationary_state_cov_md(
                a_matrix, q_cov
            )

    d.previous_state_filtered_mean = d.state_filtered_mean.copy()
    d.previous_state_filtered_cov = d.state_filtered_cov.copy()
    d.state_mean = d.state_filtered_mean.copy()
    d.state_cov = d.state_filtered_cov.copy()

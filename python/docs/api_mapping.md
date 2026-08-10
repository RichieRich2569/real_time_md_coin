# MATLAB -> Python API mapping

Every public member of the MATLAB class folders `@RealTimeCOIN/` and
`@RealTimeCOINEnsemble/`, and where it went in the `realtimecoin` package.

Conventions used throughout the right-hand column:

- `model` is a `RealTimeCOIN`; `ens` is a `RealTimeCOINEnsemble`.
- MATLAB returns row vectors and `containers.Map` objects; Python returns
  `numpy.ndarray` and `dict`. **Map/dict keys are 0-based in Python** where
  MATLAB's are 1-based.
- MATLAB density grids are `N`-by-`K` with one query point per **column**;
  Python grids are `(K,)` for the scalar model and `(K, N)` -- one query point
  per **row** -- for the multi-dimensional model.
- MATLAB draws from the global random stream; Python draws from
  `model.rng`, a `numpy.random.Generator`.

---

## `@RealTimeCOIN`

### Construction and lifecycle

| MATLAB | Python |
| --- | --- |
| `RealTimeCOIN(Name, Value, ...)` | `RealTimeCOIN(**kwargs)` -- keyword-only; adds `rng=` |
| `obj.Trial` (dependent, read-only) | `model.Trial` (read-only property) |
| `obj.num_particles`, `obj.max_contexts`, ... (20 public properties) | identically named attributes; `realtimecoin.model.PROPERTY_NAMES` lists them in MATLAB declaration order |
| `observe_q(obj, q)` | `model.observe_q(q)` |
| `observe_y(obj, y)` | `model.observe_y(y)` |
| `set_stationary(obj)` | `model.set_stationary()` |

### Persistence

| MATLAB | Python |
| --- | --- |
| `s = snapshot(obj)` | `s = model.snapshot()` -- returns a `dict`, and additionally carries `rng_state` |
| `loadSnapshot(obj, s)` | `model.load_snapshot(s)` |
| `saveModel(obj, filename, setStationary)` | `model.save_model(filename, set_stationary=True)` -- writes `.npz` + JSON (`rtcoin-model-v1`), not `.mat`; never mutates the live model |
| `loadModel(obj, filename)` | `model.load_model(filename)` |

### Point predictions and moments

| MATLAB | Python |
| --- | --- |
| `motor_output(obj)` | `model.motor_output()` |
| `predictive_motor_output(obj, q)` | `model.predictive_motor_output(q=None)` -- `q` is a raw cue value |
| `state_moments(obj)` | `mu, v = model.state_moments()` |
| `predictive_feedback_moments(obj, q)` | `mu, sigma = model.predictive_feedback_moments(q=None)` -- `q` is a **0-based cue label** |
| `explicit_component(obj)` | `model.explicit_component()` |
| `implicit_component(obj)` | `model.implicit_component()` |
| `state_cstar1(obj)` | `model.state_cstar1()` |
| `state_cstar2(obj, q)` | `model.state_cstar2(q=None)` -- raw cue value |
| `state_cstar3(obj)` | `model.state_cstar3()` |
| `predicted_probability_cstar1(obj)` | `model.predicted_probability_cstar1()` |
| `predicted_probability_cstar3(obj)` | `model.predicted_probability_cstar3()` |
| `kalman_gain_cstar1(obj)` | `model.kalman_gain_cstar1()` -- scalar model only |
| `kalman_gain_cstar2(obj, q)` | `model.kalman_gain_cstar2(q=None)` -- scalar model only |

### Densities and CDFs

| MATLAB | Python |
| --- | --- |
| `state_probability(obj, values)` | `model.state_probability(values)` |
| `state_feedback_probability(obj, values)` | `model.state_feedback_probability(values)` |
| `novel_state_probability(obj, values)` | `model.novel_state_probability(values)` |
| `novel_state_feedback_probability(obj, values)` | `model.novel_state_feedback_probability(values)` |
| `state_given_context_probability(obj, values)` | `model.state_given_context_probability(values)` -> `dict` |
| `state_feedback_given_context_probability(obj, values)` | `model.state_feedback_given_context_probability(values)` -> `dict` |
| `retention_given_context_probability(obj, values)` | `model.retention_given_context_probability(values)` -> `dict`; scalar model only |
| `drift_given_context_probability(obj, values)` | `model.drift_given_context_probability(values)` -> `dict`; scalar model only |
| `bias_given_context_probability(obj, values)` | `model.bias_given_context_probability(values)` -> `dict`; scalar model only, needs `infer_bias=True` |
| `bias_probability(obj, values)` | `model.bias_probability(values)`; scalar model only, needs `infer_bias=True` |
| `predictive_state_feedback_cdf(obj, y, q)` | `model.predictive_state_feedback_cdf(y, q=None)` |
| `predictive_cue_p_value(obj, q, u)` | `model.predictive_cue_p_value(q, u=None)` -- `u` defaults to a draw from `model.rng` |

### Context summaries -- global (aligned)

| MATLAB | Python |
| --- | --- |
| `predicted_context_probabilities_vector(obj)` | `model.predicted_context_probabilities_vector()` |
| `predicted_context_probabilities_map(obj)` | `model.predicted_context_probabilities_map()` -> `dict`, 0-based keys |
| `responsibilities_vector(obj)` | `model.responsibilities_vector()` |
| `responsibilities_map(obj)` | `model.responsibilities_map()` -> `dict`, 0-based keys |
| `sampled_context_count(obj)` | `model.sampled_context_count()` |
| `stationary_context_probabilities(obj)` | `model.stationary_context_probabilities()` |
| `global_transition_probabilities(obj)` | `model.global_transition_probabilities()` |
| `global_cue_probabilities(obj)` | `model.global_cue_probabilities()` |
| `local_transition_probabilities(obj)` | `model.local_transition_probabilities()` |
| `local_cue_probabilities(obj)` | `model.local_cue_probabilities()` |

### Context summaries -- local (unaligned)

| MATLAB | Python |
| --- | --- |
| `predicted_context_probabilities_local(obj)` | `model.predicted_context_probabilities_local()` |
| `context_responsibilities_local(obj)` | `model.context_responsibilities_local()` |
| `sampled_context_count_local(obj)` | `model.sampled_context_count_local()` |

### Alignment and diagnostics

| MATLAB | Python |
| --- | --- |
| `context_alignment(obj)` | `model.context_alignment()` -> `dict` (`K`, `assignment`, `modal_particle_mask`, `modal_particle_indices`, `modal_particle_weights`, `global_contexts`, `converged`, `iterations`, `used_seed`, `cache_state_version`, `computed_at_trial`) |
| `diagnostics(obj)` | `model.diagnostics()` -> `dict`; per-particle arrays have the modal-particle axis **leading** |

### Deprecated aliases -- not ported

| MATLAB | Python |
| --- | --- |
| `predicted_context_probabilities(obj)` | **not ported** (use `model.predicted_context_probabilities_vector()`) |
| `context_predicted_probabilities(obj)` | **not ported** (use `model.predicted_context_probabilities_map()`) |
| `responsibilities(obj)` | **not ported** (use `model.responsibilities_vector()`) |
| `context_responsibilities(obj)` | **not ported** (use `model.responsibilities_map()`) |

In MATLAB these four forward to the suffixed methods and emit a one-time
deprecation warning per session. The Python package has no legacy callers, so
they were dropped rather than carried forward.

### Static helpers

Class-static in MATLAB; module-level functions in `realtimecoin.statics` in
Python. The three that draw randomness take a `numpy.random.Generator` as their
**first** argument, since there is no global stream.

| MATLAB | Python |
| --- | --- |
| `RealTimeCOIN.systematic_resampling(w)` | `statics.systematic_resampling(rng, weights)` |
| `RealTimeCOIN.normal_pdf(x, m, v)` | `statics.normal_pdf(x, m, v)` |
| `RealTimeCOIN.normal_cdf(x, m, v)` | `statics.normal_cdf(x, m, v)` |
| `RealTimeCOIN.log_sum_exp(logP, dim)` | `statics.log_sum_exp(log_p, axis=-1)` -- MATLAB's default `dim=1` reduces over contexts, which under the particles-leading layout is the **trailing** axis |
| `RealTimeCOIN.stationary_distribution(T)` | `statics.stationary_distribution(t)` |
| `RealTimeCOIN.sample_num_tables(base, counts)` | `statics.sample_num_tables(rng, base, counts)` |

### Private helpers

`@RealTimeCOIN/private/` (~100 internal helpers) has no one-to-one public
counterpart. The algorithms live in the modules the facade delegates to:

| MATLAB private group | Python module |
| --- | --- |
| `predictContext`, `updateLocalTransitionMatrix`, `updateLocalCueMatrix`, `previewCuePmf`, `kappa`, ... | `realtimecoin.context` |
| the scalar per-trial steps (`predictStates`, `resampleParticles`, ...) | `realtimecoin.pipeline_scalar` |
| the `*MD` per-trial steps | `realtimecoin.pipeline_md` |
| `resetParticles[MD]`, `ensureCueColumn`, `consumePendingCue`, `processNoiseCov`, `observationNoiseCov`, ... | `realtimecoin.state` |
| `ensureContextAlignment`, `computeContextAlignment`, `linearAssignment`, `assignmentCostMatrix[MD]`, `global*` accessors, ... | `realtimecoin.alignment` |
| `serializableState`, `restoreSerializableState` | `realtimecoin.persist` |
| `gammaSample`, `betaSample`, `dirichletSample`, `binomialSample`, `sampleMatrixNormal`, `sampleStableTheta`, ... | `realtimecoin.samplers` |
| `choljitter`, `regularizeCovariance`, `safeDivide`, `gaussianJeffreys*`, `stationaryStateMean*`, ... | `realtimecoin.numerics` |

### Test helper

| MATLAB | Python |
| --- | --- |
| `testutil.loadFixtureModel(properties, state, ...)` | `RealTimeCOIN.from_state(properties, state, trial=1, cue_values=(1,), alignment_seed=None, rng=None)` |

`from_state` is a classmethod on the model rather than a separate test package:
it builds a model whose particle state is fully controlled by the caller,
without going through a file. `state` may be a `ParticleState` or a partial
`dict` of field names; unspecified fields stay `None`. The rest of the MATLAB
`tests/+testutil` package (`assertClose`, `assertSize`, `mustError`,
`integrate2d`, `alignmentFixture`, ...) is replaced by plain `pytest`
assertions and the helpers in `python/tests/helpers.py` and
`python/tests/fixtures/alignment_fixtures.py`.

---

## `@RealTimeCOINEnsemble`

Documented from `docs/SPEC_ensemble.md` and the agreed Python API. Both phases
are implemented: the run-averaged queries (Phase 1) and the cross-run
context-aligned queries (Phase 2).

### Construction and state machine

| MATLAB | Python |
| --- | --- |
| `RealTimeCOINEnsemble(Name, Value, ...)` | `RealTimeCOINEnsemble(runs=1, seed=0, max_cores=0, segment_length=1, **member_kwargs)` |
| `ens.runs`, `ens.seed`, `ens.max_cores`, `ens.segment_length`, `ens.weights` | identically named attributes |
| `ens.Trial` (dependent) | `ens.Trial` (read-only property) |
| `observe_q(ens, q)` | `ens.observe_q(q)` |
| `observe_y(ens, y)` | `ens.observe_y(y)` |
| `traces = simulate(ens, qSeq, ySeq)` | `traces = ens.simulate(q_seq, y_seq)` -> `dict` with `motor_output`, `state_mean`, `state_var`, `Trial` |
| per-member `RandStream` substream (Threefry, `NumStreams = runs`, `StreamIndices = k`) | per-member generator from `numpy.random.SeedSequence(seed).spawn(runs)` |

### Run-averaged queries (Phase 1)

| MATLAB | Python |
| --- | --- |
| `motor_output(ens)` | `ens.motor_output()` |
| `[mu, v] = state_moments(ens)` | `mu, v = ens.state_moments()` |
| `state_probability(ens, values)` | `ens.state_probability(values)` |
| `state_feedback_probability(ens, values)` | `ens.state_feedback_probability(values)` |
| `novel_state_probability(ens, values)` | `ens.novel_state_probability(values)` |
| `novel_state_feedback_probability(ens, values)` | `ens.novel_state_feedback_probability(values)` |

### Context-indexed queries (Phase 2, cross-run aligned)

| MATLAB | Python |
| --- | --- |
| `responsibilities_vector(ens)` | `ens.responsibilities_vector()` |
| `predicted_context_probabilities_vector(ens)` | `ens.predicted_context_probabilities_vector()` |
| `sampled_context_count(ens)` | `ens.sampled_context_count()` |
| `stationary_context_probabilities(ens)` | `ens.stationary_context_probabilities()` |
| `state_given_context_probability(ens, values)` | `ens.state_given_context_probability(values)` -> `dict`, 0-based reference-frame keys |
| `state_feedback_given_context_probability(ens, values)` | `ens.state_feedback_given_context_probability(values)` -> `dict`, 0-based reference-frame keys |

### Not exposed

`@RealTimeCOINEnsemble/private/` (member construction, stream creation, the
serial and parallel executors, the cross-run aligner) has no public
counterpart in either language.

# Using `realtimecoin`

A guide to the Python translation of the MATLAB `@RealTimeCOIN` class folder.
It assumes you know the COIN model scientifically (Heald, Lengyel & Wolpert,
2021) and possibly the MATLAB implementation; it does not re-derive the model,
only pins down what the code does and how to drive it.

**Contents**

1. [The model and the per-trial pipeline](#1-the-model-and-the-per-trial-pipeline)
2. [The state-machine API](#2-the-state-machine-api)
3. [Constructor parameters](#3-constructor-parameters)
4. [Query catalogue](#4-query-catalogue)
5. [Scalar vs multi-dimensional](#5-scalar-vs-multi-dimensional)
6. [Context alignment](#6-context-alignment)
7. [Persistence](#7-persistence)
8. [The ensemble](#8-the-ensemble)
9. [The validation suite](#9-the-validation-suite)
10. [Differences from the MATLAB implementation](#10-differences-from-the-matlab-implementation)

---

## 1. The model and the per-trial pipeline

### 1.1 The generative model

Each particle carries a set of **contexts**, learned online and capped at
`max_contexts`. A context `c` owns its own linear-Gaussian state dynamics, its
own observation bias, and its own cue-emission distribution.

**State dynamics.** Within context `c` the latent state follows an AR(1) process
with retention `a_c` and drift `d_c`:

```
x_t = a_c * x_{t-1} + d_c + w_t,        w_t ~ N(0, sigma_process_noise^2)
```

In the multi-dimensional model (`state_dim == N > 1`) the retention becomes a
matrix `A_c` and the drift a vector; the two are stored jointly as the augmented
matrix `Theta_c = [A_c | d_c]` of shape `(N, N + 1)`:

```
x_t = A_c x_{t-1} + d_c + w_t,          w_t ~ N(0, Q)
```

with `Q = process_noise_covariance` if given, else `sigma_process_noise^2 * I_N`.
`A_c` is drawn from a matrix-normal conjugate posterior and constrained to be
stable (spectral radius < 1); in the scalar case the equivalent constraint is
`a_c` truncated to `[0, 1)`.

**Observation.** The observation map is the identity, so the feedback is the
state plus a per-context bias plus noise:

```
y_t = x_t + b_c + v_t,                  v_t ~ N(0, R)
```

with `R = observation_noise_covariance` if given, else
`(sigma_sensory_noise^2 + sigma_motor_noise^2) * I_N`. The bias `b_c` is
identically zero unless `infer_bias=True`.

**Context transitions (sticky HDP-HMM).** A global (franchise) distribution
`beta` over contexts is drawn from a stick-breaking process with concentration
`gamma_context`. Each context `r` then has a local transition distribution whose
posterior mean, given the observed transition counts `n[r, c]`, is proportional
to

```
alpha_context * beta[c]  +  kappa * (r == c)  +  n[r, c]
```

renormalised over `c`, where the stickiness concentration is

```
kappa = alpha_context * rho_context / (1 - rho_context)
```

Rows and columns of contexts a particle has not instantiated are zeroed before
normalising, so an unreachable context can never draw mass. The trailing slot of
each row is the **novel context**: the probability of transitioning into a
context that does not exist yet.

**Cue emission.** Sensory cues are handled by a second, non-sticky HDP with
concentrations `gamma_cue` (novelty) and `alpha_cue` (context-cue coupling).
Context `c` emits cue label `q` with posterior-mean probability proportional to
`alpha_cue * beta_cue[q] + n_cue[c, q]`, again with a trailing novel-cue slot.

**Priors.** Retention, drift and bias have Gaussian priors with means
`prior_mean_retention` / `prior_mean_drift` / `prior_mean_bias` and precisions
`prior_precision_retention` / `prior_precision_drift` / `prior_precision_bias`.
Setting a precision very high (e.g. `1e12`) effectively fixes the corresponding
parameter, which is exactly what the Kalman validators do to reduce the model to
a textbook Kalman filter.

### 1.2 Inference

Inference is a **vectorised particle filter** over `num_particles` particles.
Every particle holds a complete hypothesis: how many contexts exist, which one
is active, the per-context state beliefs, the sampled dynamics/bias parameters,
and the sufficient statistics for their conjugate updates. All of that lives in
one dataclass, `model.D` (a `ParticleState`), whose arrays are documented in the
module docstring of `realtimecoin/state.py`.

`observe_y` runs nine steps, in this order, in **both** the scalar and the
multi-dimensional branch:

```
predict_context  ->  predict_states  ->  predict_state_feedback
  ->  resample_particles  ->  sample_context
  ->  update_belief_about_states  ->  sample_states
  ->  update_sufficient_statistics  ->  sample_parameters
```

| Step | What it does |
| --- | --- |
| `predict_context` | rebuilds the local transition and cue matrices, forms the context prior from the current context's transition row, multiplies in `p(q | c)` when a cue is staged |
| `predict_states` | Kalman predict per context: `x = a x + d`, `P = a P a' + Q`; a novel context is seeded from its stationary state distribution |
| `predict_state_feedback` | `y_hat = x + b`, `S = P + R` |
| `resample_particles` | Gaussian feedback likelihood `p(y | c)`, responsibilities as a normalised joint likelihood, then systematic (low-variance) resampling |
| `sample_context` | samples the active context per particle; a novel context is instantiated by stick-breaking growth |
| `update_belief_about_states` | Kalman update, applied to the **sampled** context only; the others keep their predicted beliefs |
| `sample_states` | draws the latent state (and its lagged value) for the conjugate regression |
| `update_sufficient_statistics` | accumulates transition counts, cue counts and the dynamics/bias Gram matrices |
| `sample_parameters` | conjugate/Gibbs draws of the global transition and cue distributions, the dynamics and the bias |

Afterwards the trial counter advances and the cached context alignment
(section 6) is invalidated.

---

## 2. The state-machine API

One cycle per trial: stage the cue, then process the feedback, then query.

```python
import numpy as np
from realtimecoin import RealTimeCOIN

model = RealTimeCOIN(rng=7)

model.observe_q(1.0)          # stage a cue; no inference, no trial advance
model.observe_y(0.25)         # run the pipeline; Trial 0 -> 1

model.observe_y(0.31)         # cue-free trial (nothing staged)
model.observe_y(None)         # missing feedback; the trial still happens
model.observe_y(np.nan)       # equivalent to None
print(model.Trial)            # 4
```

- **`observe_q(q)`** records the raw cue value for the *next* trial in
  `pending_q`. It draws no randomness and does not advance anything. Distinct
  raw values are registered, in order of first appearance, as consecutive
  0-based internal cue labels. `observe_q(None)` or `observe_q(nan)` **clears**
  any staged cue, so the next trial is cue-free. Calling it twice before an
  `observe_y` simply overwrites the staged value.
- **`observe_y(y)`** consumes the staged cue, runs the nine-step pipeline,
  increments `Trial` and invalidates the alignment cache. Call it exactly once
  per trial.
- **`model.Trial`** is a read-only counter, starting at 0.

**Missing observations.** `None` or `nan` marks a missing observation. The trial
still runs: contexts are predicted, particles are resampled on the cue
likelihood alone, and the state beliefs are propagated without a measurement
update. In the multi-dimensional model individual `nan` entries mark
*partially* observed trials, and the Kalman update is applied only to the
observed dimensions:

```python
md = RealTimeCOIN(state_dim=3, num_particles=50, rng=3)
md.observe_y([0.4, 0.1, -0.2])            # all three dimensions observed
md.observe_y([0.5, np.nan, -0.1])         # dimension 1 missing this trial
md.observe_y(None)                        # nothing observed at all
mu, cov = md.state_moments()
print(mu.shape, cov.shape)                # (3,) (3, 3)
```

An empty array (`[]`, `np.array([])`) is treated as `None`, matching MATLAB's
`isempty(y)`. Anything else must have exactly `state_dim` elements or
`observe_y` raises `ValueError`.

**Randomness.** Everything the model draws goes through `model.rng`, a
`numpy.random.Generator`. There is no global-state dependence; seeding the
constructor (`rng=0`) makes an entire run reproducible.

---

## 3. Constructor parameters

All constructor arguments are keyword-only. The 20 model properties mirror the
MATLAB properties exactly, names and defaults included; the defaults are the
fitted group-average parameters reported in Heald et al. (2021), so do not round
them. An unrecognised keyword raises `NameValuePairsError`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `num_particles` | `100` | particle count `P` |
| `max_contexts` | `10` | maximum contexts per particle; the context axis has `max_contexts + 1` slots, the extra one being the novel context |
| `state_dim` | `1` | latent (and, since the observation map is the identity, observation) dimension `N`. `1` selects the scalar pipeline, `> 1` the multi-dimensional one |
| `gamma_context` | `0.1` | DP novelty concentration for contexts. Non-negative |
| `alpha_context` | `8.955` | DP transition concentration for contexts. Positive |
| `rho_context` | `0.2501` | sticky self-transition weight, in `[0, 1)` |
| `gamma_cue` | `0.1` | DP novelty concentration for cues. Non-negative |
| `alpha_cue` | `25` | DP cue-context concentration. Positive |
| `prior_mean_retention` | `0.9425` | prior mean of the retention factor `a` |
| `prior_mean_drift` | `0.0` | prior mean of the drift `d` |
| `prior_mean_bias` | `0.0` | prior mean of the observation bias |
| `prior_precision_retention` | `837.1 ** 2` | prior precision of `a`. Non-negative |
| `prior_precision_drift` | `1.2227e3 ** 2` | prior precision of `d`. Non-negative |
| `prior_precision_bias` | `70 ** 2` | prior precision of the bias. Non-negative |
| `sigma_process_noise` | `0.0089` | process-noise standard deviation. Non-negative |
| `sigma_sensory_noise` | `0.03` | sensory-noise standard deviation. Non-negative |
| `sigma_motor_noise` | `0.0` | motor-noise standard deviation. Non-negative |
| `process_noise_covariance` | `None` | explicit `(N, N)` process-noise covariance `Q` for the MD model. `None` selects `sigma_process_noise ** 2 * I`. Ignored (with a `UserWarning`) when `state_dim == 1` |
| `observation_noise_covariance` | `None` | explicit `(N, N)` observation-noise covariance `R`. `None` selects `(sigma_sensory_noise ** 2 + sigma_motor_noise ** 2) * I`. Ignored (with a `UserWarning`) when `state_dim == 1` |
| `infer_bias` | `False` | infer a per-context observation bias |

Plus one Python-only argument:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `rng` | `None` | random source, passed through `numpy.random.default_rng`. An `int` seeds a fresh generator and makes the run reproducible; a `Generator` is used as given |

```python
q = 0.01 * np.array([[1.0, 0.3], [0.3, 1.0]])
r = 0.05 * np.array([[1.0, -0.2], [-0.2, 1.0]])
md2 = RealTimeCOIN(
    state_dim=2,
    num_particles=100,
    process_noise_covariance=q,
    observation_noise_covariance=r,
    rng=11,
)
md2.observe_y([0.2, -0.1])
print(md2.motor_output().shape)           # (2,)
```

A supplied covariance must be `(N, N)`, finite, symmetric and positive
semidefinite; violations raise `ValueError` with a MATLAB identifier prefix
(`RealTimeCOIN:BadCovarianceSize`, `:BadCovarianceValue`,
`:CovarianceNotSymmetric`, `:CovarianceNotPSD`).

**The validators run once, at construction.** MATLAB's `arguments` blocks
re-fire on every assignment for the life of the handle object; here the
attributes are plain afterwards, so `model.rho_context = 1.5` after construction
is *not* checked and will produce nonsense deep in the filter. Rebuild the model,
or go through the persistence layer, rather than mutating hyperparameters in
place.

---

## 4. Query catalogue

Every query is read-only: it never mutates the particle state, and (with the
exception of `predictive_cue_p_value` called without `u`) never draws
randomness. Queries evaluate at the *current* trial.

The examples below assume a model that has seen a few trials:

```python
model = RealTimeCOIN(num_particles=100, rng=0)
for y in [0.0, 0.2, 0.5, 0.9, 1.0, 1.0]:
    model.observe_q(1.0)
    model.observe_y(y)
```

### 4.1 Point predictions and moments

| Method | Returns |
| --- | --- |
| `motor_output()` | expected state feedback: the per-context feedback means weighted by the **predicted** (pre-observation) context probabilities, averaged over particles |
| `predictive_motor_output(q=None)` | one-step-ahead expected observation given the upcoming raw cue `q` (default: the staged cue) |
| `state_moments()` | `(mu, v)`: predictive latent-state mean and (co)variance, marginalised over contexts and particles |
| `predictive_feedback_moments(q=None)` | `(mu, sigma)`: one-step predictive observation mean and covariance. `q` is a **0-based cue LABEL**, not a raw value; `None` marginalises the cue out |
| `explicit_component()` | explicit component of adaptation: the predictive state mean of the highest-responsibility context (the `c*1` state) |
| `implicit_component()` | implicit component: motor output minus the average predicted state |
| `state_cstar1()` | expected state of the highest-responsibility context |
| `state_cstar2(q=None)` | expected *current* state of the context with the highest **next**-trial predicted probability, given raw cue `q` |
| `state_cstar3()` | expected state of the context with the highest current predicted probability |
| `predicted_probability_cstar1()` | predicted probability of the `c*1` context |
| `predicted_probability_cstar3()` | the highest predicted context probability this trial |
| `kalman_gain_cstar1()` | Kalman gain of the `c*1` context, particle-averaged. **Scalar model only** |
| `kalman_gain_cstar2(q=None)` | Kalman gain of the `c*2` context. **Scalar model only** |

```python
model.motor_output()                    # expected feedback this trial
model.observe_q(1.0)
model.predictive_motor_output()          # forecast for the staged cue
model.predictive_motor_output(2.0)       # ... or for an explicit raw cue

mu, v = model.state_moments()            # predictive latent-state moments
mu_y, s_y = model.predictive_feedback_moments(0)   # 0-BASED cue LABEL

model.explicit_component()               # state of the c*1 context
model.implicit_component()               # motor output minus mean state
model.state_cstar1(), model.state_cstar2(), model.state_cstar3()
model.predicted_probability_cstar1(), model.predicted_probability_cstar3()
model.kalman_gain_cstar1(), model.kalman_gain_cstar2()   # scalar model only
```

Note the deliberate asymmetry, preserved from MATLAB:
`predictive_feedback_moments` takes a **cue label** (it indexes the cue matrix
directly), whereas every other `q`-taking query takes a **raw cue value** and
resolves it through the registry.

`motor_output` and `state_feedback_probability` weight by the *predicted*
(pre-feedback) context probabilities, while `state_probability` weights by the
*responsibilities* (post-feedback). This is intentional and matches COIN. It
does mean that, called after `observe_y`, `motor_output` reflects the
just-processed trial's pre-update prediction; for a genuine next-step forecast
use `predictive_motor_output(q)` or `predictive_feedback_moments(q)`.

### 4.2 Densities and CDFs

Grids are `(K,)` for the scalar model and `(K, N)` -- **one query point per
row** -- for the multi-dimensional model. Every density query returns `(K,)`
values, or a `dict` keyed by global context label for the `*_given_context_*`
forms.

| Method | Returns |
| --- | --- |
| `state_probability(values)` | `(K,)` posterior latent-state density (weights: responsibilities) |
| `state_feedback_probability(values)` | `(K,)` predictive feedback density (weights: predicted probabilities, variances inflated by `R`) |
| `novel_state_probability(values)` | `(K,)` state density of the not-yet-instantiated novel context; all zeros once every particle has saturated its budget |
| `novel_state_feedback_probability(values)` | `(K,)` feedback counterpart of the above |
| `state_given_context_probability(values)` | `{global label: (K,) density}` -- triggers the alignment |
| `state_feedback_given_context_probability(values)` | `{global label: (K,) density}` -- triggers the alignment |
| `retention_given_context_probability(values)` | `{global label: (K,) density}` over retention. **Scalar model only** |
| `drift_given_context_probability(values)` | `{global label: (K,) density}` over drift. **Scalar model only** |
| `bias_given_context_probability(values)` | `{global label: (K,) density}` over the measurement bias. **Scalar model only**, requires `infer_bias=True` |
| `bias_probability(values)` | `(K,)` marginal (across-context) bias density. **Scalar model only**, requires `infer_bias=True` |
| `predictive_state_feedback_cdf(y, q=None)` | predictive CDF at `y` -- a scalar, or the `(N,)` vector of per-dimension marginal CDFs (the standard PIT) |
| `predictive_cue_p_value(q, u=None)` | randomised discrete PIT `F(q-) + u f(q)` for a raw cue value; uniform on `[0, 1]` under a correct model |

```python
grid = np.linspace(-0.5, 1.5, 201)

model.state_probability(grid)                  # (201,) posterior state density
model.state_feedback_probability(grid)         # (201,) predictive feedback
model.novel_state_probability(grid)            # (201,) novel-context state
model.novel_state_feedback_probability(grid)   # (201,) novel-context feedback

model.retention_given_context_probability(np.linspace(0.8, 1.0, 51))
model.drift_given_context_probability(np.linspace(-0.1, 0.1, 51))

model.predictive_state_feedback_cdf(0.9)       # PIT value for y = 0.9
model.predictive_cue_p_value(1.0, u=0.5)       # randomised discrete PIT
```

`predictive_cue_p_value` draws `u` from `model.rng` when it is not supplied,
which makes that call non-reproducible in isolation -- pass `u` explicitly in
tests. A cue value never observed is scored against the trailing novel-cue
column (which carries the novel-cue stick's mass, not zero) and is *not*
registered.

Bias densities need `infer_bias=True`, else they raise `BiasNotInferredError`:

```python
biased = RealTimeCOIN(num_particles=100, infer_bias=True, rng=5)
for y in [0.1, 0.3, 0.4, 0.6]:
    biased.observe_y(y)

vals = np.linspace(-0.5, 0.5, 101)
biased.bias_probability(vals)                    # (101,) marginal bias density
biased.bias_given_context_probability(vals)      # {label: (101,) density}
```

### 4.3 Context summaries -- global (aligned)

These are reported in the **aligned global frame** (section 6), so the first one
called in a trial pays for the alignment and the rest reuse the cache. Vectors
have length `max_contexts + 1`; maps are keyed by **0-based** global context
label and contain only strictly positive entries.

| Method | Returns |
| --- | --- |
| `predicted_context_probabilities_vector()` | `(max_contexts + 1,)` predicted (pre-observation) probabilities; the trailing active entry is the novel context |
| `predicted_context_probabilities_map()` | `{global label: probability}`, same weights |
| `responsibilities_vector()` | `(max_contexts + 1,)` posterior responsibilities |
| `responsibilities_map()` | `{global label: responsibility}` |
| `sampled_context_count()` | `(max_contexts + 1,)` sampled-context occupancy, normalised to sum to one |
| `stationary_context_probabilities()` | `(K,)` stationary distribution of the aligned transition matrix (novel column dropped, rows renormalised) |
| `global_transition_probabilities()` | `(max_contexts + 1,)` expected franchise transition weights; the last active entry is the novel-context stick |
| `global_cue_probabilities()` | `(Q,)` expected franchise cue weights, trailing entry the novel-cue stick. Raises `NoCuesError` if no cue has been observed |
| `local_transition_probabilities()` | `(K, K + 1)`; row `i` is the transition distribution out of global context `i`, column `K` the novel context |
| `local_cue_probabilities()` | `(K, Q)`; row `i` is global context `i`'s cue-emission distribution. Raises `NoCuesError` if no cue has been observed |

### 4.4 Context summaries -- local (fast, unaligned)

Same three quantities computed in the modal particles' **local** label frame,
deliberately skipping the global relabelling. They are cheap enough for live
plots and logging, but their entries are *not* comparable across trials, because
the local labels are arbitrary.

| Method | Returns |
| --- | --- |
| `predicted_context_probabilities_local()` | `(max_contexts + 1,)` weights |
| `context_responsibilities_local()` | `(max_contexts + 1,)` weights |
| `sampled_context_count_local()` | `(max_contexts + 1,)` frequencies |

```python
model.predicted_context_probabilities_vector()   # (max_contexts + 1,)
model.responsibilities_vector()
model.sampled_context_count()
model.stationary_context_probabilities()

model.predicted_context_probabilities_map()      # {0-based label: probability}
model.responsibilities_map()

model.global_transition_probabilities()
model.global_cue_probabilities()
model.local_transition_probabilities()           # (K, K + 1)
model.local_cue_probabilities()                  # (K, Q)

model.state_given_context_probability(grid)              # {label: (201,)}
model.state_feedback_given_context_probability(grid)     # {label: (201,)}

model.predicted_context_probabilities_local()    # fast, unaligned
model.context_responsibilities_local()
model.sampled_context_count_local()
```

### 4.5 Alignment and diagnostics

| Method | Returns |
| --- | --- |
| `context_alignment()` | the alignment `dict` (see section 6) |
| `diagnostics()` | a full globally-aligned snapshot of the current particle state |

```python
a = model.context_alignment()
print(a["K"])                        # number of aligned global contexts
print(a["assignment"].shape)         # (num_particles, max_contexts + 1)
print(a["converged"], a["iterations"])

d = model.diagnostics()
print(sorted(d)[:5])
```

`diagnostics()` returns, for the scalar model: `trial`, `C`, `context`,
`predicted_probabilities`, `responsibilities`, `state_mean`, `state_var`,
`state_feedback_mean`, `state_feedback_var`, `retention`, `drift`, `bias`,
`global_transition_probabilities`, `local_transition_matrix`,
`global_cue_probabilities`, `local_cue_matrix`, `alignment` and `raw` (a
reference to the live `ParticleState`). The multi-dimensional model returns its
own field set: `trial`, `K`, `A` `(K, N, N)`, `drift` `(K, N)`, `bias` `(K, N)`,
`state_mean` `(K, N)`, `state_cov` `(K, N, N)`, `predicted_probabilities` and
`responsibilities` `(K,)`, their `*_particles` `(P_modal, C)` counterparts,
`context`, `transition_prob` `(K, K + 1)`, `cue_prob` `(K, Q)`, `alignment` and
`raw`.

### 4.6 Errors

Every error carries its MATLAB identifier as the message prefix, so
`str(err)` reads `"RealTimeCOIN:NoCues: ..."`.

| Exception | Identifier | Raised when |
| --- | --- | --- |
| `RealTimeCOINError` | `RealTimeCOIN:Error` | base class |
| `NoCuesError` | `RealTimeCOIN:NoCues` | a cue-dependent quantity is requested but no cue has been observed |
| `BiasNotInferredError` | `RealTimeCOIN:BiasNotInferred` | a bias read-out is requested with `infer_bias=False` |
| `ScalarModelOnlyError` | `RealTimeCOIN:ScalarModelOnly` | a scalar-only query is called with `state_dim > 1` (also a `ValueError`) |
| `NameValuePairsError` | `RealTimeCOIN:NameValuePairs` | an unrecognised constructor keyword (also a `TypeError`) |
| `ModelFormatError` | `RealTimeCOIN:ModelFormat` | a persisted file is not a recognisable model archive (also a `ValueError`) |

```python
from realtimecoin import NoCuesError, ScalarModelOnlyError

fresh = RealTimeCOIN(rng=0)
fresh.observe_y(0.1)
try:
    fresh.global_cue_probabilities()
except NoCuesError as err:
    print(err)          # RealTimeCOIN:NoCues: ...

md = RealTimeCOIN(state_dim=2, num_particles=20, rng=0)
md.observe_y([0.1, 0.2])
try:
    md.kalman_gain_cstar1()
except ScalarModelOnlyError as err:
    print(err)          # RealTimeCOIN:ScalarModelOnly: ...
```

### 4.7 Static helpers

The six MATLAB static methods are module-level functions in
`realtimecoin.statics`. The three that draw randomness take a
`numpy.random.Generator` as their first argument, because there is no global
stream to fall back on.

```python
from realtimecoin.statics import (
    log_sum_exp,
    normal_cdf,
    normal_pdf,
    sample_num_tables,
    stationary_distribution,
    systematic_resampling,
)

gen = np.random.default_rng(0)
systematic_resampling(gen, np.array([0.2, 0.5, 0.3]))
normal_pdf(np.array([0.0]), 0.0, 1.0)
normal_cdf(np.array([0.0]), 0.0, 1.0)
log_sum_exp(np.array([[-1.0, -2.0]]), axis=-1)   # default axis is -1
stationary_distribution(np.array([[0.9, 0.1], [0.2, 0.8]]))
sample_num_tables(gen, np.array([1.0]), np.array([5]))
```

---

## 5. Scalar vs multi-dimensional

`state_dim` picks the pipeline, and the choice is made once, in `observe_y`:

- `state_dim == 1` runs the scalar pipeline, which is a verbatim translation of
  the MATLAB scalar path. It is the regression baseline.
- `state_dim > 1` runs the `*_md` pipeline (`predict_states_md`,
  `sample_parameters_md`, ...). `predict_context` is dimension-agnostic and
  shared by both.

The step order is identical in both branches, and the multi-dimensional path
reduces to the scalar one at `N == 1` by construction (the isotropic default
`Q` and `R` collapse to the scalar variances).

What changes for the caller:

| | scalar (`state_dim == 1`) | multi-dimensional (`state_dim == N`) |
| --- | --- | --- |
| `observe_y(y)` | scalar, `None` or `nan` | length-`N` sequence; per-element `nan` marks unobserved dimensions |
| density grids | `(K,)` | `(K, N)`, one query point per row |
| `motor_output()`, `state_cstar*()`, `explicit_component()`, `implicit_component()` | `float` | `(N,)` |
| `state_moments()` | `(float, float)` | `((N,), (N, N))` |
| `predictive_feedback_moments()` | `(float, float)` | `((N,), (N, N))` |
| dynamics representation | scalar `a`, `d` | augmented `Theta = [A | d]`, `(N, N + 1)` |

**Scalar-model-only queries.** These raise `ScalarModelOnlyError` when
`state_dim > 1`:

- `retention_given_context_probability`
- `drift_given_context_probability`
- `bias_given_context_probability`
- `bias_probability`
- `kalman_gain_cstar1`
- `kalman_gain_cstar2`

(The retention/drift/bias densities are scalar-only because the multi-dimensional
model's per-context dynamics are a matrix-normal object rather than a pair of
1-D marginals; inspect `diagnostics()["A"]` / `["drift"]` / `["bias"]` instead.)

Setting `process_noise_covariance` or `observation_noise_covariance` with
`state_dim == 1` is ignored, with a `UserWarning` carrying the identifier
`RealTimeCOIN:CovarianceIgnored`. That is deliberately a warning, not an error.

---

## 6. Context alignment

**The problem.** Context labels are per-particle and arbitrary. Particle 3's
"context 2" has nothing to do with particle 7's "context 2" -- they are just the
order in which each particle happened to instantiate contexts. Averaging a
per-context quantity across particles slot-by-slot would therefore be
meaningless.

**The fix.** Before reporting any per-context summary, the package solves an
assignment problem that maps each particle's local labels onto one globally
consistent labelling for the current trial. Concretely:

1. Pick the **modal cardinality** `K` -- the most common context count across
   particles -- and keep the particles that have exactly `K` contexts.
2. Seed an assignment (warm-starting from the previous trial's solution when it
   is still compatible).
3. Alternate min-cost matching (Hungarian assignment against per-context
   prototypes) and prototype recomputation until the labels stop changing.

The cost between a particle's local context and a global prototype is a sum of
Jeffreys divergences over the state belief, the dynamics and the cue
distribution (and optionally the transition row). The multi-dimensional variant
uses a Gaussian-Jeffreys state cost but a plain Euclidean cost on the vectorised
`Theta` -- equivalent to an isotropic reference covariance -- so MD alignment
ignores dynamics uncertainty that the scalar path accounts for.

**Alignment is reporting only.** It never feeds back into inference and never
mutates the particle arrays. The only model state it writes is
`model.alignment_seed`, the warm start for the next computation.

**Keys are 0-based.** Aligned global context labels run `0 .. K-1`, and the maps
returned by `responsibilities_map()`, `predicted_context_probabilities_map()`,
`state_given_context_probability()` and friends are keyed by those 0-based
integers. This is a deliberate deviation from MATLAB, whose `containers.Map`
keys are 1-based.

**Caching.** `model.state_version` is a monotone counter bumped by every
mutation. The alignment is computed lazily on the first context-facing query
after a state change and cached against that version, so all the aligned queries
within one trial share a single solve *and get back the very same object*. Each
`observe_y` invalidates the cache. `model.alignment_seed` is deliberately *not*
cleared on invalidation, so the next solve can warm-start.

```python
a = model.context_alignment()
```

The alignment `dict` carries: `K`, `assignment` (`(P, C)` local-to-global
labels, `-1` where a slot has no global label), `modal_particle_mask`,
`modal_particle_indices`, `modal_particle_weights`, `global_contexts` (the
per-context prototype parameters), `converged`, `iterations`, `used_seed`,
`cache_state_version` and `computed_at_trial`.

**When to use the `*_local` variants.** If you only need a live readout within
one trial -- a progress plot, a log line -- the `_local` queries skip the solve
entirely. Just do not compare their entries across trials.

---

## 7. Persistence

### 7.1 Snapshots (in memory)

`snapshot()` captures the full model state as a plain, deep-copied mapping:
every public property under `"properties"`, plus the particle state, the staged
cue, the trial counter, the cue registry, the alignment bookkeeping
(`state_version`, `alignment_seed`) and the RNG state. It has value semantics
and is safe to hand across a process boundary.

```python
s = model.snapshot()                       # plain dict, deep-copied
clone = RealTimeCOIN(num_particles=100)
clone.load_snapshot(s)
assert clone.motor_output() == model.motor_output()
```

The same snapshot can be loaded into several models: `load_snapshot` deep-copies
on the way in, so `s` stays reusable.

### 7.2 Files: the `rtcoin-model-v1` format

`save_model` / `load_model` write and read a `numpy` `.npz` archive (via
`numpy.savez_compressed`, read with `allow_pickle=False`, so loading a file can
never execute code). The archive holds:

- `__meta__` -- a `uint8` array of UTF-8 JSON with `format`
  (`"rtcoin-model-v1"`), the 20 constructor `properties`, `trial`, `pending_q`,
  `cue_values`, `state_version`, `alignment_seed`, `rng_state`, and the three
  lists `d_array_fields` / `d_scalar_fields` / `d_none_fields`;
- `D.<field>` -- one entry per non-`None` array field of `ParticleState`, with
  dtype and shape preserved by `npz` itself.

The three `d_*_fields` lists must partition the `ParticleState` dataclass, and
`load_model` checks that they do -- a truncated or hand-edited archive is
rejected (`ModelFormatError`) rather than silently loaded with `None` where an
array belongs. The property names in a file are likewise validated against the
known property list, so a hand-edited archive cannot overwrite an arbitrary
attribute.

```python
model.save_model("checkpoint.npz")                    # stationarised by default
model.save_model("live.npz", set_stationary=False)    # exact current state

resumed = RealTimeCOIN()
resumed.load_model("live.npz")
assert resumed.Trial == model.Trial
```

`load_model` restores the properties too, so you can load into a default-built
`RealTimeCOIN()` -- the constructor arguments do not have to match the file. The
write goes to a sibling temporary file that is renamed into place, so a failure
part-way through cannot destroy an existing checkpoint. The filename is used
verbatim; no `.npz` suffix is appended.

### 7.3 `set_stationary` and the stationary save

`set_stationary()` re-initialises every particle's context and state beliefs to
the stationary distribution implied by the current hyperparameters, rewinds
`Trial` to 0 and clears any pending cue. It keeps the learned parameters, so it
is the way to freeze a fitted model into a deployable, contingency-independent
prior.

```python
model.set_stationary()
print(model.Trial)                # 0
```

`save_model(filename)` does this **by default** (`set_stationary=True`), which is
why the reloaded model above reports `Trial == 0` unless you pass
`set_stationary=False`. Unlike MATLAB, the live object is never mutated in
either mode: the stationary reset runs on a private clone, so an interrupted
save cannot leave your model rewound, and `model.rng` is not advanced.

### 7.4 The RNG state travels

Because every draw goes through `model.rng`, the snapshot carries
`model.rng.bit_generator.state`. A restored model therefore resumes the *exact*
stream: streaming the same observations into it reproduces the original run bit
for bit.

```python
a_model = RealTimeCOIN(num_particles=60, rng=2)
for y in [0.1, 0.2, 0.3]:
    a_model.observe_y(y)
a_model.save_model("mid.npz", set_stationary=False)

b_model = RealTimeCOIN()
b_model.load_model("mid.npz")

for y in [0.4, 0.5]:
    a_model.observe_y(y)
    b_model.observe_y(y)

assert a_model.motor_output() == b_model.motor_output()
```

### 7.5 Building a model from a prepared state

`RealTimeCOIN.from_state(properties, state, trial=1, cue_values=(1,), ...)` is
the Python counterpart of the MATLAB test helper `testutil.loadFixtureModel`: it
installs a caller-supplied `ParticleState` (or a partial `dict` of fields)
without going through a file. It exists to drive queries from hand-crafted
particle configurations in tests.

```python
from realtimecoin import ParticleState, RealTimeCOIN

fixture = RealTimeCOIN.from_state(
    {"num_particles": 4, "max_contexts": 2, "state_dim": 1},
    {
        "n_active": np.array([1, 1, 1, 1]),
        "context": np.zeros(4, dtype=int),
        "predicted_probabilities": np.tile([1.0, 0.0, 0.0], (4, 1)),
        "state_feedback_mean": np.zeros((4, 3)),
    },
    trial=1,
)
print(fixture.motor_output())
```

Unspecified `ParticleState` fields stay `None`, which fails loudly if a query
reads them, rather than silently returning a plausible zero.

---

## 8. The ensemble

`RealTimeCOINEnsemble` orchestrates `runs` independent `RealTimeCOIN` members
that all consume the **identical** observation stream, fed one trial at a time,
and returns the equal-weight average across runs of the corresponding
single-model quantity. It is the real-time analogue of the offline COIN's
"runs": Monte-Carlo variance reduction by probability averaging. The wrapper does
not change any per-trial behaviour -- each member is an ordinary
`RealTimeCOIN`. The full behavioural contract is `docs/SPEC_ensemble.md` at the
repository root.

> The snippets in this section are **illustrative** and are not executed as part
> of the documentation checks: the ensemble module is being translated in
> parallel with this guide. Treat the shapes and names below as the agreed API,
> and once `realtimecoin.ensemble` lands, check its docstrings for anything that
> has moved.

### 8.1 Construction

```python
from realtimecoin import RealTimeCOINEnsemble

ens = RealTimeCOINEnsemble(
    runs=8,                # R independent member filters
    seed=0,                # base seed for the whole ensemble
    max_cores=0,           # 0 => serial executor; > 0 => parallel, capped here
    segment_length=1,      # trials per parallel dispatch (scheduling only)
    num_particles=200,     # ... every other keyword goes to each member
    state_dim=1,
)
```

Ensemble parameters are `runs` (positive int, default 1), `seed` (non-negative
int, default 0), `max_cores` (non-negative int, default 0) and `segment_length`
(positive int, default 1). **Every other keyword is forwarded verbatim and
identically to each member `RealTimeCOIN` constructor**, so all `R` members share
one configuration. An invalid member parameter surfaces the same error the
`RealTimeCOIN` constructor would raise.

Per-member randomness comes from `numpy.random.SeedSequence(seed).spawn(runs)`:
member `k`'s entire lifetime -- construction and every `observe_y` -- draws from
a substream that is a function of `(seed, k)` only. Consequences you can rely on:

- **Reproducibility.** Same `(seed, runs, member parameters)` and the same
  observation stream give bit-for-bit identical outputs from every query, at
  every trial.
- **Executor invariance.** `max_cores` and `segment_length` are performance
  choices with no effect on numerical output.
- **Independence.** Distinct members follow different particle-filter
  trajectories; distinct seeds give different ensembles.
- **No global-state leakage.** The ensemble never touches `numpy.random`'s
  global state.

### 8.2 Stepping

```python
for q, y in zip(cue_sequence, feedback_sequence):
    ens.observe_q(q)        # forwarded identically to every member
    ens.observe_y(y)        # every member advances one trial

print(ens.Trial)            # members advance in lockstep
```

`observe_q` and `observe_y` fan the identical `(q, y)` out to all members;
`ens.Trial` equals every member's `Trial`. Missing observations and cue-free
trials are forwarded as-is and remain well defined.

### 8.3 Run-averaged queries (Phase 1)

Shapes match the single-model methods exactly.

| Method | Averaging rule |
| --- | --- |
| `motor_output()` | mean of the per-run motor outputs |
| `state_moments()` | moments of the **pooled** mixture: `mu = mean(mu_k)`, `v = mean(v_k + mu_k mu_k') - mu mu'` -- *not* a naive mean of the covariances |
| `state_probability(values)` | mean of the per-run densities |
| `state_feedback_probability(values)` | mean of the per-run densities |
| `novel_state_probability(values)` | mean of the per-run densities |
| `novel_state_feedback_probability(values)` | mean of the per-run densities |

Averaging is NaN-aware: an output entry is the mean over the runs where it is
finite, and `nan` only if it is non-finite for *every* run. With `runs == 1`
every query equals its single member's query.

### 8.4 Batch replay

```python
traces = ens.simulate(q_seq, y_seq)
traces["motor_output"]      # per-trial run-averaged motor output
traces["state_mean"]        # per-trial run-averaged state mean
traces["state_var"]         # per-trial run-averaged state (co)variance
traces["Trial"]             # 1 .. T
```

`simulate` replays a precomputed length-`T` observation sequence and returns
per-trial averaged traces. It is numerically identical to constructing a fresh
ensemble with the same `(seed, runs, member parameters)` and stepping
`observe_q` / `observe_y` trial by trial, and it obeys the same executor
invariance. It is a one-shot batch on its own fresh member set: it does not
disturb the live stepping state of `ens`, and calling it twice gives identical
traces.

### 8.5 Context-indexed queries (Phase 2, cross-run aligned)

Context-indexed readouts cannot be averaged slot-by-slot, because each member
labels its contexts in its own member-local global frame. Phase 2 adds a
**cross-run alignment**: the member with the most contexts (ties broken by lowest
index) defines the reference frame, and every other member's contexts are matched
to reference labels by a linear assignment minimising total prototype-state-mean
distance. Every member's novel context always maps to the reference novel slot.
The alignment is deterministic, so it inherits reproducibility and executor
invariance.

These six queries are aligned before averaging:

| Method | Averaging rule in the reference frame |
| --- | --- |
| `responsibilities_vector()` | zero-fill unmatched reference slots, then divide by `R` (probability is conserved: the result sums to 1) |
| `predicted_context_probabilities_vector()` | as above |
| `sampled_context_count()` | as above |
| `stationary_context_probabilities()` | zero-filled and renormalised across runs |
| `state_given_context_probability(values)` | NaN-omit mean over the runs that *have* a context matched to that reference label |
| `state_feedback_given_context_probability(values)` | as above |

The two rules differ for a reason: a run that lacks reference context `j` has
*zero* posterior probability on it (so it contributes 0 to a probability
vector), but has *no* density for it (undefined, not zero -- so it is omitted
from a density average).

One caveat on SPEC 10.5.3's member-order invariance. The reference *labels* are
the reference member's own labels, so the readout is order-invariant only **up
to a relabelling**: if two members tie for the most contexts, reordering them
can hand the frame to the other one and permute the output. With
`argmax_r K_r` unique the readout is invariant outright -- and in practice all
members see one shared observation stream, so they order their contexts alike.
The MATLAB original has exactly the same property (both pick the first
maximiser).

---

## 9. The validation suite

Tests assert; validators report. The suite in `python/validation/` produces
metrics and `passed` flags, so you can see *how far* off something is, not just
that it failed.

```
cd python
python -m validation.run_validation
```

`run(profile="compact", seed=1001, strict=False, make_plots=False)` returns a
`dict` with one entry per stage plus `config` and an overall `passed`. Each stage
runs in isolation, so one failing validator records `passed=False` and the suite
continues. `strict=True` raises `RuntimeError` when the suite does not pass. Each
stage derives its seed as `1001 + STAGE_SEED_OFFSETS[stage]`; some stages
(notably `context_recovery`) are seed-sensitive, so reproduce failures with those
exact values.

| Stage | Module | What it checks | Gates |
| --- | --- | --- | --- |
| `single_context_kalman` | `validate_single_context_kalman` | with one context, very precise dynamics priors and `max_contexts=1`, the scalar model must reduce to a textbook Kalman filter. Compares production predictive moments against the analytic recursion, plus a PIT | `mean_rmse < 0.05`, `variance_relative_error < 0.35`, `feedback_ks < 0.15` |
| `multidim_kalman` | `validate_multidim_kalman` | the same reduction for the MD model, with **correlated** `Q` and `R` so the matrix gain and the Cholesky likelihood are genuinely exercised. Calibration via a chi-square-`N` Mahalanobis PIT | `mean_rmse < 0.05`, `variance_relative_error < 0.35`, `feedback_ks < 0.15` |
| `p_values_extended` | `validate_p_values_extended` | PIT self-calibration on a 2-context HMM stream: the feedback PIT and the randomised discrete cue PIT should be Uniform(0, 1); state and parameter ranks are reported as posterior diagnostics | `feedback_ks < 0.08`, `cue_ks < 0.08`, `state_rank_ks < 0.15` |
| `original_coin_monte_carlo` | `validate_original_coin_monte_carlo` | comparison against the offline `COIN.m` oracle, by replaying the frozen MATLAB traces in `tests/fixtures/oracle/` (there is no Python `COIN.m`). With no fixtures present it reports `skipped=True` and does not gate | `mean_rmse < 0.03`, `worst_correlation > 0.95` |
| `particle_convergence` | `validate_particle_convergence` | Monte-Carlo error should shrink roughly like `1/sqrt(P)` as the particle count grows, and runtime should grow. The oracle-RMSE arm is fixture-gated: absent fixtures make it `None` (skipped), not a failure | `best_feedback_ks < 0.12`, `best_rmse < 0.05` (when fixtures exist), runtime ratio, calibration-or-RMSE improves |
| `context_recovery` | `validate_context_recovery` | recovery of known latent contexts from a synthetic 2-context stream, scored *after* finding the best label permutation. Averaged over several seeds, because the accuracy sits close to its gate | `context_accuracy > 0.65`, `posterior_true_context > 0.45`, `mean_recovery_lag <= 20` |
| `stress_cases` | `validate_stress_cases` | interpretable qualitative probes: stable data must not proliferate contexts, an abrupt change must create one, A-B-A data should reuse an old context, `max_contexts` must be honoured, higher sensory noise must increase posterior uncertainty | mean context count `<= 1.5` on stable data; abrupt-change max context count `>= 1.5`; capped max context count `<= 2`; high/low-noise state-variance ratio `> 1.25` |
| `performance` | `benchmark_realtimecoin` | wall-clock timing over a particle-count grid. A probe, not a check -- it has **no** pass flag and does not gate | -- |

Each validator is also runnable on its own, with its own defaults:

```
python -m validation.validate_multidim_kalman --seed 0 --dim 2
```

`validation/validate_p_values.py` is a thin back-compatibility shim that
delegates to `validate_p_values_extended`.

### 9.1 The parity philosophy

A bitwise match against MATLAB is impossible: the two implementations draw from
different RNG streams, so identical inputs produce different particle
trajectories. Parity is therefore established statistically, on three legs:

1. **Analytic oracles.** Where the model degenerates to something with a closed
   form -- the one-context Kalman filter, scalar and multivariate -- the Python
   output is compared against that closed form directly. This is a stronger
   claim than agreement with MATLAB, because it does not depend on MATLAB being
   right.
2. **Self-calibration.** PIT / KS checks ask whether the model's own predictive
   distributions are calibrated against data drawn from a known generative
   process. A miscalibrated filter fails these regardless of what any reference
   implementation does.
3. **Frozen MATLAB behavioural fixtures.** Traces exported once from the MATLAB
   implementation and committed under `python/tests/fixtures/`, replayed to
   check that the Python model lands in the same *distributional* neighbourhood.

### 9.2 The frozen fixtures

`python/tests/fixtures/` holds MATLAB `.mat` files exported once and committed:

- **`equiv/`** -- behavioural equivalence traces from `@RealTimeCOIN`. Each file
  (`scalar_2ctx`, `scalar_3ctx_small`, `scalar_missing`, `md2_basic`,
  `md2_missing`, `md3_basic`) holds the driving inputs `q` and `y` plus, for
  each of 20 MATLAB seeds, the per-trial `motor`, `stateMean`, `resp`,
  `predCtx` and `counts` traces. Because the RNG streams differ, these are
  compared as distributions across seeds, not trace-by-trace.
- **`oracle/`** -- traces from the *offline* `COIN.m` (`coin_trace_seed2001`
  ... `coin_trace_seed2005`): the hyperparameters `hp`, the particle count `P`,
  the trial count `T`, and the `cues`, `pert`, `y` and `mo` (motor output)
  sequences. Plus `ensemble_blindA_coin` / `ensemble_blindB_coin` for the
  ensemble.
- **`alignment_fixtures.py`** -- hand-crafted particle states used to drive the
  alignment and the per-context queries deterministically, without running the
  filter.

`validate_particle_convergence` looks for its oracle archives as `.npz` files in
`python/tests/fixtures/oracle/` and skips its oracle arm when they are absent, so
a missing fixture never becomes a spurious failure.

---

## 10. Differences from the MATLAB implementation

| Area | MATLAB | Python | Why |
| --- | --- | --- | --- |
| **Array layout** | contexts-first, particles-**last** | particles-**leading**, matrix dimensions trailing | every per-particle matrix becomes a contiguous C-order block, which is what numpy's batched linear algebra wants |
| **Shape map** | `X(c, p)` -> `(Cmax, P)` | `X[p, c]` -> `(P, C)` | the transposition is total |
| | `X(:, :, c, p)` -> `(N, N+1, Cmax, P)` | `X[p, c, :, :]` -> `(P, C, N, N+1)` | |
| | `sub2ind([Cmax, P], context, 1:P)` (`i_observed`) | `(P,)` array of 0-based context column indices | linear indices do not survive the transposition; the MATLAB gather `X(i_observed)` becomes `X[np.arange(P), state.i_observed]` |
| **Density grids** | `N`-by-`K` array, one query point per **column** | `(K,)` scalar / `(K, N)`, one query point per **row** | consistency with the particles-leading convention |
| **Labels** | contexts and cues 1-based; `containers.Map` keys 1-based; unassigned sentinel `0` | contexts, cues and map keys **0-based**; unassigned sentinel `-1` | Python indexing; `0` is a valid 0-based label, hence `-1` for the sentinel |
| **Field names** | `D.C`, `D.Q` | `n_active`, `n_cues` | `C` was too easy to confuse with the context-axis length; both are still counts, not indices |
| **Randomness** | global stream (`rand`, `randn`, `rng(seed)`) | an explicit `numpy.random.Generator` on `model.rng`; the samplers take it as their first argument | no hidden global state; seeding the constructor reproduces a whole run |
| **Persistence** | `.mat` via `save`/`load` | `.npz` + JSON metadata, format `rtcoin-model-v1`, read with `allow_pickle=False` | a model file can never execute code on load |
| **RNG in a snapshot** | not captured (MATLAB uses the global stream) | `rng_state` travels with the snapshot | a restored model resumes the *exact* stream, so resume == uninterrupted, bit for bit |
| **Stationary save** | `saveModel(file, true)` mutates the object, writes, restores | the stationary reset runs on a private clone; the live model is never mutated | an interrupted save cannot leave the model rewound, and the live stream is not advanced |
| **Deprecated aliases** | `predicted_context_probabilities`, `responsibilities`, `context_predicted_probabilities`, `context_responsibilities` still exist with deprecation warnings | **not ported** | use the explicit `*_vector` / `*_map` forms |
| **Property validation** | `arguments` blocks re-fire on every assignment | validators run **once**, at construction | plain attributes afterwards; rebuild rather than mutate hyperparameters in place |
| **Errors** | `error('RealTimeCOIN:NoCues', ...)` | typed exceptions carrying the identifier as a class attribute *and* as the message prefix | `except NoCuesError` and `match="RealTimeCOIN:NoCues"` both work |
| **Method layout** | one `.m` file per public method in the class folder | a thin facade in `model.py` delegating to the module that owns the algorithm | the module split still maps one-to-one onto the MATLAB file layout |
| **Parity** | -- | statistical, not bitwise | the two implementations consume different RNG streams, so identical inputs give different particle trajectories; see section 9.1 |
| **Not ported** | `restoreMDBiasStatisticsCompatibility` (the pre-`bias_info_ss` migration shim) | -- | there is no pre-`v1` Python save format to migrate from |

For a method-by-method translation table, see
[`api_mapping.md`](api_mapping.md).

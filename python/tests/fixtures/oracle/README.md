# Frozen MATLAB oracle fixtures

The Python port has no `COIN.m`. Every validator that scores `RealTimeCOIN`
against the original offline COIN therefore replays a **frozen trace** captured
once from a live MATLAB session, rather than regenerating it. This directory
holds those traces, plus the run-averaged ensemble references.

A second fixture family, the cross-language equivalence battery, lives in
`../equiv/` and is documented at the bottom of this file.

All files are MATLAB v5/v7 `.mat` and are read with

```python
scipy.io.loadmat(path, squeeze_me=True)
```

`squeeze_me=True` drops singleton dimensions, so MATLAB's `1 x T` row vectors
arrive as `(T,)` and its scalars as 0-d arrays. SciPy already presents MATLAB's
column-major storage in C order, so a stored `R x T` matrix indexes as
`[run, trial]` with **no transpose**. Every loader in this package reshapes
explicitly (`np.asarray(...).reshape(-1)`) rather than indexing positionally, so
the code is robust to whether a given variable was squeezed.

---

## 1. Single-run COIN traces — `coin_trace_seed<seed>.mat`

Seeds present: **2001, 2002, 2003, 2004, 2005**.

Consumed by `validation/compare_original_coin.py` and, through it,
`validation/validate_original_coin_monte_carlo.py`.

| variable | MATLAB shape | after `squeeze_me` | meaning |
| --- | --- | --- | --- |
| `y`    | `1 x T` | `(T,)` | state feedback COIN generated internally (`S.runs{1}.state_feedback`) — the stream replayed into `RealTimeCOIN` |
| `mo`   | `1 x T` | `(T,)` | COIN's stored motor output (`S.runs{1}.stored.motor_output`) — the reference |
| `pert` | `1 x T` | `(T,)` | the perturbation schedule |
| `cues` | `1 x T` | `(T,)` | cue labels, **1-based** integers |
| `T`    | scalar | 0-d | trial count (60) |
| `P`    | scalar | 0-d | particle count (100) |
| `hp`   | struct | struct array | COIN hyperparameters (below) |

`hp` fields, each forwarded to the `RealTimeCOIN` constructor under the same
name: `gamma_context`, `alpha_context`, `rho_context`, `gamma_cue`, `alpha_cue`,
`prior_mean_retention`, `prior_precision_retention`, `prior_precision_drift`,
`sigma_process_noise`, `sigma_sensory_noise`, `sigma_motor_noise`.

Two constructor keywords are **not** in `hp` because
`validation/compare_original_coin.m` fixes them as literals rather than reading
them off the `COIN` object: `max_contexts = 4` and `prior_mean_drift = 0`. They
live in `_FIXED_KWARGS` in `compare_original_coin.py`.

### How they were produced

The capture matches `validation/compare_original_coin.m` exactly:

```matlab
rng(seed);
T = 60;
third = floor(T/3);
pert = [zeros(1,third), 0.4*ones(1,third), -0.2*ones(1, T - 2*third)];
cues = ones(1,T);  cues(floor(T/2):end) = 2;

old = COIN;
old.perturbations = pert;   old.cues = cues;
old.runs = 1;               old.particles = 100;
old.max_contexts = 4;
old.store = {'motor_output', 'state_feedback'};
old.sigma_motor_noise = 0;  old.plot_state_feedback = false;

S  = old.simulate_COIN;
y  = S.runs{1}.state_feedback;
mo = S.runs{1}.stored.motor_output;
hp = struct('gamma_context', old.gamma_context, ...);   % see the table above
save(sprintf('coin_trace_seed%d.mat', seed), 'y','mo','pert','cues','hp','T','P');
```

### How they are replayed

`compare_original_coin.py` transliterates the `.m`'s loop, and the order is
load-bearing:

```python
for t in range(T):
    rt_motor[t] = model.predictive_motor_output(cues[t])   # BEFORE the update
    model.observe_q(cues[t])
    model.observe_y(y[t])
```

`predictive_motor_output` is read **first**, so it is the prior predictive for
trial `t` — the same quantity COIN stores as `mo[t]`. Reading it after
`observe_y` would compare a posterior against a prior and shift the traces by a
trial.

Cue values are passed through as the raw 1-based MATLAB labels. The package's
cue registry maps distinct raw values to consecutive internal labels, so only
which values *repeat* matters, not their numeric value.

### What `trials` / `particles` mean now

In the `.m` they size the paradigm. Here they are **cross-checks**: passing a
value that disagrees with the fixture raises, because the paradigm cannot be
resized without re-running COIN. `run_validation.py` passes the compact
profile's 60 / 100, which match.

### Deviation from the `.m`'s seeding

MATLAB calls `rng(seed)` before *both* COIN and `RealTimeCOIN`, so the two
particle filters share a random stream and part of their agreement is
common-random-number agreement. Python cannot reproduce MATLAB's stream, so here
the replay model is seeded independently (`rng=seed` into
`numpy.random.default_rng`) and the residual is pure Monte-Carlo error at
`P = 100`. That makes the Python numbers a **harder** test of the same gate, not
an easier one, so the `.m`'s gates (`mean_rmse < 0.03`,
`worst_correlation > 0.95`) were kept verbatim.

### `.npz` conversion for `validate_particle_convergence`

`validation/validate_particle_convergence.py` documents an oracle hook that
reads `compare_original_coin_p{particles}[_s{seed}].npz`.
`compare_original_coin.export_oracle_npz()` writes that form from a `.mat`
trace:

```bash
python -m validation.compare_original_coin --seed 2001 --export-npz
```

**This does not by itself enable that validator's oracle arm, and no `.npz` is
committed here.** The convergence validator sweeps particle counts (25, 50, 100
in the compact profile) and requires a COIN trace generated *at each count* —
`validate_particle_convergence.m` calls `compare_original_coin('Particles',
particles(i))`, so COIN itself is re-run at every count. Every trace in this
directory is `P = 100`, so only the `p100` slot could be filled; the validator's
`have_oracle` test requires all three and correctly keeps reporting its oracle
arm as skipped. Reusing the `P = 100` reference for the `P = 25` and `P = 50`
slots would measure a *different* quantity (RealTimeCOIN at N particles versus
COIN at 100) and is deliberately not done. To close that gap, export
`coin_trace_seed<seed>.mat` at `P = 25` and `P = 50` from MATLAB and convert
them.

---

## 2. Ensemble run-average references — `ensemble_blind{A,B}_coin.mat`

Consumed by `validation/validate_ensemble_vs_coin_blindA.py` and
`..._blindB.py`. Both files share the same layout; only the configuration
differs.

| variable | MATLAB shape | after `squeeze_me` | meaning |
| --- | --- | --- | --- |
| `coinRuns`     | `R x T` | `(R, T)` | per-run stored motor output |
| `coinMotorAvg` | `1 x T` | `(T,)`   | `mean(coinRuns, 1, 'omitnan')`, the equal-weight run average |
| `pert`         | `1 x T` | `(T,)`   | perturbation schedule, also fed to the ensemble as observed feedback `y` |
| `cues`         | `1 x T` | `(T,)`   | cue labels, 1-based |
| `hp`           | struct | struct | member hyperparameters, same fields as above |
| `R`, `T`, `P`, `seed`, `sigmaR` | scalars | 0-d | configuration |

| | blind A | blind B |
| --- | --- | --- |
| runs `R` | 30 | 16 |
| trials `T` | 75 | 90 |
| particles `P` | 100 | 80 |
| seed | 4242 | 4201 |
| `sigmaR` (`sigma_sensory_noise`) | 1e-3 | 1e-3 |
| `sigma_motor_noise` | 0 | 0 |
| `max_contexts` | 4 | 4 |

Captures match `validation/validate_ensemble_vs_coin_blindA.m` and
`..._blindB.m` exactly.

### Why the noise is set this way

COIN *generates* its feedback as `perturbation + sensory_noise + motor_noise`,
whereas the ensemble is *fed* an observed stream. Setting
`sigma_motor_noise = 0` and `sigma_sensory_noise = 1e-3` makes each COIN run's
feedback equal the perturbation to within that tiny noise, so feeding the
ensemble the perturbation puts both sides on the same effectively-deterministic
stream. Exactly zero is not usable: it makes `COIN.m` degenerate (its motor
output becomes `NaN` after trial 2), which is why the tiny value is used and
shared with the ensemble members.

### Pending dependency

`RealTimeCOINEnsemble` is delivered by a separate translation unit. Until it
merges, both validators return
`{"errored": False, "skipped": True, "passed": True,
  "skipped_reason": "pending unit D1 merge (...)"}` and their pytest wrappers
skip. The API they are written against:

```python
RealTimeCOINEnsemble(runs=..., seed=..., max_cores=0, **member_kwargs)
ensemble.simulate(q_seq, y_seq)   # -> mapping/object with a run-averaged
                                  #    `motor_output` trace of shape (T,)
```

`member_kwargs` is built by `validate_ensemble_vs_coin_blindA.member_kwargs()`
from `hp` plus `num_particles=P`, `max_contexts=4`, `prior_mean_drift=0`.

Blind A's alignment is lag 0 (both traces are the trial-`t` prior predictive).
Blind B evaluates lag 0 *and* lag 1 (ensemble leading COIN), reports both and
gates on the better fit, because its `.m` treats the trial-phase correspondence
as underdetermined by the spec.

---

## 3. Equivalence-battery fixtures — `../equiv/<scenario>.mat`

Six files, one per scenario of `tests/+equiv/scenarioBattery.m`:
`scalar_2ctx`, `scalar_missing`, `scalar_3ctx_small`, `md2_basic`,
`md2_missing`, `md3_basic`. Consumed by `tests/equiv/` and
`tests/test_equiv_parity.py`.

| variable | MATLAB shape | after `squeeze_me` | meaning |
| --- | --- | --- | --- |
| `name`      | char | 0-d `<U` | scenario name |
| `seeds`     | `1 x R` | `(R,)` | the R = 20 MATLAB model seeds used |
| `q`         | `1 x T` | `(T,)` | cue labels, 1-based |
| `y`         | `dim x T` | `(T,)` if `dim == 1` else `(dim, T)` | feedback; an all-NaN column is a missing trial |
| `motor`     | `dim x T x R` | `(T, R)` if `dim == 1` else `(dim, T, R)` | `motor_output()` after each trial |
| `stateMean` | `dim x T x R` | same | first output of `state_moments()` |
| `predCtx`   | `L x T x R` | `(L, T, R)` | `predicted_context_probabilities()`, globally aligned |
| `resp`      | `L x T x R` | `(L, T, R)` | `responsibilities()`, globally aligned |
| `counts`    | `L x T x R` | `(L, T, R)` | `sampled_context_count()`, globally aligned |

`L = max_contexts + 2` is a fixed capacity, one entry wider than the
`max_contexts + 1` the queries actually return, so every trial carries exactly
one all-NaN pad row. **A NaN pad means the MATLAB query returned no entry for
that context slot and is treated as ZERO mass** when averaging.

The captured surface is `tests/+equiv/captureRun.m`'s, recorded after
`observe_q(q[t])` then `observe_y(y[:, t])` on every trial.

Five fields, but **not five independent checks**: `motor` and `stateMean`
coincide numerically in this model (predictive feedback mean == predictive state
mean, with no bias term and an identity observation map — they agree to
`<= 7e-16` in every fixture). Both are kept because the MATLAB capture keeps
both and they exercise distinct code paths.

### What the Python side re-derives and checks

`tests/equiv/scenario_battery.py` cannot regenerate `y` (it comes from a MATLAB
`RandStream`), so it loads `q` and `y` from the fixture — which is the point:
both languages then provably consume identical inputs. The *deterministic* parts
of `makeInputs` are re-derived and asserted against the fixture:

* the cue schedule — blocks of `max(1, round(T / (2 * cues)))` trials cycling
  through `1..cues` — is regenerated and compared element-wise;
* the missing-trial cadence — MATLAB indices `step:step:T` with
  `step = max(1, round(1 / missing_frac))` — is compared against the fixture's
  NaN column pattern;
* shapes (`dim`, `T`, `L >= max_contexts + 1`, run count vs `seeds`) and the
  stored `name` are asserted.

`round` is MATLAB's round-half-away-from-zero, not Python's banker's rounding;
`_matlab_round` implements the MATLAB rule.

So a fixture regenerated from a drifted `scenarioBattery.m` fails loudly instead
of being silently replayed.

### How the two languages are compared

Not bit-for-bit — that is impossible. MATLAB draws from a Mersenne Twister and
this package from NumPy's PCG64, and one differing variate on trial 1 re-labels
contexts and changes every later draw. Instead both sides average `R = 20`
independent seeds and the run-averages are compared inside a Monte-Carlo band:

```
|py_mean[i,t] - matlab_mean[i,t]| <= 3 * SE[i,t] + floor

SE[i,t] = sqrt( matlab_std[i,t]^2 / R_matlab + py_std[i,t]^2 / R_py )
```

`SE` is the exact standard error of the *difference* of two independent sample
means (Welch, no equal-variance assumption). Both terms matter: the Python
across-seed spread on the context quantities runs ~35 % *larger* than the
fixture's, so a band built from the MATLAB spread alone is more than `sqrt(2)`
too narrow, and being a 20-sample estimate it collapses onto the floor wherever
the 20 MATLAB seeds happened to agree.

Floors: `2.5e-3` (motor, stateMean) and `4e-2` (all three probability vectors,
including `counts` — it is a normalised occupancy *fraction*, not an integer
count, so it gets the same floor as the others). They exist only to cover the
heavy tail of these near-one-hot quantities and to keep the band off zero at the
sorted-tail entries that are identically zero in every seed. Each is set from
the measured `max(deviation - 3*SE)` across the battery (`1.0e-3` for motor,
`1.03e-2`/`4.1e-3`/`1.62e-2` for pred_ctx/resp/counts) with a 2.5–10x margin.

The band math was validated with a **null control**: 20 Python seeds against 20
*different* Python seeds — same code, so the only difference is Monte-Carlo
noise. Under the two-sample `SE` the null needs essentially no floor
(max requirement `5.2e-3`, on `counts`); under a one-sided
`3*matlab_std/sqrt(20)` band it would have required floors of 0.002–0.038, i.e.
that band failed a comparison of the implementation with itself.

The null control also settles whether the cross-language context-probability
deviations (up to ~0.077) are real: in **four of the six scenarios the
same-code null deviates as much as or more than Python deviates from MATLAB**
(e.g. `md2_basic` `resp`: null 0.108 vs cross-language 0.060). Averaging 20
seeds simply does not pin these near-one-hot quantities down more tightly.

The Python seeds are `SeedSequence`-derived and deliberately do *not* match the
MATLAB seeds — the streams cannot be aligned, so matching integers would only
mislead.

The three context-indexed quantities are compared as **sorted-descending**
vectors (NaN pads zero-filled first), because context labels are per-run and
arbitrary: nothing ties MATLAB run 3's "context 2" to Python run 7's. That tests
the *multiset* of context masses — how mass is spread over however many contexts
are active — and is blind to a relabelling, by design. It still catches a wrong
number of occupied contexts, mass concentrated too narrowly or too broadly, and
a systematically different novel-context probability.

See `tests/equiv/compare_runs.py` for the full derivation of the band and the
per-quantity floor rationale.

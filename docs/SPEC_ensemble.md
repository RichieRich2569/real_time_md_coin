# SPEC — `RealTimeCOINEnsemble` (multi-run averaging wrapper)

**Status:** contract for implementation and test authoring. This document is the
*only* shared reference between the implementers and the (blind) test authors. It
specifies **observable behaviour**, not internal algorithms. Where a requirement is
needed to make behaviour well-defined (e.g. RNG substreams), it is stated as a
**normative guarantee** you may test, not as an implementation recipe.

Everything here is derivable from the offline `COIN.m` "runs" semantics
(equal-weight averaging of R independent stochastic realizations) applied to the
real-time `RealTimeCOIN` particle filter.

---

## 1. Purpose

`RealTimeCOINEnsemble` orchestrates **R independent `RealTimeCOIN` filters**
("members" / "runs") that all consume the **identical** observation stream, fed
one trial at a time. Its query methods return the **equal-weight average across
runs** of the corresponding single-model quantity — i.e. the readout of the pooled
mixture distribution formed by giving every run weight `1/R`. This reduces
Monte-Carlo variance ("probability averaging") and underpins validation.

The wrapper does **not** modify any `RealTimeCOIN` per-trial behaviour; each member
is an ordinary `RealTimeCOIN` object.

---

## 2. Class and construction

`RealTimeCOINEnsemble` is a MATLAB **handle** class (a class folder
`@RealTimeCOINEnsemble/`).

### 2.1 Constructor

```matlab
ens = RealTimeCOINEnsemble(Name, Value, ...)
```

Accepts name/value pairs. Two groups:

**Ensemble parameters**

| Name             | Type / default        | Meaning |
|------------------|-----------------------|---------|
| `runs`           | positive integer, `1` | number R of independent member filters |
| `seed`           | nonnegative integer, `0` | base RNG seed for the whole ensemble |
| `max_cores`      | nonnegative integer, `0` | worker cap: `0` ⇒ serial executor; `>0` ⇒ parallel executor capped at this many workers (mirrors `COIN.max_cores`) |
| `segment_length` | positive integer, `1` | live-path parallel batch size (trials replayed per `parfor` dispatch); no effect on results, only on scheduling |

**Member parameters** — *every other* name/value pair is forwarded **verbatim and
identically** to each member `RealTimeCOIN` constructor (e.g. `state_dim`,
`num_particles`, `max_contexts`, `gamma_context`, `sigma_process_noise`, …). All R
members receive the **same** configuration.

Invalid member parameters must surface the same error the `RealTimeCOIN`
constructor would raise.

### 2.2 Read-only properties / accessors

- `ens.runs`, `ens.seed`, `ens.max_cores`, `ens.segment_length` — as constructed.
- `ens.Trial` — the common trial counter. All members advance in lockstep, so
  `ens.Trial == member_k.Trial` for every k.
- `ens.weights` — the 1×R run weights; **uniform** `ones(1,R)/R` in this version.

---

## 3. RNG determinism contract (normative)

This is the core reproducibility guarantee and **must** hold exactly.

1. **Per-run substreams.** All randomness consumed by member k over its entire
   lifetime — construction/reset **and** every `observe_y` — derives from a
   dedicated random substream that is a function of `(seed, k)` **only**. Use a
   sub-streamable generator (e.g. `Threefry` or `Philox`) with `NumStreams = R`
   and `StreamIndices = k`. Consequently member k's realization is independent of
   `max_cores`, `segment_length`, worker placement, and of the other members.

2. **Reproducibility.** Two ensembles constructed with the same `seed` and `runs`
   and fed an identical observation stream produce **bit-for-bit identical** outputs
   from every query method, at every trial.

3. **Executor invariance.** For fixed `(seed, runs)` and an identical observation
   stream, results are **bit-for-bit identical** whether `max_cores == 0` (serial)
   or `max_cores > 0` (parallel), and for any `segment_length`. The executor is a
   performance choice with **no** effect on numerical output.

4. **Independence.** Distinct members (different k) use distinct substreams, so with
   probability 1 they follow different particle-filter trajectories even given the
   same observations. Distinct `seed` values give (a.s.) different ensemble results.

5. **No global-stream side effects.** After any ensemble call returns, the caller's
   global RNG stream is left in the state it was in before the call (the ensemble
   must not leak its members' streams onto the global default stream).

---

## 4. State-machine API

Mirrors `RealTimeCOIN`. Call order per trial: any number of `observe_q`, then
exactly one `observe_y`.

### 4.1 `observe_q(ens, q)`
Stage the cue `q` for the upcoming trial by forwarding the **identical** `q` to
every member (each member's `observe_q(q)`). `q` is a scalar cue id, or `[]`/`NaN`
for a cue-free trial. Draws no randomness. Does not advance the trial.

### 4.2 `observe_y(ens, y)`
Feed the **identical** feedback `y` to every member (each member's `observe_y(y)`,
under that member's substream), advancing every member by one trial. `y` is:
- scalar model (`state_dim == 1`): a scalar, or `[]`/`NaN` for a missing observation;
- multi-dimensional (`state_dim > 1`): an `N×1` column, with `NaN` entries marking
  partially-observed dimensions; `[]` or all-`NaN` marks a fully missing observation.

`y` must be forwarded byte-identically to all members. After the call, `ens.Trial`
has incremented by 1.

---

## 5. Averaged query methods (Phase 1)

Each returns the **equal-weight average across the R members** of the single-model
query of the same name. Shapes match the single-model method exactly (scalar model
vs multi-dimensional model). Averaging is **NaN-aware** (§5.4).

### 5.1 `u = motor_output(ens)`
`u = (1/R) Σ_k motor_output(member_k)`.
Scalar when `state_dim == 1`; `N×1` when `state_dim > 1`.

### 5.2 `[mu, v] = state_moments(ens)`
Moments of the **pooled** (equal-weight over runs) predictive-state mixture — i.e.
the law-of-total-(co)variance combination, **not** a naive mean of the per-run
covariances. With per-run moments `(mu_k, v_k)` from `state_moments(member_k)`:

```
mu = (1/R) Σ_k mu_k
v  = (1/R) Σ_k ( v_k + mu_k*mu_k' )  −  mu*mu'
```

Scalar model: `mu`, `v` scalars (`v = (1/R)Σ(v_k+mu_k^2) − mu^2`, floored at 0).
MD: `mu` is `N×1`, `v` is the symmetric `N×N` covariance (symmetrise as
`(v+v')/2`).

### 5.3 Density queries on a grid
For `values` shaped as the single-model methods require (scalar model: a length-K
vector ⇒ `1×K` row; MD: `N×K` columns ⇒ `1×K` row), each returns the average of
the per-run densities:

- `d = state_probability(ens, values)` = `(1/R) Σ_k state_probability(member_k, values)`
- `d = state_feedback_probability(ens, values)` = `(1/R) Σ_k state_feedback_probability(member_k, values)`
- `d = novel_state_probability(ens, values)` = `(1/R) Σ_k novel_state_probability(member_k, values)`
- `d = novel_state_feedback_probability(ens, values)` = `(1/R) Σ_k novel_state_feedback_probability(member_k, values)`

(Averaging per-run densities is the density of the pooled `1/R`-weighted mixture,
matching `COIN.m`'s weighted sum of per-run densities.)

### 5.4 NaN-aware averaging rule
For every averaged output entry (element of `u`, `mu`, `v`, or `d`): if the
corresponding per-run value is finite for at least one run, the result is the mean
over the runs where it is finite; if it is non-finite (`NaN`/`Inf`) for **every**
run, the result is `NaN`. (This mirrors `COIN.weighted_sum_along_dimension`'s
`omitnan`-with-all-NaN⇒NaN behaviour.)

### 5.5 `runs == 1` reduction
With `runs == 1`, every ensemble query must equal the query of its single member
(the ensemble is a transparent, seeded wrapper).

### 5.6 Context-indexed readouts → Phase 2 (see Part 10)
Context-indexed readouts (responsibilities, predicted-context probabilities,
per-context state/feedback densities, stationary probabilities) require a cross-run
context relabelling (COIN.m `find_optimal_context_labels`). These are specified in
**Part 10 (Phase 2)**. The transition/cue probability *matrices* remain out of scope
(single-model only) for now.

---

## 6. Batch replay: `simulate`

```matlab
traces = simulate(ens, qSeq, ySeq)
```

Replays a full **precomputed** observation sequence of length `T` across the R runs
and returns per-trial averaged traces. This is the offline analog of
`COIN.simulate_COIN` and the primary parallel-throughput path.

**Inputs**
- `qSeq` — `1×T` cue row (use `NaN` for cue-free trials), or `[]` ⇒ all trials
  cue-free.
- `ySeq` — feedback: `N×T` (scalar model ⇒ `1×T`). `NaN` entries mark missing /
  partially-observed feedback, exactly as for `observe_y`.

**Behaviour**
`simulate` must be **numerically identical** to constructing a fresh ensemble with
the same `(seed, runs, member params)` and executing, for `t = 1:T`:
`observe_q(qSeq(t))` (or the cue-free path) then `observe_y(ySeq(:,t))`, recording
the ensemble queries after each trial. It obeys the executor-invariance guarantee
(§3.3): identical results for serial and parallel `max_cores`.

**Output** `traces` is a struct with at least:
- `traces.motor_output` — `N×T`, column t = `motor_output(ens)` after trial t.
- `traces.state_mean` — `N×T`, column t = the `mu` of `state_moments(ens)` after trial t.
- `traces.state_var` — scalar model: `1×T`; MD: `N×N×T`; slice t = the `v` of
  `state_moments(ens)` after trial t.
- `traces.Trial` — `1×T`, equal to `1:T`.

A `simulate` call is a one-shot batch: it operates on its own fresh member set
seeded from `ens` and does not disturb the live stepping state of `ens`. (Calling
`simulate` twice on the same `ens` yields identical `traces`.)

---

## 7. Additive `RealTimeCOIN` public API (implementer track)

To let the ensemble checkpoint/resume members across the parallel boundary without
disk I/O, add two **additive** public methods to `@RealTimeCOIN` (no change to any
existing per-trial module or its behaviour):

- `s = snapshot(obj)` — return a plain struct capturing the full model state
  (equivalent to the existing private `serializableState`).
- `loadSnapshot(obj, s)` — restore state in place from such a struct (equivalent to
  the existing private `restoreSerializableState`).

**Round-trip guarantee:** after `loadSnapshot(obj, snapshot(other))`, `obj`
produces identical outputs from every query method as `other` did, for the same
inputs. A snapshot is serialisable (safe to pass to/from `parfor` workers).

---

## 8. Edge cases (all must be well-defined)

- **Missing observations:** `[]`/`NaN` feedback is forwarded to all members; the
  trial still advances; averaged queries remain defined.
- **Cue-free trials:** `observe_q([])`/`observe_q(NaN)` (or never calling
  `observe_q`) forwards a cue-free trial to all members.
- **Saturated novel context:** if a member's novel-context density is all zeros
  (context budget exhausted), it contributes zeros to the density average (a valid
  finite contribution, not a NaN).
- **`runs == 1`:** transparent single-member behaviour (§5.5).
- **No Parallel Computing Toolbox / `max_cores == 0`:** the serial executor is a
  complete fallback producing the contract results.

---

## 9. Summary of testable guarantees

1. `observe_q`/`observe_y` fan out the identical `(q, y)` to all R members; trials
   advance in lockstep (`ens.Trial`).
2. Each averaged query equals the equal-weight (NaN-aware) average of the per-run
   query, with the moment/density combination rules of §5.
3. Same `(seed, runs)` + same stream ⇒ bit-identical outputs (reproducibility).
4. Serial (`max_cores=0`) and parallel (`max_cores>0`), and any `segment_length`,
   give bit-identical outputs (executor invariance) — for both stepping and
   `simulate`.
5. Members are mutually independent; different seeds diverge.
6. `simulate(qSeq,ySeq)` ≡ the trial-by-trial stepping loop.
7. `runs == 1` ⇒ ensemble ≡ its single member.
8. `snapshot`/`loadSnapshot` round-trip preserves all query outputs.
9. The ensemble leaves the caller's global RNG stream unchanged.

---

## Part 10 — Phase 2: cross-run context alignment

Context-indexed readouts cannot be averaged slot-by-slot: each member labels its
contexts with an arbitrary, member-local global frame (see
`RealTimeCOIN/context_alignment`). Phase 2 adds a cross-run alignment that maps
every member's contexts onto one common **reference frame**, then averages. This is
the real-time analogue of COIN.m's `find_optimal_context_labels` + `integrate_over_runs`.

### 10.1 Single-member frame recap (given, not to be re-derived)
For a single `RealTimeCOIN` member, per SPEC of that class:
- Context probability vectors (`responsibilities_vector`,
  `predicted_context_probabilities_vector`, `sampled_context_count`) are
  `1×(max_contexts+1)` rows that **sum to 1**. If the member has `K` instantiated
  contexts, real mass sits in slots `1..K`, the **novel-context** mass sits in slot
  `K+1`, and slots `K+2..max_contexts+1` are zero.
- `stationary_context_probabilities` is a `1×K` row over the instantiated contexts
  (no novel entry), summing to 1 (or `[]` when `K==0`).
- `state_given_context_probability(values)` /
  `state_feedback_given_context_probability(values)` return a `containers.Map`
  keyed by global context label `1..K`, each value a `1×size(values,2)` density row.
- `context_alignment(member)` exposes `.K` and `.global_contexts` prototypes; for the
  scalar model `.global_contexts.state_mean` is `1×K`, for MD it is `N×K`.

### 10.2 The reference frame and the cross-run matching (normative)
Let `K_r` be member r's context count and `Kref = max_r K_r` (ties → lowest member
index); the member attaining `Kref` is the **reference member**, and its contexts
`1..Kref` define the reference labels. For every member r, compute a matching
`π_r` from member r's contexts `1..K_r` to distinct reference labels `1..Kref` that
**minimises total prototype distance**, where the distance between member-r context
`i` and reference context `j` is the Euclidean distance between their prototype state
means `‖ global_contexts_r.state_mean(:,i) − global_contexts_ref.state_mean(:,j) ‖`
(scalar model: `|·|`). Solve as a linear assignment (Hungarian / `matchpairs`);
since `K_r ≤ Kref` all of member r's contexts are matched and `Kref − K_r`
reference labels are left unmatched for member r. The reference member matches itself
by the identity. The **novel** context of every member always maps to the reference
novel slot (`Kref+1`). The alignment is a deterministic function of the members'
current states (no randomness), so it inherits reproducibility and executor
invariance.

### 10.3 Averaging rules in the reference frame
Two distinct rules, by quantity type:

- **Probability vectors** (`responsibilities_vector`,
  `predicted_context_probabilities_vector`, `sampled_context_count`,
  `stationary_context_probabilities`): a run that lacks reference context `j`
  contributes **0** to slot `j` (its posterior on that specific context is ~0; its
  mass lives in novel / other contexts). So the reference-frame value of slot `j` is
  `(1/R) Σ_r [ member r's value for the context matched to j, or 0 if unmatched ]`,
  with each member's novel value added into the novel slot. Because each member's
  vector sums to 1, the aligned average **sums to 1** exactly (probability is
  conserved). Zero-fill, divide by R — **not** NaN-omit.

- **Per-context densities** (`state_given_context_probability`,
  `state_feedback_given_context_probability`): a run that lacks reference context `j`
  has **no** density for `j` (undefined, not zero). So the reference-frame density of
  context `j` is the **NaN-omit mean** over the runs that *do* have a context matched
  to `j` (average over contributing runs only; if no run has `j`, `j` is absent from
  the output Map).

### 10.4 New ensemble methods
Shapes mirror the single-member methods, but in the reference frame:

- `p = responsibilities_vector(ens)` → `1×(max_contexts+1)`, sums to 1 (real in
  `1..Kref`, novel in slot `Kref+1`, rest 0).
- `p = predicted_context_probabilities_vector(ens)` → as above.
- `n = sampled_context_count(ens)` → as above.
- `p = stationary_context_probabilities(ens)` → `1×Kref` over reference contexts,
  sums to 1 (or `[]` when `Kref==0`); zero-filled + renormalised across runs.
- `m = state_given_context_probability(ens, values)` → `containers.Map` keyed by
  reference label `1..Kref`, each value the NaN-omit mean density (§10.3).
- `m = state_feedback_given_context_probability(ens, values)` → as above.

`values` shape rules and the `state_dim==1` vs `>1` behaviour are exactly the
single-member ones. These are read-only (no randomness), so §3.5 (global stream
untouched) still holds.

### 10.5 Testable guarantees (Phase 2)
These hold **without** re-implementing the aligner and are the recommended oracles:
1. **`runs == 1` reduction:** every Phase-2 query equals its single member's query
   (`Kref = K_1`, identity matching; density Map keys and values identical; vectors
   identical including novel-slot placement). This is **bit-exact for the scalar
   model**; for the MD model it holds **to within a few eps**, because the vector
   average consolidates each member's tiny (~1e-16) residual mass in unused slots
   into the novel slot to keep the sum exactly 1 — use a small tolerance for MD.
2. **Probability conservation:** `responsibilities_vector`,
   `predicted_context_probabilities_vector`, `sampled_context_count`, and
   `stationary_context_probabilities` each sum to 1 (within a few eps) whenever the
   members hold ≥1 context.
3. **Member-permutation invariance:** the result is invariant to the order in which
   members are stored/averaged (equal weights).
4. **Reproducibility & executor invariance:** same `(seed, runs)` ⇒ bit-identical
   Phase-2 outputs; `max_cores` and `segment_length` do not change them.
5. **Trivial-alignment regime:** in a single-context (or well-separated,
   consistently-labelled) scenario where all members share the same frame, the
   aligned average equals the naive slot-by-slot / key-by-key average of the member
   queries.
6. **Density normalisation:** each per-context density integrates to ≈1 over a
   sufficiently wide, fine grid (same property the single-member densities satisfy).
7. **Shape/key correctness:** vector lengths `max_contexts+1`; density Map keys are a
   subset of `1..Kref` with `1×size(values,2)` rows.

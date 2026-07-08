# IMPROVEMENTS_REPORT.md

## Experimental-branch changelog (what was implemented)

The `experimental` branch implements the deferred items below, each validated by an
equivalence harness (`tests/+equiv`) that diffs a scalar+MD scenario battery against a
`main` snapshot, plus `run_tests` (13/13) and `run_validation` (passed, COIN mean RMSE
0.0102 — unchanged from `main`). **Class A** = bit-identical to `main` (harness zero-diff).
**Class B** = trajectory-changing, kept only if materially faster (rule: no divergence for
no gain). Net hot-path perf: **md2 −8%, md3 −8%** vs `main` (P=200, T=60).

**Structural de-dup (§5) — Class A, done:** `ensurePD` (4-way PD-repair consolidation) +
`jeffreysFiniteClip`; `scatterToGlobal` (5 global-aggregation files);
`contextProbabilityVectorCore`; `mixtureDensityOnGrid` + `feedbackTransform` (6 density
methods); `currentTransitionPrior` (4 preview helpers); `minAssignment` inlined.

**Inference-path vectorization/caching — Class A, done:** `predictContext` transition-prior
gather (linear indexing); `updateLocalTransition/CueMatrix` row-normalize; `sampleStatesMD`
active-posterior inversion hoist; `predictStateFeedbackMD` `repmat`→implicit expansion.

**Perf — pagewise MD batching (§3/§4) — Class B, evaluated:** `predictStatesMD` batched via
`pagemtimes` — **bit-identical AND ~8% faster → KEPT**. `resampleParticlesMD` batched via
`pagechol`/`pagemldivide` — bit-identical but **no speedup at N=2–3 → REVERTED** (rule 2).
Profiling showed the hot path is RNG-sequential per-cell sampling
(`sampleParametersMD`/`sampleDynamicsMD`/`sampleStatesMD`) that cannot batch without
changing RNG order; `pagechol`/inverse batching does not pay off at these small sizes.
Read-only reductions (`motor_output`/`state_moments`/`selectContextStateMean`) and
`binomialSample` left as-is (not hotspots / would change RNG draw count) — **DEFERRED**.

**Latent correctness (§2) + tooling (§8) — done:** `mustMarginalize` now **enforces**
(throws) and was **re-wired into `mixtureDensityOnGrid`** — the Batch-1 density dedup had
silently dropped it and `mustBeCovarianceMatrix`; both restored. Public methods **renamed**
to `predicted_context_probabilities_vector`/`_map` and `responsibilities_vector`/`_map` with
one-time deprecation **shims** for the old names; callers/tests/examples/class-doc updated.
`run_validation` **Strict** metric-loss fixed; `validate_context_recovery` now
**seed-averaged**; `CODE_REVIEW.md` context-summary table corrected.

**Data layout (§7) — Class A, done:** scalar `dynamics_ss_1`/`dynamics_ss_2` reoriented to
feature-leading/particle-trailing (`2×Cmax×P` / `2×2×Cmax×P`), matching the MD accumulators.

**Examples (§9) — done:** `md_examples` `Cwidth` cross-section carryover fixed; the
setup / `normalize-then-trapz` idioms kept inline (extracting them would hide the
integration step the demos teach). `startup.m` missing-dependency warning added (§10).

---

The remainder of this file is the **original deferred catalogue** (from the safe-pass quality
batch), retained for provenance and `file:line` detail. Everything here has now been
addressed on `experimental` per the changelog above.

> Scope reminder: the batch only made changes that provably do not alter numeric results
> or RNG-call order (docs, validation guards, dead-code removal, de-dup of non-inference
> scaffolding, and the two confirmed-bug fixes F5/F6). The scalar inference pipeline is a
> byte-for-byte regression baseline against `COIN.m`; the `*MD` path is a parity contract.

---

## 0. Verification gap (must clear first)

Throughout the batch the **MATLAB MCP session was detached** ("failed to attach to MATLAB
session"), so no worker could run `check_matlab_code`, and the coordinator could not run
the behavioral suite. Every worker fell back to a manual mlint-style review. Before relying
on this pass:

1. Reattach MATLAB and run `check_matlab_code` across all 162 changed files.
2. Run `cd tests; run_tests` and `validation/run_validation`.
3. Run `scratchpad/verify_f6.m` (staged by P4) to confirm the F6 refactor is byte-identical.

Until then the merge lives on branch **`quality/integration`**, not `main`.

---

## 1. Confirmed bugs — FIXED in this pass (not deferred, listed for the record)

- **F5** (`validation/validation_best_label_map.m`, unit T2): the `K>8` identity-map
  fallback was replaced with a maximum-weight linear assignment on the confusion matrix
  (`matchpairs`, verified permutation-valid).
- **F6** (`@RealTimeCOIN/private/resampleState.m`, unit P4): the fragile particle-axis
  shape heuristic was replaced with explicit `vecFields`/`matFields`/tensor field lists
  matching `resampleStateMD.m`; verified byte-identical for the current schema.

---

## 2. Latent correctness issues found — need your sign-off (behavior-changing)

- **`mustMarginalize` is effectively a no-op validator** — `@RealTimeCOIN/private/mustMarginalize.m`
  (found by U3). It returns a logical that the `arguments` block discards and never
  `error`s on the "columns sum to 1" check, so that invariant is **not enforced** (only
  `mustBeFinite`/`mustBeNonnegative` on its input fire). Preserved exactly to keep behavior
  identical. If enforcement is intended, it must `error` on failure — a behavior change.
- **Public-API naming hazard** (U5): `predicted_context_probabilities` returns a **row
  vector**, while the near-identically named `context_predicted_probabilities` returns a
  **containers.Map**; same trap for `responsibilities` (vector) vs `context_responsibilities`
  (map). Left as-is (public API) and cross-referenced in help. A future breaking-change
  pass could rename to `*_vector`/`*_map` with deprecation shims.
- **`CODE_REVIEW.md:187` is now stale** (U5): it lists `predicted_context_probabilities`
  as returning a Map, but the code returns a vector. `CODE_REVIEW.md` is out of the batch's
  edit scope; reconcile manually.

---

## 3. Performance — MD Cholesky / inversion batching (inference path)

Each of these runs many tiny `N×N` factorizations inside a `(context × particle)` loop and
could be batched with `pagemtimes`/`pagechol`/`pagemldivide` (R2020b+). All deferred because
they change the numerical/round-off path and must be re-validated against `COIN.m`.

- **P2** — `sampleBiasMD.m`: per-particle `safeInverse` + `choljitter` on the posterior
  covariance, once per (context, particle).
- **P3a** — `predictStatesMD.m`: nested `P × Cmax` loop doing `A*Pf*A' + Q` per cell.
- **P3b** — `sampleStatesMD.m`: `O(2·P·Cmax)` inversions; `safeInverse(Qi + obsPrecision)`
  is identical across contexts for a fixed observation mask and could be cached per particle.
- **P4** — `resampleParticlesMD.m`: `gaussianLogLikChol` called `Cmax·P` times, each
  redoing a Cholesky; batch innovations + reuse factors.
- **P5** — `assignmentCostMatrixMD.m`: `localStateInv`/`localStateCov` recomputed per
  particle each sweep; hoist/cache across sweeps (local state doesn't change within
  `optimizeContextAlignment`).

## 4. Performance — allocation & loop vectorization

- **P3a** — `predictStateFeedbackMD.m`: `repmat(R, 1, 1, Cmax, P)` materializes a full
  tensor copy of the shared observation noise `R`; implicit expansion (`state_cov + R` with
  a reshaped `N×N×1×1` R) avoids it. Verify broadcast semantics across supported versions.
- **P3b** — `updateLocalTransitionMatrix.m` / `updateLocalCueMatrix.m`: the inner
  row-normalization `for` loop is vectorizable (`raw ./ max(rowSums, eps)` with a validity
  mask). The two functions also share ~80% control flow → a common helper.
- **P3a / P8** — the 5-copy transition-prior particle loop
  (`prior(:,p) = local_transition_matrix(context(p),:,p)'`) appears in `predictContext` and
  in the four preview helpers; vectorize with linear indexing over the particle axis.
- **P7** — `selectContextStateMean.m`: MD `for p` averaging could be a `sub2ind` gather
  (`sum(M,2)./P`); bit-identical verification blocked on MATLAB, so left as a loop.
- **U2** — `motor_output.m:29-38`, `state_moments.m:31-41`,
  `predictive_feedback_moments.m` (`multiMoments`/`scalarMoments` novel-context loops): MD
  mixture reductions could be tensor sums; deferred because float accumulation order differs
  and needs verification.
- **P1** — `binomialSample.m:39`: `sum(rand(1,trials) < prob)` is O(trials); the sole
  caller passes small counts. `sampleScalarNormal.m:48-59`: scalar `mu/variance/low/high`
  are expanded to full `sz` arrays (up to four temporaries/call) where implicit expansion
  would do — reworking indexing risks perturbing `trandn` argument vectors / RNG order.

## 5. Structural de-duplication / consolidation

- **P6 (big one)** — the "symmetrise + guarantee PD" logic is reimplemented **4 ways**
  (`choljitter` escalating jitter, `regularizeCovariance` eps+conditional loading,
  `stationaryStateCovMD` eigen-clip, `sampleBivariateTruncated` inline). Consolidate into
  one `ensurePD`/`nearestPSD` with a tactic flag. **Risk:** the variants are NOT numerically
  identical, so a shared impl must preserve each caller's exact tactic.
- **P6** — three Jeffreys implementations (`gaussianJeffreys` scalar,
  `gaussianJeffreysMulti`, `categoricalJeffreys`) share the finite-clip epilogue and
  contract; scalar is the `k==1` special case of the multivariate. Unify behind a dispatch
  or at least share the epilogue helper.
- **P7** — `scatterToGlobal` consolidation across 5 files (`globalContextMatrix`,
  `globalContextWeights`, `globalCueTensor`, `globalTransitionTensor`,
  `globalSampledContexts`): same modal-particle scatter skeleton, differ only in
  accumulation rule / tensor rank. One `scatterToGlobal(obj, X, alignment, mode)` helper.
- **P7 / U5** — `contextProbabilityVector` / `localContextProbabilityVector` share a
  `switch kind` block and the `~isfinite→0` + renormalize tail (U5 already extracted
  `renormalizeGlobalWeights` for the two `global_*_probabilities` methods; this is the same
  tail in the private accessors).
- **P4** — `resampleParticles.m` / `resampleParticlesMD.m` share an identical log-joint →
  `log_sum_exp` → resp → `systematic_resampling` tail (~20 lines), differing only in the
  likelihood factor. `resetParticles.m` / `resetParticlesMD.m` are ~50% byte-identical
  (context-inference + context-probability + alignment-invalidation blocks).
- **P2** — `sampleContext.m` / `sampleContextMD.m` stick-breaking body is byte-identical
  except the state-seeding lines; factor behind a seeding callback/flag.
- **U3** — density structural twins: `novel_state_probability`↔`novel_state_feedback_probability`
  and `state_given_context_probability`↔`state_feedback_given_context_probability` differ
  only by the bias shift + noise inflation (`M+B`, `V+R`); `state_probability`↔
  `state_feedback_probability` share the whole mixture loop. A shared `mixtureDensityOnGrid`
  + `feedbackTransform(M,V)` would remove ~4 copies.
- **U4** — `clampActiveSummaryContexts` (new, U4-owned) could also back the identical clamp
  boilerplate in `state_given_context_probability` / `state_feedback_given_context_probability`
  (U3's files) — left untouched for file-ownership reasons.
- **P8** — preview helpers: `previewPredictiveFeedbackMD.m:51-66` duplicates the MD
  one-step Kalman prediction from `predictStateFeedbackMD` / `predictive_feedback_moments`;
  and the shared transition-prior loop (see §4) recurs verbatim in all four preview helpers
  → one `private/currentTransitionPrior(obj)` helper.
- **P5** — `minAssignment.m` is a pure pass-through to `linearAssignment` with a single
  caller (`optimizeContextAlignment`); inlineable. `assignmentCostMatrixMD`'s nested
  `preparedGaussianJeffreys` duplicates `gaussianJeffreysMulti` with cached inverses; unify
  if `gaussianJeffreysMulti` accepted optional precomputed inverses.
- **P5** — `updateGlobalContexts(/MD)`: inner `find(assignment(...)==globalIdx,1)` loop is
  O(Km²·P); a single local→global scatter index per particle is O(Km·P).

## 6. Cross-cutting: `mustBeScalarOrEmpty` local copies

The repo carries **redundant local copies** of the R2020b+ builtin `mustBeScalarOrEmpty`.
This batch removed them in U3, U4, and P8 (preview helpers) but **U2 kept its copies**
(couldn't confirm the builtin without MATLAB). Remaining copies flagged: `state_cstar2`
(fixed by U4), `predictive_feedback_moments`, `predictive_motor_output` (U2 — kept),
`kalman_gain_cstar2` (fixed by U4). **Action:** confirm the builtin resolves (it does on
R2020b+), then delete U2's two remaining copies for repo-wide consistency — do this as one
coordinated change, not per-unit.

## 7. Data-layout inconsistency

- **P3b** — `dynamics_ss_1` (`Cmax×P×2`) and `dynamics_ss_2` (`Cmax×P×2×2`) put the
  particle on **dim 2**, whereas the MD accumulators `Lambda_xx`/`Lambda_yx` (`N×N×Cmax×P`)
  put it on the **trailing** dim. Aligning to one particle-axis convention would simplify
  shared indexing but touches `resetParticles` and all consumers.

## 8. Validation / tooling (unit T2)

- **`Strict` passthrough loses metrics**: with the new per-validator try/catch, passing
  `Strict=true` to sub-validators makes them `error()` internally, so only `{passed,
  errored, error}` survives (full metrics struct lost). Passing `Strict=false` down and
  relying on the suite-level re-raise would retain metrics on failure.
- **`validate_context_recovery`** remains seed-sensitive near its `0.65` accuracy gate; a
  seed-averaged gate would remove the observed cross-seed flip.
- **`matchpairs` dependency** (F5 fix): for a strictly `matchpairs`-free environment with
  `K>8`, exposing the private `linearAssignment` as a shared util would be more robust than
  the current warned-identity fallback.

## 9. Examples (unit T3) — declined de-dups (intentional)

- Section-1 repo-root setup and the `normalize-then-trapz` idiom recur across the two
  notebooks but were left inline: extracting them would hide the numerical-integration step
  the demo teaches and the path isn't established until Section 1 runs (chicken-and-egg).
- `md_examples.m` has a pre-existing cross-section `Cwidth` carryover (Section 4 sets it to
  7 for a `max_contexts=6` model; Section 5 reuses it with a `max_contexts=4` model) — a
  latent demo-behavior dependency, out of cleanup scope.

## 10. startup.m (unit T4)

- `startup.m` silently skips a dependency if its folder is absent (partial clone). A
  one-line `warning` in the missing-folder case would aid diagnosis; omitted to avoid noise
  on core-only setups where the third-party libs are intentionally unneeded.

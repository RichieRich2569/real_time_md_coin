# Real‑Time COIN MATLAB Implementation

This repository provides a MATLAB re‑implementation of the Contextual
Inference (COIN) motor‑learning model (Heald et al.), designed for
*real‑time* / sequential operation. The original COIN code runs
off‑line: the entire sequence of observations is generated inside the
model before any inference is performed. In contrast, `RealTimeCOIN`
exposes a state‑machine API that is updated one observation at a time.
It supports sequential cues and state feedback, maintains a particle
filter internally, and produces contextual probabilities and state
distributions on demand. Both scalar and multi‑dimensional states are
supported.

## Repository layout

* `@RealTimeCOIN/` – the real‑time class (a MATLAB *class folder*).
  `RealTimeCOIN.m` holds the `classdef` (properties + static helpers);
  every public method is its own `.m` file in the folder, and
  `@RealTimeCOIN/private/` holds the internal helpers. See the class
  doc‑comment for the full catalogue of query methods.
* `COIN.m` – the original **off‑line** COIN, kept unmodified as the
  reference / ground‑truth oracle that validation compares against.
* `examples/` – plain‑text live scripts, run section‑by‑section in the
  MATLAB Editor: `scalar_examples.m` (scalar demos) and
  `md_examples.m` (multi‑dimensional demos). The `+coinviz` package
  holds shared plotting helpers.
* `tests/` – fast behavioural tests written as plain functions (not
  `matlab.unittest`) to minimise dependencies, plus `run_tests.m`.
* `validation/` – scientific validators that report metrics and pass
  flags against `COIN.m` (`run_validation.m`, `validate_*.m`).
* `startup.m` – compiles and paths the third‑party dependencies
  (`lightspeed` and `npbayes-r21`) via MEX.
* `CODE_REVIEW.md` – a detailed, verified code review of
  `@RealTimeCOIN`: the deepest available map of the algorithm, the
  particle‑state layout, and known subtleties.

## Setup

Core `RealTimeCOIN` use only needs the repository root on the path.
The `lightspeed` and `npbayes-r21` MEX dependencies and `COIN.m` are
required only by the off‑line reference and some validation scripts.
On a fresh clone / new machine, run `startup` once to compile the MEX
dependencies:

```matlab
startup            % compiles lightspeed + npbayes-r21 MEX, adds them to path
addpath(pwd)       % put the repo root on the path so RealTimeCOIN resolves
```

## Usage

The API mimics a state machine, driven one trial at a time:

1. `observe_q(q)` – stage the cue for the upcoming trial.
2. `observe_y(y)` – process feedback; this advances the trial and runs
   the full inference pipeline. `[]` or `NaN` is a missing observation
   (still a trial).
3. Query methods read out the current posterior / predictive summaries.

```matlab
coin = RealTimeCOIN('num_particles', 200, 'max_contexts', 5, 'infer_bias', true);

cues      = [1 1 2 2 1 3 1];
feedbacks = [0.2 0.2 0.5 0.5 0.2 -0.1 0.2];   % noisy scalar feedback per trial
grid      = linspace(-1.5, 1.5, 201);

for t = 1:numel(cues)
    coin.observe_q(cues(t));
    coin.observe_y(feedbacks(t));

    dens = coin.state_probability(grid);      % posterior state density on the grid
    resp = coin.responsibilities;             % per-context responsibilities
end
```

For runnable, annotated demos see `examples/scalar_examples.m` and
`examples/md_examples.m` (open in the Editor and run section by section).

### Scalar vs multi‑dimensional

The state dimension is set via the `state_dim` constructor argument.
`state_dim == 1` runs the original scalar pipeline verbatim (this is the
regression baseline validated against `COIN.m`); `state_dim > 1` runs
the multi‑dimensional (`*MD`) variants of the pipeline. Context labels
are per‑particle and arbitrary, so context‑facing summaries use a lazy
global alignment across particles, computed and cached on demand.

Some read‑outs — retention / drift / bias densities and the scalar
Kalman gains — are scalar‑model only (`state_dim == 1`).

## Testing

Run the fast behavioural tests:

```matlab
cd tests; run_tests            % runs every tests/test_*.m
```

Run the scientific validation suite (slower; compares against `COIN.m`):

```matlab
validation/run_validation                       % compact suite, returns a results struct
run_validation('Strict', true, 'MakePlots', true)
```

## Notes

The code is written to be as clear as possible rather than
micro‑optimised; heavy use of vectorisation and pre‑allocation can
improve performance for large particle counts. See `CODE_REVIEW.md` for
a thorough walkthrough of the algorithm and its implementation.

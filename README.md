# realtimecoin

`realtimecoin` is a Python translation of the MATLAB `@RealTimeCOIN` class
folder (kept on this repository's `main` branch): a real-time (sequential)
implementation of the COIN
(COntextual INference) model of motor learning -- Heald, Lengyel & Wolpert
(2021), "Contextual inference underlies the learning of sensorimotor
repertoires", *Nature* 600:489-493. Where the original COIN code is offline (it
generates the whole observation sequence internally, then infers over the block),
`RealTimeCOIN` exposes a state machine that consumes one trial at a time:
`observe_q(q)` stages the sensory cue, `observe_y(y)` runs a full particle-filter
update and advances the trial, and a catalogue of query methods reads out the
current posterior and predictive summaries. Scalar and multi-dimensional latent
states are both supported, along with a run-averaging ensemble wrapper. Only
`numpy` and `scipy` are required.

## Install

```
pip install -e .[test]
```

(`pip install -e .` without the `[test]` extra if you do not need
`pytest`.) Python 3.10 or newer; the runtime dependencies are `numpy>=1.24` and
`scipy>=1.10`.

## Quickstart

```python
import numpy as np
from realtimecoin import RealTimeCOIN

model = RealTimeCOIN(num_particles=200, max_contexts=5, rng=0)

rng = np.random.default_rng(1)
for t in range(60):
    perturbation = 0.0 if t < 30 else 1.0       # a step change halfway through
    model.observe_q(1.0 if t < 30 else 2.0)     # stage the cue for this trial
    model.observe_y(perturbation + 0.03 * rng.standard_normal())

print(model.Trial)                              # 60
print(model.motor_output())                     # expected state feedback
print(model.responsibilities_map())             # {global context label: weight}

grid = np.linspace(-1.0, 2.0, 121)
density = model.state_feedback_probability(grid)          # (121,) densities

model.save_model("coin_model.npz", set_stationary=False)  # exact current state
restored = RealTimeCOIN()
restored.load_model("coin_model.npz")
print(restored.Trial)                           # 60 - resumes where it left off
```

`save_model` defaults to `set_stationary=True`, which writes a
contingency-independent stationary prior instead of the live posterior; pass
`set_stationary=False` (as above) to checkpoint a run in progress.

## Documentation

- [`docs/usage.md`](docs/usage.md) -- the guide: the generative model and the
  per-trial pipeline, the state-machine API, the full constructor-parameter
  table, the query catalogue, scalar vs multi-dimensional dispatch, context
  alignment, persistence, the ensemble, the validation suite, and the
  differences from the MATLAB original.
- [`docs/api_mapping.md`](docs/api_mapping.md) -- a MATLAB method to Python call
  mapping for `@RealTimeCOIN` and `@RealTimeCOINEnsemble`.

## Package layout

| Module | Contents |
| --- | --- |
| `realtimecoin/model.py` | the `RealTimeCOIN` facade: state machine, validation, delegation |
| `realtimecoin/state.py` | `ParticleState` (the particle arrays) and the array-shape contract |
| `realtimecoin/context.py` | the dimension-agnostic context-prediction step |
| `realtimecoin/pipeline_scalar.py` | the `state_dim == 1` per-trial pipeline |
| `realtimecoin/pipeline_md.py` | the multi-dimensional (`*MD`) per-trial pipeline |
| `realtimecoin/queries_core.py` | point predictions, moments, `c*` traces, local context summaries |
| `realtimecoin/queries_density.py` | densities and CDFs |
| `realtimecoin/queries_aligned.py` | context summaries that need the global alignment |
| `realtimecoin/alignment.py` | the cross-particle context alignment |
| `realtimecoin/persist.py` | snapshot / save / load / stationarisation |
| `realtimecoin/samplers.py` | random samplers (gamma, beta, Dirichlet, matrix-normal, ...) |
| `realtimecoin/numerics.py` | deterministic numerical helpers (PD repair, jittered Cholesky, ...) |
| `realtimecoin/statics.py` | the public static helpers (resampling, normal pdf/cdf, log-sum-exp, ...) |
| `realtimecoin/exceptions.py` | the error hierarchy, carrying the MATLAB error identifiers |

## Tests

```
pytest
```

Fast behavioural and numerical tests covering the samplers, both pipelines, the
query catalogue, alignment, persistence and the validators.

## Validation

```
python -m validation.run_validation
```

The scientific validation suite: analytic Kalman-filter oracles for the scalar
and multi-dimensional models, probability-integral-transform self-calibration,
particle-count convergence, context recovery from synthetic data, behavioural
stress cases, and a wall-clock benchmark. Each stage reports metrics and a
`passed` flag rather than asserting; `run(strict=True)` turns a failing suite
into a `RuntimeError`. See the validation section of
[`docs/usage.md`](docs/usage.md) for what each stage checks and what gates it.

## Licence and provenance

Copyright (C) 2026 Richard Marques Monteiro.

`realtimecoin` is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the [`LICENSE`](LICENSE) file, or
<https://www.gnu.org/licenses/>, for the full terms.

The GPL-3 terms are inherited, not chosen. This package is a translation of the
MATLAB `@RealTimeCOIN` implementation on this repository's `main` branch, which
in turn re-implements the COIN model of Heald, Lengyel & Wolpert (2021) for
sequential operation. Parts of the random samplers in
[`realtimecoin/samplers.py`](realtimecoin/samplers.py) were adapted, by way of
COINRL, from Changmin Yu's [`COIN_Python`](https://github.com/changmin-yu/COIN_Python)
port (Copyright (C) 2024 Changmin Yu, "version 3 of the License, or (at your
option) any later version"), which itself derives from James Heald's original
[MATLAB COIN implementation](https://github.com/jamesheald/COIN), licensed
GPL-3.0. Because that code is copyleft, the work as a whole is conveyed under
the same terms.

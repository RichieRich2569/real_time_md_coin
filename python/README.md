# realtimecoin

Python translation of the MATLAB `@RealTimeCOIN` class folder: a real-time
(sequential) implementation of the COIN (COntextual INference) motor-learning
model of Heald et al., driven by a vectorised particle filter.

## Layout

- `realtimecoin/exceptions.py` - error hierarchy preserving the MATLAB error
  identifiers (`RealTimeCOIN:NoCues`, ...).
- `realtimecoin/statics.py` - the public static helpers (resampling, normal
  pdf/cdf, log-sum-exp, stationary distribution, CRP table counts).
- `realtimecoin/samplers.py` - random samplers (gamma, beta, Dirichlet,
  binomial, truncated normal, matrix normal, stable dynamics).
- `realtimecoin/numerics.py` - deterministic numerical helpers (PD repair,
  Cholesky with jitter, safe divide/log/inverse, Jeffreys divergences,
  stationary moments).

## Conventions

- Arrays are **particles-leading with matrix dimensions trailing**. Where the
  MATLAB code uses `(Cmax, P)` this package uses `(P, C)`; `(N, N+1, Cmax, P)`
  becomes `(P, C, N, N+1)`. Batched linear algebra then broadcasts naturally.
- Randomness always flows through an explicit `numpy.random.Generator` passed
  as the first argument; the global `numpy.random` state is never used.

## Development

```
python -m venv .venv
.venv\Scripts\pip install -e .[test]
.venv\Scripts\python -m pytest tests/ -q
```

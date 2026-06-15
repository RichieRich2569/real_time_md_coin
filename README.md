# Real‑Time COIN MATLAB Implementation

This repository provides a MATLAB re‑implementation of the Contextual
Inference (COIN) algorithm designed for *real‑time* operation.  The
original COIN code supplied with the Heald et al. paper runs
off‑line: the entire sequence of observations is generated inside the
model before any inference is performed.  In contrast, this
implementation exposes a state machine API that can be updated one
observation at a time.  It supports sequential cues and state
feedback, maintains a particle filter internally and produces
contextual probabilities and state distributions on demand.

## Contents

The main components of the repository are:

* `RealTimeCOIN.m` – the main class implementing the real‑time COIN
  particle filter.  See the class documentation for full details of
  properties and methods.
* `run_example.m` – a short script demonstrating how to use
  `RealTimeCOIN` to process a stream of cues and observations and
  extract posterior quantities at each trial.
* `tests/` – simple test scripts checking basic behaviour such as
  probability normalisation, Kalman filter consistency in the
  single‑context limit, and serialisation via MATLAB’s `save` and
  `load` functions.

## Usage

To get started, add the `real_time_coin_matlab` directory to your
MATLAB path, then run `run_example.m`:

```matlab
addpath('real_time_coin_matlab');
run_example;
```

The example initialises a `RealTimeCOIN` object with a modest number
of particles, feeds in a short sequence of cues and observations, and
prints the inferred context probabilities and predicted state mean
after each trial.

The API of `RealTimeCOIN` mimics a state machine: call
`observe_q(q)` to register a new cue, `observe_y(y)` to register a
state feedback observation for the current trial, and then query
properties like `context_probabilities` or methods like
`state_probability(values)` to inspect posterior distributions.

## Testing

The `tests` directory contains basic validation scripts.  They are
written as plain functions rather than using the MATLAB Unit Test
framework to minimise dependencies.  You can run them individually or
execute all tests via the helper script `run_tests.m`:

```matlab
addpath('real_time_coin_matlab/tests');
run_tests;
```

## Notes

* This implementation is intended as a demonstration and starting
  point.  It includes lazy global context alignment for context-facing
  summaries, while multi-dimensional contingencies remain future work.
* The code has been written to be as clear as possible rather than
  micro‑optimised.  Heavy use of vectorisation and pre‑allocation can
  improve performance for large particle counts.

## License

This project is released under the MIT License.  See the `LICENSE`
file for details.

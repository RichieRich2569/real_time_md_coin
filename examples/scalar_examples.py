# %% [markdown]
# # RealTimeCOIN - scalar examples
#
# Python port of the MATLAB live script `examples/scalar_examples.m`. Companion
# notebook: `md_examples.py` (the multi-dimensional cases).
#
# The file is a cell-mode script: the `# %%` markers are understood by VS Code,
# Spyder and Jupytext, so you can step through it section by section, but it is
# also a plain program you can run end-to-end with
# `python examples/scalar_examples.py`.
#
# ## Section 1 - Setup
#
# Import the package and the local `viz` plot helpers (the port of the MATLAB
# `+coinviz` package). Run this section once before the others.
#
# Two knobs matter here:
#
# * `RTCOIN_EXAMPLES_FAST=1` in the environment shrinks every long block and
#   every particle count by about 4x, which is what the smoke test uses. The
#   Python filter costs roughly 150 ms per trial at 100 particles, so the long
#   sections take minutes at full size.
# * randomness is split in two: the *data* stream uses its own
#   `numpy.random.Generator` (standing in for MATLAB's `rng(n)` plus `randn`),
#   while each model gets its own seed via `rng=`. The model never touches the
#   global NumPy random state.

# %%
import os
import sys
import tempfile

import numpy as np

# Make the sibling `viz` module importable however the script was started
# (`python examples/scalar_examples.py`, "Run Cell" from the repo root, ...).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # cell mode: __file__ is not defined
    _HERE = ""
    _probe = os.path.abspath(os.getcwd())
    for _ in range(4):
        for _cand in (_probe, os.path.join(_probe, "examples"),
                      os.path.join(_probe, "python", "examples")):
            if os.path.isfile(os.path.join(_cand, "viz.py")):
                _HERE = _cand
                break
        if _HERE:
            break
        _probe = os.path.dirname(_probe)
    if not _HERE:
        raise RuntimeError(
            "Cannot locate examples/viz.py; cd to the repo root and "
            "re-run this section."
        )
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
import matplotlib.pyplot as plt

import viz
from realtimecoin import RealTimeCOIN, RealTimeCOINEnsemble

plt.rcParams["figure.max_open_warning"] = 0

#: Shrink long runs and particle counts when RTCOIN_EXAMPLES_FAST is set.
FAST = os.environ.get("RTCOIN_EXAMPLES_FAST", "").strip().lower() not in (
    "", "0", "false", "no",
)
SCALE = 4 if FAST else 1


def nt(n, minimum=4):
    """Block length `n`, shrunk by `SCALE` in fast mode (never below `minimum`)."""
    return max(minimum, int(round(n / SCALE)))


def npart(n, minimum=20):
    """Particle count `n`, shrunk by `SCALE` in fast mode."""
    return max(minimum, int(round(n / SCALE)))


def nruns(n, minimum=4):
    """Ensemble member count `n`, shrunk by `SCALE` in fast mode."""
    return max(minimum, int(round(n / SCALE)))


# np.trapz was renamed np.trapezoid in NumPy 2.0; the package supports both.
trapz = getattr(np, "trapezoid", None) or np.trapz

print("realtimecoin on path; fast mode: %s (scale 1/%d)" % (FAST, SCALE))

# %% [markdown]
# ## Section 2 - Bias inference on a short cue sequence
#
# The smallest demo: a handful of cued trials with noisy scalar feedback. The
# model infers, per trial, a posterior over the latent state (via numerical
# integration of `state_probability`) and a distribution over contexts.
#
# Note that context labels are 0-based in the Python translation, so the map
# keys start at 0 where MATLAB's start at 1.

# %%
data_rng = np.random.default_rng(1)
coin = RealTimeCOIN(num_particles=npart(200), max_contexts=5, infer_bias=True,
                    rng=1)

cues = [1, 1, 2, 2, 1, 3, 1]
true_values = np.array([0.2, 0.2, 0.5, 0.5, 0.2, -0.1, 0.2])   # latent target
feedbacks = true_values + coin.sigma_sensory_noise * data_rng.standard_normal(
    true_values.shape
)

grid = np.linspace(-1.5, 1.5, 201)
T = len(cues)
pred_mean = np.zeros(T)
for t in range(T):
    coin.observe_q(cues[t])
    coin.observe_y(feedbacks[t])

    dens = coin.state_probability(grid)          # posterior density on the grid
    area = trapz(dens, grid)
    if area > 0:
        dens = dens / area
    pred_mean[t] = trapz(dens * grid, grid)      # posterior mean by integration

    probs = coin.responsibilities_map()          # dict: context -> responsibility
    detail = "".join(" %d:%.2f" % (k, probs[k]) for k in sorted(probs))
    print("Trial %d (cue %d): E[state]=%+.3f, contexts={%s }"
          % (t + 1, cues[t], pred_mean[t], detail))
print("Final trial count (Trial property): %d" % coin.Trial)

viz.state_trace(np.arange(1, T + 1), feedbacks, true_values, pred_mean,
                fig_name="Scalar: short cue sequence",
                predicted_name="Posterior mean")

# %% [markdown]
# ## Section 3 - Long run with missing observations
#
# A 340-trial perturbation schedule that mirrors the original COIN tests: a
# baseline, a positive block, a short negative block, then a long stretch of
# **missing** feedback (NaN), which the model must coast through on its
# dynamics prior alone. The final figure shows the per-context posterior state
# density.

# %%
data_rng = np.random.default_rng(3)
coin = RealTimeCOIN(num_particles=npart(100), rng=3)          # defaults (scalar)

true_pert = np.concatenate([
    np.zeros(nt(50)), np.ones(nt(125)), -np.ones(nt(15)), np.full(nt(150), np.nan),
])
obs = true_pert + coin.sigma_sensory_noise * data_rng.standard_normal(true_pert.shape)

T = obs.size
ctx_width = coin.max_contexts + 1
pred_mean = np.zeros(T)
post_ctx = np.zeros((T, ctx_width))
grid = np.linspace(-1.5, 1.5, 201)
for t in range(T):
    coin.observe_y(obs[t])                       # NaN = missing observation
    post_ctx[t] = coin.context_responsibilities_local()   # fast local weights
    dens = coin.state_probability(grid)
    area = trapz(dens, grid)
    if area > 0:
        dens = dens / area
    pred_mean[t] = trapz(dens * grid, grid)

trials = np.arange(1, T + 1)
viz.state_trace(trials, obs, true_pert, pred_mean,
                fig_name="Scalar: long run (missing obs)",
                predicted_name="Posterior mean")
viz.context_bars(None, post_ctx, fig_name="Scalar: context responsibilities",
                 post_title="Local/modal context responsibilities")

state_by_ctx = coin.state_given_context_probability(grid)     # dict: ctx -> density
viz.density_lines(grid, state_by_ctx,
                  fig_name="Scalar: per-context state density",
                  title="Final posterior state density given aligned context",
                  xlabel="State value")

# %% [markdown]
# ## Section 4 - Tuned "coin-rl" settings with cues
#
# Same idea as Section 3 but with custom noise/retention priors (the settings
# used by the coin-rl tests) and an explicit cue stream, so we can compare the
# **predicted** context probabilities (before each observation) against the
# **responsibilities** (after).

# %%
data_rng = np.random.default_rng(3)
prior_mean_retention = 0.9425
coin = RealTimeCOIN(
    num_particles=npart(100),
    sigma_sensory_noise=0.03,
    sigma_motor_noise=0.0182,
    prior_mean_retention=prior_mean_retention,
    prior_precision_drift=1 / (0.0001 + (1 - prior_mean_retention ** 2)),
    rng=3,
)

block_pert = [0.0, 0.5, -0.5, 0.5]               # target per block
block_cues = [1, 2, 3, 2]                        # cue per block
block_len = nt(80, 15)
true_pert = np.repeat(block_pert, block_len)
cues = np.repeat(block_cues, block_len)
obs = true_pert + coin.sigma_sensory_noise * data_rng.standard_normal(true_pert.shape)

T = obs.size
ctx_width = coin.max_contexts + 1
pred_mean = np.zeros(T)
prev_ctx = np.zeros((T, ctx_width))
post_ctx = np.zeros((T, ctx_width))
grid = np.linspace(-1.5, 1.5, 201)
for t in range(T):
    coin.observe_q(cues[t])
    prev_ctx[t] = coin.predicted_context_probabilities_local()

    coin.observe_y(obs[t])
    post_ctx[t] = coin.context_responsibilities_local()

    dens = coin.state_probability(grid)
    area = trapz(dens, grid)
    if area > 0:
        dens = dens / area
    pred_mean[t] = trapz(dens * grid, grid)
block_edges = block_len * np.arange(1, len(block_cues))

viz.state_trace(np.arange(1, T + 1), obs, true_pert, pred_mean, block_edges,
                fig_name="Scalar: tuned settings",
                predicted_name="Posterior mean")
viz.context_bars(prev_ctx, post_ctx,
                 fig_name="Scalar: predicted vs responsibilities")
viz.density_lines(grid, coin.state_given_context_probability(grid),
                  fig_name="Scalar: per-context state density (tuned)",
                  title="Per-context state density", xlabel="State value")

# %% [markdown]
# ## Section 5 - Save, load and stationarise
#
# `save_model` serialises a trained model to a file; `load_model` restores it
# into a fresh object. With `set_stationary=True` the saved model is first put
# into its steady state (`Trial` reset to 0, context probabilities set to the
# stationary distribution) so it can be reused as a prior. `set_stationary()`
# does that in place. Here we verify a load round-trip reproduces the
# responsibilities.

# %%
data_rng = np.random.default_rng(4)
coin = RealTimeCOIN(num_particles=npart(100), infer_bias=True, rng=4)
cue_seq = [1, 1, 2, 2, 1, 2, 1, 2]
val_seq = [0.2, 0.2, -0.3, -0.3, 0.2, -0.3, 0.2, -0.3]
for q, v in zip(cue_seq, val_seq):
    coin.observe_q(q)
    coin.observe_y(v + coin.sigma_sensory_noise * data_rng.standard_normal())
print("Before save: Trial = %d, active contexts = %d"
      % (coin.Trial, len(coin.responsibilities_map())))

tmp_file = os.path.join(tempfile.gettempdir(), "rtcoin_scalar_example.npz")

# (a) Non-stationary save preserves the live trial state exactly.
coin.save_model(tmp_file, False)
reloaded = RealTimeCOIN(num_particles=npart(100), infer_bias=True)
reloaded.load_model(tmp_file)
r1 = coin.responsibilities_map()
r2 = reloaded.responsibilities_map()
assert set(r1) == set(r2), "the reload must recover the same context set"
max_diff = max(abs(r1[k] - r2[k]) for k in r1)
print("Non-stationary reload: Trial = %d, max responsibility diff = %.2e"
      % (reloaded.Trial, max_diff))

# (b) Stationary save resets Trial while retaining the learned contexts.
coin.save_model(tmp_file, True)
stationary_model = RealTimeCOIN(num_particles=npart(100), infer_bias=True)
stationary_model.load_model(tmp_file)
print("Stationary reload:     Trial = %d (reset), active contexts = %d (retained)"
      % (stationary_model.Trial, len(stationary_model.responsibilities_map())))

# (c) set_stationary applied in place. It drives the model onto the stationary
#     distribution of its own learned context transition matrix, so the analytic
#     stationary_context_probabilities computed *before* the reset matches the
#     distribution the model actually adopts *after* it (over the instantiated
#     contexts).
pi_analytic = coin.stationary_context_probabilities()      # (K,) analytic
coin.set_stationary()
print("After set_stationary:  Trial = %d" % coin.Trial)

predicted = coin.predicted_context_probabilities_vector()  # (max_contexts + 1,)
resp = coin.responsibilities_vector()
kk = pi_analytic.size
pred_known = predicted[:kk] / predicted[:kk].sum()          # drop novel, renormalise
print("stationary_context_probabilities (analytic) : [%s]"
      % " ".join("%.3f" % v for v in pi_analytic))
print("predicted context probs after (known/renorm): [%s]"
      % " ".join("%.3f" % v for v in pred_known))
print("max|analytic - adopted| = %.3f" % np.max(np.abs(pi_analytic - pred_known)))
# After the reset the chain sits at its fixed point, so predicted == responsibilities.
assert np.max(np.abs(predicted - resp)) < 1e-9, \
    "predicted == responsibilities after set_stationary"
assert np.max(np.abs(pi_analytic - pred_known)) < 0.1, \
    "model settles onto its analytic stationary distribution"
os.remove(tmp_file)

# %% [markdown]
# ## Section 6 - Method coverage (remaining scalar API)
#
# This section deliberately calls every public method not already exercised
# above, so the notebook demonstrates the full scalar API surface. It trains a
# short model and then queries it every which way, printing a checklist.
#
# Watch the cue conventions, which differ between methods (and from MATLAB,
# where cue labels are 1-based): `predictive_motor_output`,
# `predictive_state_feedback_cdf` and `predictive_cue_p_value` take a RAW cue
# value, while `predictive_feedback_moments` takes a 0-based cue LABEL.

# %%
data_rng = np.random.default_rng(5)
coin = RealTimeCOIN(num_particles=npart(100), infer_bias=True, rng=5)
cue_seq = [1, 1, 2, 2, 1, 3, 1, 2, 3, 1]
val_seq = [0.2, 0.2, -0.3, -0.3, 0.2, 0.6, 0.2, -0.3, 0.6, 0.2]
for q, v in zip(cue_seq, val_seq):
    coin.observe_q(q)
    coin.observe_y(v + coin.sigma_sensory_noise * data_rng.standard_normal())
grid = np.linspace(-1.5, 1.5, 201)

print("\n================ Scalar method coverage ================")
m_s, v_s = coin.state_moments()
print("state_moments ................ mean %+.3f, var %.4f" % (m_s, v_s))
print("motor_output ................. %+.3f" % coin.motor_output())
print("predictive_motor_output(1) ... %+.3f" % coin.predictive_motor_output(1))
m_f, v_f = coin.predictive_feedback_moments(0)     # 0-based cue LABEL
print("predictive_feedback_moments(0) mean %+.3f, var %.4f" % (m_f, v_f))
print("predictive_state_feedback_cdf(0.2) . %.3f"
      % coin.predictive_state_feedback_cdf(0.2, 1))
print("predictive_cue_p_value(1, 0.5) ..... %.3f"
      % coin.predictive_cue_p_value(1, 0.5))

# Grid densities (feedback space and per-context feedback space).
fb_dens = coin.state_feedback_probability(grid)
print("state_feedback_probability ... integral %.3f" % trapz(fb_dens, grid))
viz.density_lines(grid, coin.state_feedback_given_context_probability(grid),
                  fig_name="Scalar: per-context feedback density",
                  title="state_feedback_given_context_probability",
                  xlabel="Feedback",
                  novel_density=coin.novel_state_feedback_probability(grid))

# Novel (not-yet-instantiated) context density overlaid on the state density.
viz.density_lines(grid, coin.state_given_context_probability(grid),
                  fig_name="Scalar: state density + novel context",
                  title="state_given_context_probability with novel overlay",
                  xlabel="State value",
                  novel_density=coin.novel_state_probability(grid))

# Context summaries: global maps, global vectors, and the alignment struct.
cpm = coin.predicted_context_probabilities_map()
crm = coin.responsibilities_map()
print("predicted_context_probabilities_map .. %d contexts" % len(cpm))
print("responsibilities_map ................. %d contexts" % len(crm))
print("predicted_context_probabilities_vector [%s]"
      % " ".join("%.2f" % v for v in coin.predicted_context_probabilities_vector()))
print("responsibilities_vector .............. [%s]"
      % " ".join("%.2f" % v for v in coin.responsibilities_vector()))
print("sampled_context_count ................ [%s]"
      % " ".join("%.2f" % v for v in coin.sampled_context_count()))
print("sampled_context_count_local .......... [%s]"
      % " ".join("%.2f" % v for v in coin.sampled_context_count_local()))
al = coin.context_alignment()
print("context_alignment .................... dict with keys: %s"
      % ", ".join(sorted(al)))

# Per-trial c*/component scalars (the COIN "single most likely context" traces).
print("explicit_component ........... %+.3f" % coin.explicit_component())
print("implicit_component ........... %+.3f" % coin.implicit_component())
print("state_cstar1/2/3 ............. %+.3f / %+.3f / %+.3f"
      % (coin.state_cstar1(), coin.state_cstar2(), coin.state_cstar3()))
print("predicted_probability_cstar1/3 %.3f / %.3f"
      % (coin.predicted_probability_cstar1(), coin.predicted_probability_cstar3()))
print("kalman_gain_cstar1/2 ......... %.3f / %.3f"
      % (coin.kalman_gain_cstar1(), coin.kalman_gain_cstar2()))

# Transition / cue / stationary distributions over contexts.
ltp = coin.local_transition_probabilities()
lcp = coin.local_cue_probabilities()
scp = coin.stationary_context_probabilities()
gtp = coin.global_transition_probabilities()
gcp = coin.global_cue_probabilities()
print("local_transition_probabilities  %d-by-%d (rows sum to 1: %s)"
      % (ltp.shape[0], ltp.shape[1], bool(np.all(np.abs(ltp.sum(1) - 1) < 1e-9))))
print("local_cue_probabilities ...... %d-by-%d (rows sum to 1: %s)"
      % (lcp.shape[0], lcp.shape[1], bool(np.all(np.abs(lcp.sum(1) - 1) < 1e-9))))
print("stationary_context_probabilities [%s] (sum %.3f)"
      % (" ".join("%.3f" % v for v in scp), scp.sum()))
print("global_transition_probabilities  [%s] (sum %.3f)"
      % (" ".join("%.2f" % v for v in gtp), gtp.sum()))
print("global_cue_probabilities ..... [%s] (sum %.3f)"
      % (" ".join("%.2f" % v for v in gcp), gcp.sum()))
assert np.all(np.abs(ltp.sum(1) - 1) < 1e-9), "local transition rows must sum to 1"
assert abs(scp.sum() - 1) < 1e-9, "stationary distribution must sum to 1"

print("=======================================================")

# Per-context parameter densities (scalar-dynamics only): retention, drift, bias.
r_grid = np.linspace(0.80, 1.00, 201)
d_grid = np.linspace(-0.06, 0.06, 201)
b_grid = np.linspace(-0.80, 0.80, 201)
viz.density_lines(r_grid, coin.retention_given_context_probability(r_grid),
                  fig_name="Scalar: retention | context",
                  title="retention_given_context_probability",
                  xlabel="Retention a")
viz.density_lines(d_grid, coin.drift_given_context_probability(d_grid),
                  fig_name="Scalar: drift | context",
                  title="drift_given_context_probability", xlabel="Drift d")
viz.density_lines(b_grid, coin.bias_given_context_probability(b_grid),
                  fig_name="Scalar: bias | context",
                  title="bias_given_context_probability", xlabel="Bias")
# Marginal (across-context) bias density.
_fig = viz.new_figure("Scalar: marginal bias density")
_ax = _fig.subplots()
_ax.plot(b_grid, coin.bias_probability(b_grid), linewidth=1.4,
         color=viz.palette(1)[0])
_ax.set_xlabel("Bias")
_ax.set_ylabel("Density")
_ax.set_title("bias_probability (marginal)")
_fig.tight_layout()

# %% [markdown]
# ## Section 7 - Explicit vs implicit decomposition (c* traces)
#
# The COIN adaptation curve can be read out in several ways.
# `explicit_component` is the state of the single most-responsible context
# (== `state_cstar1`); `implicit_component` is the motor output minus the
# average state; and the c* traces track the most-probable context's state
# under different timing conventions. Here we record them through a
# perturbation that flips sign.

# %%
data_rng = np.random.default_rng(7)
coin = RealTimeCOIN(num_particles=npart(100), infer_bias=True, rng=7)
lens = [nt(10, 4), nt(20, 6), nt(20, 6), nt(10, 4)]
perturb = np.concatenate([
    np.zeros(lens[0]), 0.3 * np.ones(lens[1]),
    -0.3 * np.ones(lens[2]), np.zeros(lens[3]),
])
T = perturb.size
mo = np.zeros(T)
ex = np.zeros(T)
im = np.zeros(T)
cs1 = np.zeros(T)
cs2 = np.zeros(T)
cs3 = np.zeros(T)
for t in range(T):
    coin.observe_y(perturb[t] + coin.sigma_sensory_noise * data_rng.standard_normal())
    mo[t] = coin.motor_output()
    ex[t] = coin.explicit_component()
    im[t] = coin.implicit_component()
    cs1[t] = coin.state_cstar1()
    cs2[t] = coin.state_cstar2()
    cs3[t] = coin.state_cstar3()
block_edges = np.cumsum(lens)[:-1]
cols = viz.palette(4)
trials = np.arange(1, T + 1)

fig = viz.new_figure("Scalar: explicit / implicit read-outs")
ax = fig.subplots()
ax.plot(trials, perturb, "k--", linewidth=1.2, label="perturbation")
ax.plot(trials, mo, color=cols[0], linewidth=1.6, label="motor_output")
ax.plot(trials, ex, color=cols[1], linewidth=1.4, label="explicit")
ax.plot(trials, im, color=cols[2], linewidth=1.4, label="implicit")
for e in block_edges:
    ax.axvline(e, color=(0.5, 0.5, 0.5), alpha=0.4)
ax.set_xlabel("Trial")
ax.set_ylabel("Adaptation")
ax.set_title("motor_output, explicit_component and implicit_component")
ax.legend(loc="best", fontsize="small")
fig.tight_layout()

# The c* state estimates all track the dominant context's state.
fig = viz.new_figure("Scalar: c* state estimates")
ax = fig.subplots()
ax.plot(trials, perturb, "k--", linewidth=1.2, label="perturbation")
ax.plot(trials, cs1, color=cols[0], linewidth=1.4, label="cstar1")
ax.plot(trials, cs2, color=cols[1], linewidth=1.4, label="cstar2")
ax.plot(trials, cs3, color=cols[2], linewidth=1.4, label="cstar3")
for e in block_edges:
    ax.axvline(e, color=(0.5, 0.5, 0.5), alpha=0.4)
ax.set_xlabel("Trial")
ax.set_ylabel("State estimate")
ax.set_title("state_cstar1 / state_cstar2 / state_cstar3")
ax.legend(loc="best", fontsize="small")
fig.tight_layout()

# explicit_component is exactly state_cstar1 by construction.
assert np.max(np.abs(ex - cs1)) < 1e-9, "explicit_component == state_cstar1"
print("Section 7: max|explicit - state_cstar1| = %.2e" % np.max(np.abs(ex - cs1)))

# %% [markdown]
# ## Section 8 - Prior / hyperparameter exploration
#
# The same feedback stream shown to two differently-tuned models: an "eager"
# prior (small `alpha_context`, large `rho_context`) that instantiates new
# contexts readily, versus a "sticky" prior that resists switching. We compare
# context creation and the c* state read-out. The last (grey) bar is the novel
# context.

# %%
data_rng = np.random.default_rng(8)
cols = viz.palette(4)
lens = [nt(8, 4), nt(16, 6), nt(16, 6)]
perturb = np.concatenate([
    np.zeros(lens[0]), 0.4 * np.ones(lens[1]), -0.4 * np.ones(lens[2]),
])
T = perturb.size
fb = perturb + 0.03 * data_rng.standard_normal(T)   # identical feedback for both

eager = RealTimeCOIN(num_particles=npart(100), alpha_context=2, rho_context=0.6,
                     rng=8)
sticky = RealTimeCOIN(num_particles=npart(100), alpha_context=30,
                      rho_context=0.05, rng=8)
prev_e = np.zeros((T, eager.max_contexts + 1))
prev_s = np.zeros((T, sticky.max_contexts + 1))
cs_e = np.zeros(T)
cs_s = np.zeros(T)
for t in range(T):
    eager.observe_y(fb[t])
    sticky.observe_y(fb[t])
    prev_e[t] = eager.predicted_context_probabilities_vector()
    prev_s[t] = sticky.predicted_context_probabilities_vector()
    cs_e[t] = eager.state_cstar1()
    cs_s[t] = sticky.state_cstar1()

viz.context_bars(prev_e, None, fig_name="Eager prior: context creation",
                 novel_context=True,
                 prev_title="Eager: predicted context probabilities")
viz.context_bars(prev_s, None, fig_name="Sticky prior: context creation",
                 novel_context=True,
                 prev_title="Sticky: predicted context probabilities")
fig = viz.new_figure("Prior comparison: c*1 state")
ax = fig.subplots()
trials = np.arange(1, T + 1)
ax.plot(trials, fb, "x", color=(0.7, 0.7, 0.7), markersize=4, label="feedback")
ax.plot(trials, cs_e, color=cols[0], linewidth=1.5, label="eager")
ax.plot(trials, cs_s, color=cols[1], linewidth=1.5, label="sticky")
ax.set_xlabel("Trial")
ax.set_ylabel("state_cstar1")
ax.set_title("Prior effect on the dominant-context state")
ax.legend(loc="best", fontsize="small")
fig.tight_layout()
print("Eager model contexts: %d;  Sticky model contexts: %d"
      % (eager.diagnostics()["C"], sticky.diagnostics()["C"]))

# %% [markdown]
# ## Section 9 - State|context probability evolution (composite heat-map)
#
# Each trial we evaluate `state_given_context_probability` on a fixed grid and
# stack the columns into one image. Every context is tinted its own colour and
# the densities are additively blended, so you can watch contexts claim regions
# of state space as the contingency changes; the novel context is grey. Below
# the figure we print every inferred per-context parameter.

# %%
data_rng = np.random.default_rng(9)
coin = RealTimeCOIN(num_particles=npart(100), infer_bias=True, rng=9)
blk = nt(15, 6)
perturb = np.concatenate([
    0.20 * np.ones(blk), -0.35 * np.ones(blk), 0.20 * np.ones(blk),
])
cues = np.concatenate([np.ones(blk), 2 * np.ones(blk), np.ones(blk)]).astype(int)
T = perturb.size
s_grid = np.linspace(-0.8, 0.8, 161)
recorded = {}
novel_dens = np.zeros((s_grid.size, T))
for t in range(T):
    coin.observe_q(cues[t])
    coin.observe_y(perturb[t] + coin.sigma_sensory_noise * data_rng.standard_normal())
    dmap = coin.state_given_context_probability(s_grid)
    for key, value in dmap.items():
        recorded.setdefault(key, np.zeros((s_grid.size, T)))[:, t] = value
    novel_dens[:, t] = coin.novel_state_probability(s_grid)
true_line = perturb.copy()

viz.context_density_evolution(
    np.arange(1, T + 1), s_grid, recorded,
    fig_name="Scalar: state|context evolution",
    title="state_given_context_probability over trials (colours = contexts)",
    novel_dens=novel_dens, true_line=true_line, block_edges=[blk, 2 * blk],
)

# Diagnostics: every inferred per-context parameter. The scalar diagnostics
# arrays are (modal particle, context) and the relabelling leaves a hard zero
# in any modal particle that has no local slot mapped to global context c, so
# average only over the particles that actually carry the context.
D = coin.diagnostics()


def ctx_mean(field, c, populated):
    """Mean of diagnostics `field` for context `c` over the particles holding it."""
    column = np.asarray(D[field], dtype=float)[:, c]
    return float(np.mean(column[populated])) if populated.any() else float("nan")


print("\nInferred contexts (K = %d):" % D["C"])
for c in range(D["C"]):
    held = np.asarray(D["retention"], dtype=float)[:, c] != 0.0
    print("  ctx %d: retention %.3f  drift %+.4f  bias %+.3f  state_mean %+.3f"
          " (%d/%d modal particles)"
          % (c, ctx_mean("retention", c, held), ctx_mean("drift", c, held),
             ctx_mean("bias", c, held), ctx_mean("state_mean", c, held),
             held.sum(), held.size))

# %% [markdown]
# ## Section 10 - Parallel runs: probability averaging across an ensemble
#
# `RealTimeCOINEnsemble` runs R independent `RealTimeCOIN` filters on the
# **same** feedback stream and returns their equal-weight average. Each run
# draws from its own reproducible substream (seeded from `seed`), so the runs
# are independent yet the whole ensemble is deterministic given `seed`. A single
# run is a noisy Monte-Carlo estimate of the model's expected output; averaging
# R runs cuts that estimation noise by roughly a factor of sqrt(R). Below, five
# individual runs (thin) scatter around the 12-run ensemble average (bold).
#
# Run counts here are deliberately modest (12, not the 30-50 an offline study
# would use) to keep this notebook to a few minutes end to end. Nothing about
# the behaviour changes with R -- only the Monte-Carlo error, as 1/sqrt(R).

# %%
data_rng = np.random.default_rng(10)
lens = [nt(10, 4), nt(25, 8), nt(25, 8), nt(15, 5)]
perturb = np.concatenate([
    np.zeros(lens[0]), 0.4 * np.ones(lens[1]),
    -0.3 * np.ones(lens[2]), np.zeros(lens[3]),
])
T = perturb.size
fb = perturb + 0.03 * data_rng.standard_normal(T)   # one stream, shared by all
n_show = 5
n_p = npart(40, 10)

indiv = np.zeros((n_show, T))                       # single-run trajectories
for s in range(n_show):
    m = RealTimeCOINEnsemble(runs=1, seed=s + 1, num_particles=n_p)
    for t in range(T):
        m.observe_y(fb[t])
        indiv[s, t] = m.motor_output()
ens = RealTimeCOINEnsemble(runs=nruns(12), seed=101, num_particles=n_p)
mo_ens = np.zeros(T)
for t in range(T):
    ens.observe_y(fb[t])
    mo_ens[t] = ens.motor_output()

cols = viz.palette(3)
trials = np.arange(1, T + 1)
fig = viz.new_figure("Ensemble: probability averaging across runs")
ax = fig.subplots()
ax.plot(trials, perturb, "k--", linewidth=1.2, label="perturbation")
for s in range(n_show):
    ax.plot(trials, indiv[s], "-", color=cols[1], alpha=0.35, linewidth=0.8,
            label="individual runs" if s == 0 else None)
ax.plot(trials, mo_ens, "-", color=cols[0], linewidth=2.0,
        label="%d-run ensemble average" % nruns(12))
ax.set_xlabel("Trial")
ax.set_ylabel("motor_output")
ax.legend(loc="best", fontsize="small")
ax.set_title("Five single runs (thin) vs the ensemble average (bold)")
fig.tight_layout()

across_run_std = float(np.mean(np.std(indiv, axis=0, ddof=1)))
r = nruns(12)
print("Mean across-run std of a single run = %.4f;  ~%d-run estimator SE = "
      "%.4f (that / sqrt(%d))"
      % (across_run_std, r, across_run_std / np.sqrt(r), r))

# Reproducibility: same seed => bit-identical ensemble.
ens_a = RealTimeCOINEnsemble(runs=nruns(4), seed=99, num_particles=npart(40))
ens_b = RealTimeCOINEnsemble(runs=nruns(4), seed=99, num_particles=npart(40))
for t in range(T):
    ens_a.observe_y(fb[t])
    ens_b.observe_y(fb[t])
print("Reproducibility (same seed): max|motor diff| = %.2e over %d trials"
      % (abs(ens_a.motor_output() - ens_b.motor_output()), T))

# %% [markdown]
# ## Section 11 - Batch replay across runs
#
# When the whole observation sequence is known in advance, `simulate(q_seq,
# y_seq)` replays every run in a single call and returns per-trial run-averaged
# traces: `motor_output`, the pooled `state_mean`, and the pooled predictive
# `state_var`. It is **bit-identical** to stepping the ensemble trial by trial,
# which the cell checks. The shaded band is +/- one pooled predictive standard
# deviation.
#
# `max_cores > 0` dispatches the runs across worker processes, again
# bit-identically (only throughput changes). This cell keeps `max_cores=0`:
# on Windows and macOS `multiprocessing` uses the *spawn* start method, so each
# worker re-imports the module that launched the program - which a cell-mode
# script or notebook cannot guard with `if __name__ == "__main__":`. In an
# ordinary script, put the `simulate` call under that guard and set
# `max_cores` to the worker count you want.
#
# Size note: 6 members x 80 trials at 60 particles, not the 50 x 120 x 100 an
# offline study would use, so the full-size notebook stays within a few
# minutes (this cell runs the whole schedule TWICE, once batched and once
# stepped). The averaging behaviour is identical; only the Monte-Carlo error
# scales, as 1/sqrt(R).

# %%
import time

def trace(traces, name):
    """Read a named trace out of simulate()'s result.

    The ensemble may return the traces as a mapping or as a small record
    object; both are handled so this cell does not care which.
    """
    value = traces[name] if hasattr(traces, "keys") else getattr(traces, name)
    return np.asarray(value, dtype=float)

data_rng = np.random.default_rng(11)
lens = [nt(15, 6), nt(25, 10), nt(25, 10), nt(15, 6)]
perturb = np.concatenate([
    np.zeros(lens[0]), 0.5 * np.ones(lens[1]),
    -0.4 * np.ones(lens[2]), np.zeros(lens[3]),
])
T = perturb.size
cues = np.ones(T, dtype=int)
obs = perturb + 0.03 * data_rng.standard_normal(T)

n_runs = nruns(6)
ens_batch = RealTimeCOINEnsemble(runs=n_runs, seed=7, max_cores=0,
                                 num_particles=npart(60))
t0 = time.perf_counter()
tr_b = ens_batch.simulate(cues, obs)
t_batch = time.perf_counter() - t0

# The same ensemble stepped one trial at a time, as the equivalence check.
ens_step = RealTimeCOINEnsemble(runs=n_runs, seed=7, num_particles=npart(60))
mo_step = np.zeros(T)
t0 = time.perf_counter()
for t in range(T):
    ens_step.observe_q(cues[t])
    ens_step.observe_y(obs[t])
    mo_step[t] = ens_step.motor_output()
t_step = time.perf_counter() - t0
print("%d runs x %d trials: simulate %.2fs, stepping loop %.2fs"
      % (n_runs, T, t_batch, t_step))
print("simulate vs stepping: max|motor diff| = %.2e (bit-identical)"
      % np.max(np.abs(trace(tr_b, "motor_output") - mo_step)))

mu = trace(tr_b, "motor_output")
sd = np.sqrt(np.maximum(trace(tr_b, "state_var"), 0.0))
cols = viz.palette(3)
trials = np.arange(1, T + 1)
fig = viz.new_figure("Ensemble: batch simulate across runs")
ax = fig.subplots()
ax.fill_between(trials, mu - sd, mu + sd, color=cols[0], alpha=0.15,
                linewidth=0, label="+/- 1 pooled SD")
ax.plot(trials, perturb, "k--", linewidth=1.2, label="perturbation")
ax.plot(trials, mu, "-", color=cols[0], linewidth=1.8,
        label="run-averaged motor_output")
ax.set_xlabel("Trial")
ax.set_ylabel("motor_output")
ax.legend(loc="best", fontsize="small")
ax.set_title("Batch replay of %d runs" % n_runs)
fig.tight_layout()

# %% [markdown]
# ## Section 12 - Ensemble context-aligned summaries
#
# Context labels are arbitrary in each run, so before averaging any
# context-indexed quantity the ensemble matches every run's contexts onto a
# common reference frame (by prototype similarity). The averaged
# context-probability vectors still sum to 1, and each per-context state density
# is averaged over the runs that instantiated that context. An A/B/A cued
# schedule instantiates two contexts across 12 runs (kept small for runtime;
# the alignment does not care how many runs there are).
#
# The reference frame is the member holding the most contexts (ties to the
# lowest index); every other member's contexts are matched onto it by minimum
# prototype distance. Probability vectors are ZERO-FILLED before averaging
# (a run lacking a reference context contributes 0), which is what keeps them
# summing to 1; per-context densities use a NaN-OMIT mean over the runs that
# actually hold the context.

# %%
data_rng = np.random.default_rng(12)
block_pert = [0.3, -0.3, 0.3]
block_cue = [1, 2, 1]
block_len = nt(25, 8)
perturb = np.repeat(block_pert, block_len)
cues = np.repeat(block_cue, block_len)
obs = perturb + 0.03 * data_rng.standard_normal(perturb.shape)
T = perturb.size
max_ctx = 5

ens = RealTimeCOINEnsemble(runs=nruns(12), seed=5, max_contexts=max_ctx,
                           num_particles=npart(60))
prev_ctx = np.zeros((T, max_ctx + 1))
post_ctx = np.zeros((T, max_ctx + 1))
for t in range(T):
    ens.observe_q(cues[t])
    prev_ctx[t] = ens.predicted_context_probabilities_vector()
    ens.observe_y(obs[t])
    post_ctx[t] = ens.responsibilities_vector()

viz.context_bars(prev_ctx, post_ctx,
                 fig_name="Ensemble: aligned context probabilities",
                 novel_context=True,
                 prev_title="Run-averaged predicted (aligned)",
                 post_title="Run-averaged responsibilities (aligned)")

grid = np.linspace(-0.8, 0.8, 201)
viz.density_lines(
    grid, ens.state_given_context_probability(grid),
    fig_name="Ensemble: per-context state density",
    title="Run-averaged state_given_context_probability",
    xlabel="State value")
resp = ens.responsibilities_vector()
pi_e = ens.stationary_context_probabilities()
print("Run-averaged responsibilities : [%s] (sum %.3f)"
      % (" ".join("%.2f" % v for v in resp), resp.sum()))
print("Ensemble stationary context   : [%s] (sum %.3f)"
      % (" ".join("%.3f" % v for v in pi_e), pi_e.sum()))
assert abs(resp.sum() - 1) < 1e-9, "aligned responsibilities sum to 1"

# %% [markdown]
# ## Wrap-up
#
# Nothing above called `plt.show()`, so the figures appear inline in a notebook
# front-end. When the file is run as a plain program with an interactive
# backend, show them all now.

# %%
if matplotlib.get_backend().lower() != "agg":
    plt.show()
print("scalar_examples: done (%d figures)" % len(plt.get_fignums()))

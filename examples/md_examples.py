# %% [markdown]
# # RealTimeCOIN - multi-dimensional examples
#
# Python port of the MATLAB live script `examples/md_examples.m`. Companion
# notebook: `scalar_examples.py` (the scalar cases).
#
# The file is a cell-mode script: the `# %%` markers are understood by VS Code,
# Spyder and Jupytext, so you can step through it section by section, but it is
# also a plain program you can run end-to-end with
# `python examples/md_examples.py`.
#
# ## Section 1 - Setup
#
# Import the package and the local `viz` plot helpers (the port of the MATLAB
# `+coinviz` package). Run this section once before the others.
#
# As in the scalar notebook, `RTCOIN_EXAMPLES_FAST=1` shrinks every long block
# and every particle count by about 4x, the *data* stream has its own
# `numpy.random.Generator`, and each model is seeded through `rng=`.
#
# One layout convention to keep in mind throughout: per-trial data arrays are
# `(N, T)` exactly as in MATLAB, but density **query grids** are `(K, N)` with
# one query point per ROW (MATLAB passes `N`-by-`K`, one point per column).

# %%
import os
import sys
import tempfile

import numpy as np

# Make the sibling `viz` module importable however the script was started
# (`python examples/md_examples.py`, "Run Cell" from the repo root, ...).
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


def ngrid(n, minimum=20):
    """Density-grid resolution `n` (pixels per axis), shrunk by `SCALE`."""
    return max(minimum, int(round(n / SCALE)))


print("realtimecoin on path; fast mode: %s (scale 1/%d)" % (FAST, SCALE))

# %% [markdown]
# ## Section 2 - Small 2-D demo with correlated noise
#
# A 2-D model with full (correlated) process and observation noise covariances.
# Two cues point at two latent targets. Before each observation we read the
# one-step predictive feedback distribution (`predictive_feedback_moments`) and
# draw it as a +/-1 sd band; after each we read the posterior state moments.
#
# `predictive_feedback_moments` takes a **0-based cue label**, so the raw cues
# 1 and 2 become labels 0 and 1.

# %%
data_rng = np.random.default_rng(7)
N = 2
Q = np.array([[2.5e-3, 1.0e-3], [1.0e-3, 2.0e-3]])
R = np.array([[9.0e-3, -2.0e-3], [-2.0e-3, 1.2e-2]])
coin = RealTimeCOIN(num_particles=npart(300), max_contexts=4, state_dim=N,
                    infer_bias=True, process_noise_covariance=Q,
                    observation_noise_covariance=R, rng=7)

cues = [1, 1, 1, 2, 2, 2, 2, 1, 1, 2]
targets = {1: np.array([0.5, -0.2]), 2: np.array([-0.3, 0.6])}

T = len(cues)
true_state = np.zeros((N, T))
feedbacks = np.zeros((N, T))
pred_mean = np.zeros((N, T))
pred_std = np.zeros((N, T))
post_mean = np.zeros((N, T))
Lr = np.linalg.cholesky(R)
for t in range(T):
    true_state[:, t] = targets[cues[t]]
    feedbacks[:, t] = true_state[:, t] + Lr @ data_rng.standard_normal(N)

    coin.observe_q(cues[t])
    mu, sigma = coin.predictive_feedback_moments(cues[t] - 1)   # read-only
    pred_mean[:, t] = mu
    pred_std[:, t] = np.sqrt(np.maximum(np.diag(sigma), 0.0))

    coin.observe_y(feedbacks[:, t])
    post_mean[:, t] = coin.state_moments()[0]                   # (N,) posterior mean

trials = np.arange(1, T + 1)
viz.state_trace(trials, feedbacks, true_state, pred_mean,
                fig_name="MD: small 2-D demo", band=pred_std,
                predicted_name="Predictive +/-1 sd")
viz.trajectory_2d(post_mean, np.column_stack([targets[1], targets[2]]),
                  fig_name="MD: 2-D trajectory",
                  title="Posterior state trajectory",
                  path_name="Posterior mean", target_name="Targets",
                  observed_xy=feedbacks)

# %% [markdown]
# ## Section 3 - Long 2-D cued run with recall, densities and the novel context
#
# 500 trials over four cued 2-D "contingencies" (distinct latent targets),
# including a **recall** of context 1 to show a previously learned contingency
# is re-engaged rather than relearned. At the end we draw per-context density
# heat-maps in both feedback and state space, including the novel
# (not-yet-instantiated) context.

# %%
import time

data_rng = np.random.default_rng(11)
N = 2
max_contexts = 6
coin = RealTimeCOIN(num_particles=npart(200), max_contexts=max_contexts,
                    state_dim=N, infer_bias=True, rng=11)

targets = np.array([[0.6, -0.4, 0.2, 0.6],
                    [-0.3, 0.5, 0.6, -0.3]])
block_cues = [1, 2, 3, 1]
block_lens = [nt(150, 20), nt(150, 20), nt(130, 18), nt(70, 10)]
T = int(np.sum(block_lens))
cues = np.zeros(T, dtype=int)
true_state = np.zeros((N, T))
bnd = np.concatenate([[0], np.cumsum(block_lens)])
for b in range(len(block_lens)):
    idx = slice(bnd[b], bnd[b + 1])
    cues[idx] = block_cues[b]
    true_state[:, idx] = targets[:, [b]]
feedbacks = true_state + coin.sigma_sensory_noise * 0.01 * data_rng.standard_normal(
    (N, T)
)

c_width = max_contexts + 1
motor_output = np.zeros((N, T))
post_state_mean = np.zeros((N, T))
prev_ctx = np.zeros((T, c_width))
post_ctx = np.zeros((T, c_width))
print("Running %d-trial 2-D cued example..." % T)
t0 = time.perf_counter()
for t in range(T):
    coin.observe_q(cues[t])
    motor_output[:, t] = coin.predictive_motor_output(cues[t])  # expected feedback
    prev_ctx[t] = coin.predicted_context_probabilities_local()

    coin.observe_y(feedbacks[:, t])
    post_state_mean[:, t] = coin.state_moments()[0]
    post_ctx[t] = coin.context_responsibilities_local()
print("Done in %.2f s. Sampled-context occupancy: [%s]"
      % (time.perf_counter() - t0,
         " ".join("%.2f" % v for v in coin.sampled_context_count())))

block_edges = bnd[1:-1]
viz.state_trace(np.arange(1, T + 1), feedbacks, true_state, motor_output,
                block_edges, fig_name="MD: motor output vs truth",
                predicted_name="Motor output")
viz.context_bars(prev_ctx, post_ctx, fig_name="MD: context probabilities",
                 novel_context=True)
viz.trajectory_2d(motor_output, targets, fig_name="MD: motor-output path",
                  title="Motor-output path and block targets",
                  path_name="Motor output")

# Per-context densities incl. the novel context, in feedback and state space.
D = coin.diagnostics()
dominant_cue = np.argmax(D["cue_prob"], axis=1)          # 0-based cue label per ctx
ctx_keys = sorted(coin.state_given_context_probability(targets[:, 0][None, :]))
ctx_targets = np.zeros((N, len(ctx_keys)))
for i, key in enumerate(ctx_keys):
    # Cue labels are 0-based, the raw block cues 1-based; match them up.
    matches = [b for b, c in enumerate(block_cues) if c - 1 == dominant_cue[key]]
    ctx_targets[:, i] = targets[:, matches[0] if matches else 0]
viz.density_heatmaps(coin, D, ctx_keys, ctx_targets, space="feedback",
                     cue_labels=dominant_cue, grid=ngrid(120, 40))
viz.density_heatmaps(coin, D, ctx_keys, ctx_targets, space="state",
                     cue_labels=dominant_cue, grid=ngrid(120, 40))

# Inferred per-context parameters (multi-dimensional): retention matrix A,
# drift and bias vectors, plus the transition and cue prototypes.
print("\nInferred MD contexts (K = %d):" % D["K"])
for i in range(D["K"]):
    print("  ctx %d: drift [%+.3f %+.3f]  bias [%+.3f %+.3f]  "
          "state_mean [%+.3f %+.3f]"
          % (i, D["drift"][i, 0], D["drift"][i, 1], D["bias"][i, 0],
             D["bias"][i, 1], D["state_mean"][i, 0], D["state_mean"][i, 1]))
    print("         A = [%+.3f %+.3f ; %+.3f %+.3f]"
          % (D["A"][i, 0, 0], D["A"][i, 0, 1], D["A"][i, 1, 0], D["A"][i, 1, 1]))
print("  transition_prob (K x K+1):")
print(np.round(D["transition_prob"], 4))
print("  cue_prob (K x Q):")
print(np.round(D["cue_prob"], 4))

# %% [markdown]
# ## Section 4 - N = 4: fixed, drifting and dimension-swapping contingencies
#
# A higher-dimensional, **cue-free** run over three visually distinct dynamic
# regimes, then a return to the first: (1) a fixed target, (2) a linearly
# drifting target, (3) a "swap" regime whose latent dynamics exchange dims
# 1 <-> 2 each trial. We track an "effective number of contexts" as the count of
# context slots that have ever carried appreciable responsibility.

# %%
data_rng = np.random.default_rng(21)
N = 4
coin = RealTimeCOIN(num_particles=npart(200), max_contexts=6, state_dim=N,
                    infer_bias=False,
                    process_noise_covariance=np.diag(
                        np.array([0.05, 0.10, 0.15, 0.20]) ** 2),
                    rng=21)

block_lens = [nt(90, 12), nt(80, 12), nt(80, 12), nt(90, 12)]
T = int(np.sum(block_lens))
bnd = np.concatenate([[0], np.cumsum(block_lens)])
true_state = np.zeros((N, T))
fixed_tgt = np.array([0.5, -0.5, 0.5, -0.5])
swap_a = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                  dtype=float)       # swaps dims 1 and 2
for b in range(len(block_lens)):
    idx = np.arange(bnd[b], bnd[b + 1])
    if b in (0, 3):                              # fixed target (block 4 = recall)
        true_state[:, idx] = fixed_tgt[:, None]
    elif b == 1:                                 # linear drift
        k = np.arange(idx.size)
        true_state[:, idx] = (fixed_tgt[:, None]
                              + np.array([0.004, -0.003, 0.002, -0.001])[:, None] * k)
    else:                                        # dimension-swapping dynamics
        s = fixed_tgt.copy()
        for col in idx:
            s = swap_a @ s
            true_state[:, col] = s
feedbacks = true_state + 0.02 * data_rng.standard_normal((N, T))

c_width = coin.max_contexts + 1
post_state_mean = np.zeros((N, T))
pred_feedback = np.zeros((N, T))
post_ctx = np.zeros((T, c_width))
eff_contexts = np.zeros(T)
print("Running %d-trial N=4 example..." % T)
for t in range(T):
    pred_feedback[:, t] = coin.predictive_motor_output()   # no cue -> pending state
    coin.observe_y(feedbacks[:, t])
    post_state_mean[:, t] = coin.state_moments()[0]
    post_ctx[t] = coin.context_responsibilities_local()
    eff_contexts[t] = np.count_nonzero(post_ctx[: t + 1].max(axis=0) > 0.05)
block_edges = bnd[1:-1]

viz.state_trace(np.arange(1, T + 1), feedbacks, true_state, post_state_mean,
                block_edges, fig_name="MD N=4: state tracking",
                predicted_name="Posterior mean")
viz.context_bars(None, post_ctx, fig_name="MD N=4: context responsibilities",
                 post_title="Context responsibilities "
                            "(fixed / drift / swap / recall)")
fig = viz.new_figure("MD N=4: effective number of contexts")
ax = fig.subplots()
ax.plot(np.arange(1, T + 1), eff_contexts, linewidth=1.4, color=viz.palette(1)[0])
for e in block_edges:
    ax.axvline(e, color=(0.5, 0.5, 0.5), alpha=0.4)
ax.set_xlabel("Trial")
ax.set_ylabel("Effective # contexts")
ax.set_title("Contexts discovered over the run")
fig.tight_layout()

# %% [markdown]
# ## Section 5 - Rotation / oscillation pattern
#
# A 2-D latent target that traces a slowly shrinking spiral (a rotation applied
# each trial). This is a "pattern worth identifying": the observations circle
# the origin, and the motor output should follow the spiral with a lag. A
# distinctive trajectory plot results.

# %%
data_rng = np.random.default_rng(31)
N = 2
coin = RealTimeCOIN(num_particles=npart(300), max_contexts=4, state_dim=N,
                    infer_bias=True, rng=31)

T = nt(240, 30)
omega = 2 * np.pi / 60
decay = 0.995
rot = np.array([[np.cos(omega), -np.sin(omega)],
                [np.sin(omega), np.cos(omega)]])
true_state = np.zeros((N, T))
s = np.array([0.8, 0.0])
for t in range(T):
    true_state[:, t] = s
    s = decay * (rot @ s)
feedbacks = true_state + 0.02 * data_rng.standard_normal((N, T))

motor_output = np.zeros((N, T))
post_mean = np.zeros((N, T))
c_width = coin.max_contexts + 1        # from THIS section's model
post_ctx = np.zeros((T, c_width))
for t in range(T):
    motor_output[:, t] = coin.predictive_motor_output()
    coin.observe_y(feedbacks[:, t])
    post_mean[:, t] = coin.state_moments()[0]
    post_ctx[t] = coin.context_responsibilities_local()

# Show the inferred parameters.
info = coin.diagnostics()
print("Retention matrices A (K x N x N):")
print(np.round(info["A"], 4))
print("Drift vectors (K x N):")
print(np.round(info["drift"], 4))
print("Bias vectors (K x N):")
print(np.round(info["bias"], 4))

viz.state_trace(np.arange(1, T + 1), feedbacks, true_state, post_mean,
                fig_name="MD: oscillation (per dim)",
                predicted_name="Posterior mean")
viz.trajectory_2d(post_mean, None, fig_name="MD: spiral trajectory",
                  title="Rotation / oscillation pattern",
                  path_name="Posterior mean", observed_xy=feedbacks)
viz.context_bars(None, post_ctx,
                 fig_name="Rotation/oscillation context responsibilities",
                 post_title="Context responsibilities")

# %% [markdown]
# ## Section 6 - N = 6 with structured (correlated) covariance
#
# A larger 6-D model with fully correlated, non-isotropic process and
# observation covariances. Two cued targets alternate. Besides tracking, we
# check predictive **calibration** with `predictive_state_feedback_cdf`: the
# probability-integral-transform values should be roughly uniform on [0, 1] if
# the predictive distribution is well calibrated.

# %%
data_rng = np.random.default_rng(41)
N = 6
B = 0.04 * (np.eye(N) + 0.5 * np.diag(np.ones(N - 1), 1)
            + 0.5 * np.diag(np.ones(N - 1), -1))
Q = B @ B.T                                       # SPD process covariance
C = 0.06 * (np.eye(N) + 0.3 * np.triu(np.ones((N, N)), 1)
            + 0.3 * np.tril(np.ones((N, N)), -1)) / N
R = C @ C.T + 1e-3 * np.eye(N)                    # SPD observation covariance
coin = RealTimeCOIN(num_particles=npart(150), max_contexts=4, state_dim=N,
                    infer_bias=True, process_noise_covariance=Q,
                    observation_noise_covariance=R, rng=41)

tgt_a = np.linspace(0.5, -0.5, N)
tgt_b = -tgt_a
block_lens = [nt(70, 10), nt(70, 10), nt(60, 10)]
block_cues = [1, 2, 1]
T = int(np.sum(block_lens))
bnd = np.concatenate([[0], np.cumsum(block_lens)])
cues = np.zeros(T, dtype=int)
true_state = np.zeros((N, T))
for b in range(len(block_lens)):
    idx = slice(bnd[b], bnd[b + 1])
    cues[idx] = block_cues[b]
    true_state[:, idx] = (tgt_b if block_cues[b] == 2 else tgt_a)[:, None]
Lr = np.linalg.cholesky(R)
feedbacks = true_state + Lr @ data_rng.standard_normal((N, T))

post_mean = np.zeros((N, T))
pit = np.zeros((N, T))
for t in range(T):
    coin.observe_q(cues[t])
    pit[:, t] = coin.predictive_state_feedback_cdf(feedbacks[:, t], cues[t])
    coin.observe_y(feedbacks[:, t])
    post_mean[:, t] = coin.state_moments()[0]
block_edges = bnd[1:-1]

viz.state_trace(np.arange(1, T + 1), feedbacks, true_state, post_mean,
                block_edges, fig_name="MD N=6: state tracking",
                predicted_name="Posterior mean")
fig = viz.new_figure("MD N=6: predictive calibration (PIT)")
ax = fig.subplots()
ax.hist(pit.ravel(), bins=10, range=(0.0, 1.0), density=True,
        color=viz.palette(1)[0])
ax.plot([0, 1], [1, 1], "k--", linewidth=1.2)
ax.set_xlabel("Probability-integral-transform value")
ax.set_ylabel("Density")
ax.set_title("PIT of one-step predictions (flat = well calibrated)")
fig.tight_layout()

# %% [markdown]
# ## Section 7 - Correlated dimensions with a strong recall block
#
# Two 2-D contexts whose dimensions are strongly correlated, presented as A, B,
# then A again. The point is the recall: when context A returns, the
# responsibilities should jump back to the original A context rather than
# spawning a new one. Watch the first context re-engage in the third block.

# %%
data_rng = np.random.default_rng(51)
N = 2
Q = np.array([[3e-3, 2.6e-3], [2.6e-3, 3e-3]])      # strongly correlated dims
coin = RealTimeCOIN(num_particles=npart(300), max_contexts=5, state_dim=N,
                    infer_bias=True, process_noise_covariance=Q, rng=51)

tgt_a = np.array([0.6, 0.5])
tgt_b = np.array([-0.5, -0.6])
block_lens = [nt(120, 16), nt(120, 16), nt(120, 16)]
which_t = [tgt_a, tgt_b, tgt_a]
T = int(np.sum(block_lens))
bnd = np.concatenate([[0], np.cumsum(block_lens)])
cues = np.zeros(T, dtype=int)
true_state = np.zeros((N, T))
for b in range(len(block_lens)):
    idx = slice(bnd[b], bnd[b + 1])
    cues[idx] = int(which_t[b][0] > 0) + 1          # cue 1 for A, 2 for B
    true_state[:, idx] = which_t[b][:, None]
feedbacks = true_state + 0.02 * data_rng.standard_normal((N, T))

c_width = coin.max_contexts + 1
post_mean = np.zeros((N, T))
post_ctx = np.zeros((T, c_width))
for t in range(T):
    coin.observe_q(cues[t])
    coin.observe_y(feedbacks[:, t])
    post_mean[:, t] = coin.state_moments()[0]
    post_ctx[t] = coin.context_responsibilities_local()
block_edges = bnd[1:-1]

viz.state_trace(np.arange(1, T + 1), feedbacks, true_state, post_mean,
                block_edges, fig_name="MD: correlated dims + recall",
                predicted_name="Posterior mean")
viz.context_bars(None, post_ctx, fig_name="MD: recall responsibilities",
                 post_title="Responsibilities (A, B, A) - "
                            "watch context 0 re-engage")
viz.trajectory_2d(post_mean, np.column_stack([tgt_a, tgt_b]),
                  fig_name="MD: recall trajectory", title="A -> B -> A recall",
                  path_name="Posterior mean", observed_xy=feedbacks)

# %% [markdown]
# ## Section 8 - MD method coverage (remaining API) and stationary save/load
#
# Trains a compact 2-D model, then calls every remaining public method in the
# multi-dimensional setting and prints a checklist, finishing with a
# `set_stationary` / save / load round-trip in MD.

# %%
data_rng = np.random.default_rng(61)
N = 2
coin = RealTimeCOIN(num_particles=npart(200), max_contexts=4, state_dim=N,
                    infer_bias=True, rng=61)
cues = [1, 1, 2, 2, 1, 2, 1, 2, 1, 2]
tgt = {1: np.array([0.5, -0.3]), 2: np.array([-0.4, 0.5])}
for q in cues:
    coin.observe_q(q)
    coin.observe_y(tgt[q] + 0.02 * data_rng.standard_normal(N))

print("\n================ MD method coverage ================")
print("motor_output ..................... [%s]"
      % " ".join("%.3f" % v for v in coin.motor_output()))
print("predictive_motor_output(1) ....... [%s]"
      % " ".join("%.3f" % v for v in coin.predictive_motor_output(1)))
print("predictive_cue_p_value(2, 0.5) ... %.3f"
      % coin.predictive_cue_p_value(2, 0.5))
m_s, c_s = coin.state_moments()
print("state_moments .................... mean [%s], cov diag [%s]"
      % (" ".join("%.3f" % v for v in m_s),
         " ".join("%.4f" % v for v in np.diag(c_s))))
print("predicted_context_probabilities .. [%s]"
      % " ".join("%.2f" % v for v in coin.predicted_context_probabilities_vector()))
print("responsibilities ................. [%s]"
      % " ".join("%.2f" % v for v in coin.responsibilities_vector()))
print("sampled_context_count ............ [%s]"
      % " ".join("%.2f" % v for v in coin.sampled_context_count()))
print("sampled_context_count_local ...... [%s]"
      % " ".join("%.2f" % v for v in coin.sampled_context_count_local()))
print("predicted_context_probabilities_map %d contexts"
      % len(coin.predicted_context_probabilities_map()))
print("responsibilities_map ............. %d contexts"
      % len(coin.responsibilities_map()))
print("context_alignment ................ K = %d" % coin.context_alignment()["K"])

# The per-trial c*/component read-outs return (N,) states in MD.
print("explicit_component ............... [%s]"
      % " ".join("%.3f" % v for v in coin.explicit_component()))
print("implicit_component ............... [%s]"
      % " ".join("%.3f" % v for v in coin.implicit_component()))
print("state_cstar1 ..................... [%s]"
      % " ".join("%.3f" % v for v in coin.state_cstar1()))
print("state_cstar2 ..................... [%s]"
      % " ".join("%.3f" % v for v in coin.state_cstar2()))
print("state_cstar3 ..................... [%s]"
      % " ".join("%.3f" % v for v in coin.state_cstar3()))
print("predicted_probability_cstar1/3 ... %.3f / %.3f"
      % (coin.predicted_probability_cstar1(), coin.predicted_probability_cstar3()))
# Transition / cue / stationary distributions (dimension-independent).
ltp = coin.local_transition_probabilities()
lcp = coin.local_cue_probabilities()
scp = coin.stationary_context_probabilities()
print("local_transition_probabilities ... %d-by-%d (rows sum to 1: %s)"
      % (ltp.shape[0], ltp.shape[1], bool(np.all(np.abs(ltp.sum(1) - 1) < 1e-9))))
print("local_cue_probabilities .......... %d-by-%d" % (lcp.shape[0], lcp.shape[1]))
print("stationary_context_probabilities . [%s]"
      % " ".join("%.3f" % v for v in scp))
print("global_transition_probabilities .. [%s]"
      % " ".join("%.2f" % v for v in coin.global_transition_probabilities()))
print("global_cue_probabilities ......... [%s]"
      % " ".join("%.2f" % v for v in coin.global_cue_probabilities()))

# Scalar-dynamics-only methods must reject an N-dimensional model. Only
# ScalarModelOnlyError counts as a correct guard; anything else is a real bug.
from realtimecoin import ScalarModelOnlyError    # noqa: E402  (cell-local import)

guards = [
    ("kalman_gain_cstar1", lambda: coin.kalman_gain_cstar1()),
    ("kalman_gain_cstar2", lambda: coin.kalman_gain_cstar2()),
    ("retention_given_context_probability",
     lambda: coin.retention_given_context_probability(np.array([0.9]))),
    ("drift_given_context_probability",
     lambda: coin.drift_given_context_probability(np.array([0.0]))),
    ("bias_given_context_probability",
     lambda: coin.bias_given_context_probability(np.array([0.0]))),
]
for name, call in guards:
    try:
        call()
        print("  (unexpected: scalar-only method %s did not raise)" % name)
    except ScalarModelOnlyError:
        print("  guarded scalar-only method -> %s: ScalarModelOnlyError" % name)
    except Exception as exc:
        print("  (unexpected error from %s: %s: %s)"
              % (name, type(exc).__name__, exc))

# Stationary save / load round-trip (MD).
tmp_file = os.path.join(tempfile.gettempdir(), "rtcoin_md_example.npz")
coin.save_model(tmp_file, True)                  # stationarise + save
reloaded = RealTimeCOIN(state_dim=N, infer_bias=True)
reloaded.load_model(tmp_file)
print("save_model/load_model (MD) ....... reloaded Trial = %d "
      "(reset by set_stationary)" % reloaded.Trial)

# set_stationary drives the live model onto the analytic stationary distribution
# of its learned transition matrix (same demonstration as the scalar notebook).
pi_analytic = coin.stationary_context_probabilities()
coin.set_stationary()
print("set_stationary (MD) .............. Trial = %d" % coin.Trial)
predicted = coin.predicted_context_probabilities_vector()
kk = pi_analytic.size
pred_known = predicted[:kk] / predicted[:kk].sum()
print("stationary match (MD): analytic [%s] vs adopted [%s], max diff %.3f"
      % (" ".join("%.3f" % v for v in pi_analytic),
         " ".join("%.3f" % v for v in pred_known),
         np.max(np.abs(pi_analytic - pred_known))))
assert np.max(np.abs(coin.predicted_context_probabilities_vector()
                     - coin.responsibilities_vector())) < 1e-9, \
    "MD predicted == responsibilities after set_stationary"
os.remove(tmp_file)
print("===================================================")

# %% [markdown]
# ## Section 9 - Missing (NaN) observations in the MD model
#
# Two kinds of missingness: (a) whole trials where *every* coordinate is NaN (a
# channel trial - no feedback at all), and (b) trials where only *some*
# coordinates are observed. During a NaN trial the filter propagates the prior,
# so the prediction drifts back toward the process mean and the state posterior
# widens; an observed coordinate still corrects its own dimension. We watch the
# feedback prediction (with a +/-1 sd band) and the per-dimension
# state-probability evolution across the gaps.

# %%
data_rng = np.random.default_rng(90)
N = 2
coin = RealTimeCOIN(num_particles=npart(200), max_contexts=3, state_dim=N,
                    infer_bias=True, rng=90)
tgt = np.array([0.4, -0.3])
T = nt(45, 16)
# 0-based trial indices; at full size these are MATLAB's trials 16-22 and 30-38.
all_nan = np.arange(round(15 * T / 45), round(22 * T / 45))   # (a) every coord NaN
partial = np.arange(round(29 * T / 45), round(38 * T / 45))   # (b) only dim 1 seen
feedbacks = np.full((N, T), np.nan)
pred_mean = np.zeros((N, T))
pred_sd = np.zeros((N, T))
s_grid = np.linspace(-1.0, 1.0, 161)
dens1 = np.zeros((s_grid.size, T))
dens2 = np.zeros((s_grid.size, T))
prev_ctx = np.zeros((T, coin.max_contexts + 1))
for t in range(T):
    coin.observe_q(1)
    mu, sigma = coin.predictive_feedback_moments(0)      # 0-based cue label
    pred_mean[:, t] = mu
    pred_sd[:, t] = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    prev_ctx[t] = coin.predicted_context_probabilities_vector()

    y = tgt + 0.03 * data_rng.standard_normal(N)
    if t in all_nan:
        y = np.array([np.nan, np.nan])                  # all coordinates missing
    elif t in partial:
        y[1] = np.nan                                   # dim 2 unobserved
    feedbacks[:, t] = y
    coin.observe_y(y)

    # Per-dimension marginal state density (slice the other dim at its mean).
    mm = coin.state_moments()[0]
    dens1[:, t] = coin.state_probability(
        np.column_stack([s_grid, np.full(s_grid.size, mm[1])]))
    dens2[:, t] = coin.state_probability(
        np.column_stack([np.full(s_grid.size, mm[0]), s_grid]))
true_state = np.repeat(tgt[:, None], T, axis=1)
miss_trials = np.concatenate([all_nan, partial]) + 1     # 1-based for plotting
trials = np.arange(1, T + 1)

viz.state_trace(trials, feedbacks, true_state, pred_mean,
                fig_name="MD NaN: prediction through missing trials",
                band=pred_sd, predicted_name="Predicted feedback")
viz.density_evolution(trials, s_grid, dens1,
                      fig_name="MD NaN: dim 1 state evolution",
                      title="p(state dim 1) over trials (missing trials dotted)",
                      ylabel="State dim 1", missing_trials=miss_trials)
viz.density_evolution(trials, s_grid, dens2,
                      fig_name="MD NaN: dim 2 state evolution",
                      title="p(state dim 2) over trials (missing trials dotted)",
                      ylabel="State dim 2", missing_trials=miss_trials)
viz.context_bars(prev_ctx, None, fig_name="MD NaN: context probabilities",
                 novel_context=True,
                 prev_title="Predicted context probs across missing trials")
print("MD NaN section: all-NaN trials %d-%d, partial (dim 1 only) %d-%d"
      % (all_nan[0] + 1, all_nan[-1] + 1, partial[0] + 1, partial[-1] + 1))
assert np.all(np.isfinite(pred_mean)) and np.all(np.isfinite(dens1)) \
    and np.all(np.isfinite(dens2)), \
    "predictions and densities remain finite through NaN trials"

# %% [markdown]
# ## Section 10 - MD prior / covariance exploration
#
# The same 2-D cued sequence shown to two models with different
# observation-noise priors: isotropic versus strongly anti-correlated. The
# correlated prior couples the two dimensions during the Kalman update, so the
# same feedback pulls the estimate along a different direction. We compare the
# motor-output trajectories and the number of inferred contexts.

# %%
data_rng = np.random.default_rng(100)
N = 2
cues = [1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2]
tgt_map = {1: np.array([0.5, 0.4]), 2: np.array([-0.4, -0.5])}
Tt = len(cues)
fb = np.zeros((N, Tt))
for t in range(Tt):
    fb[:, t] = tgt_map[cues[t]] + 0.03 * data_rng.standard_normal(N)

r_iso = 1e-3 * np.eye(N)
r_corr = np.array([[1e-3, -0.8e-3], [-0.8e-3, 1e-3]])   # SPD, anti-correlated
iso_model = RealTimeCOIN(num_particles=npart(100), state_dim=N, max_contexts=3,
                         observation_noise_covariance=r_iso, rng=100)
corr_model = RealTimeCOIN(num_particles=npart(100), state_dim=N, max_contexts=3,
                          observation_noise_covariance=r_corr, rng=100)
iso_path = np.zeros((N, Tt))
corr_path = np.zeros((N, Tt))
for t in range(Tt):
    iso_model.observe_q(cues[t])
    iso_model.observe_y(fb[:, t])
    corr_model.observe_q(cues[t])
    corr_model.observe_y(fb[:, t])
    iso_path[:, t] = iso_model.motor_output()
    corr_path[:, t] = corr_model.motor_output()
targets = np.column_stack([tgt_map[1], tgt_map[2]])
viz.trajectory_2d(iso_path, targets, fig_name="MD prior: isotropic R",
                  title="Isotropic observation noise", observed_xy=fb)
viz.trajectory_2d(corr_path, targets, fig_name="MD prior: correlated R",
                  title="Anti-correlated observation noise", observed_xy=fb)
print("Isotropic contexts: %d;  Correlated contexts: %d"
      % (iso_model.diagnostics()["K"], corr_model.diagnostics()["K"]))

# %% [markdown]
# ## Section 11 - Parallel runs: the ensemble (multi-dimensional)
#
# `RealTimeCOINEnsemble` wraps R independent N-dimensional `RealTimeCOIN`
# filters fed the same vector-valued feedback and returns their equal-weight
# average. `simulate` batch-replays every run and returns the per-trial
# run-averaged `motor_output` (N-by-T) plus the pooled `state_mean` /
# `state_var`; context-aligned averages (responsibilities, per-context
# densities) work exactly as in the scalar notebook. Here a 2-D A/B/A recall
# schedule is run over 8 members at 50 particles -- modest sizes, chosen so the
# full-size notebook stays within a few minutes (the schedule is replayed
# twice, once batched and once stepped); only the Monte-Carlo error depends on
# R, as 1/sqrt(R).
#
# `max_cores > 0` would dispatch the runs across worker processes,
# bit-identically. This cell keeps the serial default: on Windows and macOS
# `multiprocessing` spawns workers by re-importing the launching module, which a
# cell-mode script or notebook cannot guard with `if __name__ == "__main__":`.
# In an ordinary script, put the `simulate` call under that guard and set
# `max_cores`.

# %%
def trace(traces, name):
    """Read a named trace out of simulate()'s result.

    The ensemble may return the traces as a mapping or as a small record
    object; both are handled so this cell does not care which.
    """
    value = traces[name] if hasattr(traces, "keys") else getattr(traces, name)
    return np.asarray(value, dtype=float)

data_rng = np.random.default_rng(21)
N = 2
tgt_a = np.array([0.4, -0.3])
tgt_b = np.array([-0.35, 0.25])
block_len = nt(30, 8)
targets_seq = np.concatenate(
    [np.repeat(tgt_a[:, None], block_len, axis=1),
     np.repeat(tgt_b[:, None], block_len, axis=1),
     np.repeat(tgt_a[:, None], block_len, axis=1)], axis=1)
cues = np.concatenate([np.ones(block_len), 2 * np.ones(block_len),
                       np.ones(block_len)]).astype(int)
T = targets_seq.shape[1]
obs = targets_seq + 0.03 * data_rng.standard_normal((N, T))

# Batch replay across runs (serial; see the note above on max_cores).
n_runs = nruns(8)
ens = RealTimeCOINEnsemble(runs=n_runs, seed=3, max_cores=0, state_dim=N,
                           max_contexts=4, num_particles=npart(50))
tr = ens.simulate(cues, obs)
print("Ensemble (MD): runs = %d, motor trace = %s"
      % (n_runs, trace(tr, "motor_output").shape))
viz.trajectory_2d(trace(tr, "motor_output"), np.column_stack([tgt_a, tgt_b]),
                  fig_name="Ensemble MD: run-averaged trajectory",
                  title="Run-averaged motor_output (%d runs)" % n_runs,
                  observed_xy=obs)

# Live stepping, to read the ensemble summaries at the final trial.
ens_live = RealTimeCOINEnsemble(runs=n_runs, seed=3, state_dim=N,
                                max_contexts=4, num_particles=npart(50))
for t in range(T):
    ens_live.observe_q(cues[t])
    ens_live.observe_y(obs[:, t])
mu_e, cov_e = ens_live.state_moments()
print("Pooled state mean = [%+.3f %+.3f]" % (mu_e[0], mu_e[1]))
print("Pooled state covariance:")
print(np.round(cov_e, 6))

resp = ens_live.responsibilities_vector()
print("Run-averaged responsibilities: [%s] (sum %.3f)"
      % (" ".join("%.2f" % v for v in resp), resp.sum()))

# Per-context 2-D state density (reference frame), averaged over runs.
gx = np.linspace(-0.8, 0.8, ngrid(61, 21))
GX, GY = np.meshgrid(gx, gx)
grid_pts = np.column_stack([GX.ravel(), GY.ravel()])   # one point per ROW
dmap = ens_live.state_given_context_probability(grid_pts)
ctx_keys = sorted(dmap)
fig = viz.new_figure("Ensemble MD: per-context state density",
                     figsize=(4.0 * max(len(ctx_keys), 1), 4.0))
axes = fig.subplots(1, max(len(ctx_keys), 1), squeeze=False)[0]
for i, key in enumerate(ctx_keys):
    ax = axes[i]
    im = ax.imshow(np.asarray(dmap[key], float).reshape(GX.shape),
                   extent=(gx[0], gx[-1], gx[0], gx[-1]),
                   origin="lower", aspect="equal", cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("context %d" % key)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
fig.suptitle("Run-averaged state_given_context_probability "
             "(reference frame)")
fig.tight_layout()

# %% [markdown]
# ## Wrap-up
#
# Nothing above called `plt.show()`, so the figures appear inline in a notebook
# front-end. When the file is run as a plain program with an interactive
# backend, show them all now.

# %%
if matplotlib.get_backend().lower() != "agg":
    plt.show()
print("md_examples: done (%d figures)" % len(plt.get_fignums()))

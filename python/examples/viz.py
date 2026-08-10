"""Plot helpers for the ``realtimecoin`` examples (matplotlib port of ``+coinviz``).

Python translation of the eight MATLAB helpers in ``examples/+coinviz``:

======================================= ====================================
this module                              MATLAB counterpart
======================================= ====================================
:func:`palette`                          ``coinviz.palette``
:func:`state_trace`                      ``coinviz.stateTrace``
:func:`density_lines`                    ``coinviz.densityLines``
:func:`density_evolution`                ``coinviz.densityEvolution``
:func:`context_bars`                     ``coinviz.contextBars``
:func:`context_density_evolution`        ``coinviz.contextDensityEvolution``
:func:`trajectory_2d`                    ``coinviz.trajectory2D``
:func:`density_heatmaps`                 ``coinviz.densityHeatmaps``
======================================= ====================================

Every function is a plain function over NumPy arrays that returns the
:class:`matplotlib.figure.Figure` it drew on; the single-axes helpers also take
an optional ``ax`` so they can be embedded in a larger layout.

Two conventions differ from the MATLAB originals, following the Python package:

* context labels are **0-based** (the ``dict`` keys returned by
  ``state_given_context_probability`` and friends), so "Context 0" here is
  MATLAB's "Context 1";
* multi-dimensional density grids are ``(K, N)`` with one query point per ROW,
  whereas MATLAB passes ``N``-by-``K`` with one point per column. The per-trial
  data arrays (``observed``, ``predicted``, ...) keep the MATLAB ``(N, T)``
  orientation, because that is what a trial loop naturally accumulates.

matplotlib is NOT a dependency of the ``realtimecoin`` package - it is imported
lazily here so that importing this module never breaks a plain model install.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "new_figure",
    "palette",
    "state_trace",
    "density_lines",
    "density_evolution",
    "context_bars",
    "context_density_evolution",
    "trajectory_2d",
    "density_heatmaps",
]

#: The Okabe-Ito colourblind-safe qualitative palette, as RGB rows in [0, 1].
#: Identical to the MATLAB ``coinviz.palette`` base table.
_BASE = np.array(
    [
        [0.00, 0.45, 0.70],   # blue
        [0.90, 0.60, 0.00],   # orange
        [0.00, 0.60, 0.50],   # bluish green
        [0.80, 0.40, 0.00],   # vermillion
        [0.35, 0.70, 0.90],   # sky blue
        [0.80, 0.60, 0.70],   # reddish purple
        [0.95, 0.90, 0.25],   # yellow
        [0.00, 0.00, 0.00],   # black
    ]
)

#: Grey used for the novel (not-yet-instantiated) context, everywhere.
_NOVEL_COLOUR = (0.6, 0.6, 0.6)

#: Grey used for raw observation markers, everywhere.
_OBS_COLOUR = (0.7, 0.7, 0.7)


def _plt():
    """Import :mod:`matplotlib.pyplot` lazily, with a helpful error message.

    Returns
    -------
    module
        The :mod:`matplotlib.pyplot` module.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:   # pragma: no cover - depends on the install
        raise ImportError(
            "matplotlib is required by the realtimecoin examples but is not "
            "installed; run 'pip install matplotlib for the examples'."
        ) from exc
    return plt


def new_figure(fig_name, figsize=(8.0, 5.0)):
    """Create a figure and give it a window title, as MATLAB's ``'Name'`` does.

    Parameters
    ----------
    fig_name : str
        Window-title text.
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _plt()
    fig = plt.figure(figsize=figsize)
    fig.set_label(fig_name)
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        # Headless backends (Agg) have no window; setting the title is a no-op
        # there but must not raise.
        try:
            manager.set_window_title(fig_name)
        except Exception:   # pragma: no cover - backend dependent
            pass
    return fig


def _pixel_extent(x, y):
    """Image extent whose OUTER edges sit half a cell beyond the sample centres.

    MATLAB's ``imagesc(x, y, C)`` treats ``x[0]`` and ``x[-1]`` as the centres
    of the first and last cells, whereas matplotlib's ``imshow(extent=...)``
    treats them as the outer edges. Padding by half a cell here keeps the two
    renderings aligned.

    Parameters
    ----------
    x, y : numpy.ndarray
        Monotonic 1-D sample positions along the horizontal and vertical axes.

    Returns
    -------
    tuple of float
        ``(x_left, x_right, y_bottom, y_top)`` for ``imshow``.
    """
    dx = (x[-1] - x[0]) / (2 * (x.size - 1)) if x.size > 1 else 0.5
    dy = (y[-1] - y[0]) / (2 * (y.size - 1)) if y.size > 1 else 0.5
    return (x[0] - dx, x[-1] + dx, y[0] - dy, y[-1] + dy)


def _as_rows(x, name):
    """Normalise a per-trial data array to ``(N, T)``.

    Accepts the scalar ``(T,)`` form and the multi-dimensional ``(N, T)`` form,
    matching MATLAB's uniform treatment of 1-by-T and N-by-T rows.

    Parameters
    ----------
    x : array_like or None
        The data, or ``None`` to omit the series.
    name : str
        Argument name, quoted in the error message.

    Returns
    -------
    numpy.ndarray or None
        ``(N, T)`` array, or ``None`` when ``x`` was ``None`` or empty.

    Raises
    ------
    ValueError
        If ``x`` has more than two dimensions.
    """
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    raise ValueError("%s must be 1-D (T,) or 2-D (N, T); got shape %r"
                     % (name, arr.shape))


def palette(n=None):
    """Consistent, colourblind-friendly line and patch colours.

    Parameters
    ----------
    n : int or None, optional
        Number of colours wanted. The eight Okabe-Ito base colours cycle when
        ``n`` exceeds eight. ``None`` returns all eight.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` RGB rows in [0, 1] (``(8, 3)`` when ``n`` is ``None``).

    Examples
    --------
    >>> palette(2).shape
    (2, 3)
    """
    if n is None:
        return _BASE.copy()
    n = int(n)
    if n <= 0:
        return _BASE[:0].copy()
    return _BASE[np.arange(n) % _BASE.shape[0], :].copy()


def state_trace(
    t,
    observed=None,
    true_val=None,
    predicted=None,
    block_edges=None,
    *,
    fig_name="State trace",
    dim_label="State dim",
    band=None,
    predicted_name="Predicted",
    observed_name="Observed",
    true_name="True",
):
    """Per-dimension observed / true / predicted trace over trials.

    One stacked subplot per state dimension comparing the noisy observations,
    the true latent target and the model's prediction as a function of trial.
    Handles the scalar ``(T,)`` and the multi-dimensional ``(N, T)`` cases
    uniformly, so the same helper serves both example scripts.

    Parameters
    ----------
    t : array_like
        ``(T,)`` trial indices.
    observed : array_like or None, optional
        ``(N, T)`` noisy observations, drawn as grey ``x`` markers.
    true_val : array_like or None, optional
        ``(N, T)`` true latent targets, drawn as a black dashed line.
    predicted : array_like or None, optional
        ``(N, T)`` model prediction, drawn as a coloured line.
    block_edges : array_like or None, optional
        Trial indices at which the contingency switches, drawn as faint
        vertical guides.
    fig_name : str, optional
        Window-title text.
    dim_label : str, optional
        Stem of the per-dimension y-axis label.
    band : array_like or None, optional
        ``(N, T)`` predictive standard deviation. When supplied a +/-1 sd
        shaded band is drawn around ``predicted``.
    predicted_name, observed_name, true_name : str, optional
        Legend labels for the three series.

    Returns
    -------
    matplotlib.figure.Figure
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    observed = _as_rows(observed, "observed")
    true_val = _as_rows(true_val, "true_val")
    predicted = _as_rows(predicted, "predicted")
    band = _as_rows(band, "band")
    edges = [] if block_edges is None else np.asarray(block_edges, float).reshape(-1)

    n_dim = max([a.shape[0] for a in (observed, true_val, predicted) if a is not None]
                + [1])
    pred_colour = palette(3)[0]

    fig = new_figure(fig_name, figsize=(8.5, 2.6 * n_dim + 0.8))
    axes = fig.subplots(n_dim, 1, squeeze=False, sharex=True)[:, 0]
    for dim in range(n_dim):
        ax = axes[dim]
        # Predictive +/-1 sd band first, so it sits behind the lines.
        if band is not None and predicted is not None:
            ax.fill_between(
                t,
                predicted[dim] - band[dim],
                predicted[dim] + band[dim],
                color=pred_colour,
                alpha=0.15,
                linewidth=0,
            )
        if observed is not None:
            ax.plot(t, observed[dim], "x", color=_OBS_COLOUR, markersize=4,
                    label=observed_name)
        if true_val is not None:
            ax.plot(t, true_val[dim], "k--", linewidth=1.2, label=true_name)
        if predicted is not None:
            ax.plot(t, predicted[dim], "-", color=pred_colour, linewidth=1.4,
                    label=predicted_name)
        for e in edges:
            ax.axvline(e, color=(0.5, 0.5, 0.5), alpha=0.4, linewidth=1.0)
        ax.set_xlabel("Trial")
        ax.set_ylabel("State value" if n_dim == 1 else "%s %d" % (dim_label, dim + 1))
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def density_lines(
    grid,
    mapping,
    *,
    fig_name="Per-context densities",
    title="Per-context densities",
    xlabel="Value",
    novel_density=None,
    ax=None,
):
    """1-D per-context density line plot (scalar model).

    Parameters
    ----------
    grid : array_like
        ``(G,)`` query grid the densities were evaluated on.
    mapping : dict
        ``{global_context_label: (G,) density}``, as returned by
        ``state_given_context_probability`` and friends. Labels are 0-based.
    fig_name, title, xlabel : str, optional
        Figure and axes text.
    novel_density : array_like or None, optional
        ``(G,)`` density of the novel context, drawn dashed in grey.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on; a new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    grid = np.asarray(grid, dtype=float).reshape(-1)
    keys = sorted(mapping)
    cols = palette(max(len(keys), 1) + 1)

    if ax is None:
        fig = new_figure(fig_name)
        ax = fig.subplots()
    else:
        fig = ax.figure

    for i, key in enumerate(keys):
        ax.plot(grid, np.asarray(mapping[key], float).reshape(-1),
                color=cols[i], linewidth=1.4, label="Context %d" % key)
    if novel_density is not None:
        ax.plot(grid, np.asarray(novel_density, float).reshape(-1), "--",
                color=(0.4, 0.4, 0.4), linewidth=1.4, label="Novel context")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def density_evolution(
    trials,
    grid,
    dens,
    *,
    fig_name="Density evolution",
    title="Marginal density over trials",
    ylabel="State value",
    true_line=None,
    pred_line=None,
    missing_trials=None,
    block_edges=None,
    ax=None,
):
    """Trial-by-trial marginal density heat-map.

    A single image panel with trial on the x-axis and the query grid on the
    y-axis, the colour encoding a marginal density recorded once per trial. The
    natural way to watch a posterior evolve - for instance through a block of
    missing (NaN) observations, where the distribution widens while no feedback
    arrives.

    Parameters
    ----------
    trials : array_like
        ``(T,)`` trial indices (x-axis).
    grid : array_like
        ``(G,)`` query grid (y-axis).
    dens : array_like
        ``(G, T)``; column ``t`` is the density over ``grid`` at trial ``t``.
    fig_name, title, ylabel : str, optional
        Figure and axes text.
    true_line : array_like or None, optional
        ``(T,)`` true latent target, overlaid as a black dashed line.
    pred_line : array_like or None, optional
        ``(T,)`` model prediction, overlaid as a white line.
    missing_trials : array_like or None, optional
        Trial indices to mark with faint vertical guides (e.g. NaN feedback).
    block_edges : array_like or None, optional
        Contingency-switch trials, drawn as vertical guides.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on; a new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    trials = np.asarray(trials, dtype=float).reshape(-1)
    grid = np.asarray(grid, dtype=float).reshape(-1)
    dens = np.asarray(dens, dtype=float)

    if ax is None:
        fig = new_figure(fig_name, figsize=(9.0, 4.5))
        ax = fig.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(dens, extent=_pixel_extent(trials, grid), origin="lower",
                   aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax)

    for e in np.asarray([] if missing_trials is None else missing_trials,
                        float).reshape(-1):
        ax.axvline(e, color=(0.85, 0.85, 0.85), alpha=0.6, linestyle=":",
                   linewidth=1.0)
    for e in np.asarray([] if block_edges is None else block_edges,
                        float).reshape(-1):
        ax.axvline(e, color=(0.5, 0.5, 0.5), alpha=0.5, linewidth=1.0)

    if true_line is not None:
        ax.plot(trials, np.asarray(true_line, float).reshape(-1), "k--",
                linewidth=1.3, label="True")
    if pred_line is not None:
        ax.plot(trials, np.asarray(pred_line, float).reshape(-1), "-",
                color="w", linewidth=1.3, label="Predicted")

    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(grid[0], grid[-1])
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def context_bars(
    prev_probs=None,
    post_probs=None,
    *,
    fig_name="Context probabilities",
    mode="area",
    prev_title="Predicted context probabilities (before observation)",
    post_title="Context responsibilities (after observation)",
    novel_context=False,
):
    """Context probabilities over trials.

    Pass the predicted (pre-observation) probabilities and/or the
    responsibilities (post-observation); each is a ``(T, C)`` trials-by-context
    matrix. When both are supplied they are drawn in a stacked pair of
    subplots.

    Trailing all-zero context columns are trimmed so the legend is not
    cluttered by never-instantiated contexts.

    Parameters
    ----------
    prev_probs : array_like or None, optional
        ``(T, C)`` predicted context probabilities.
    post_probs : array_like or None, optional
        ``(T, C)`` context responsibilities.
    fig_name : str, optional
        Window-title text.
    mode : {'area', 'line'}, optional
        Stacked areas (the default) or one line per context.
    prev_title, post_title : str, optional
        Subplot titles.
    novel_context : bool, optional
        When true the final retained column is the novel (not-yet-instantiated)
        slot: it is relabelled "Novel context" and drawn in grey. Only pass true
        when that is actually the case - compare the number of retained columns
        with ``diagnostics()["C"]`` (or ``["K"]``).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If neither matrix is supplied, or ``mode`` is unknown.
    """
    if mode not in ("area", "line"):
        raise ValueError("mode must be 'area' or 'line'; got %r" % (mode,))

    panels, titles, ylabels = [], [], []
    if prev_probs is not None and np.asarray(prev_probs).size:
        panels.append(np.asarray(prev_probs, dtype=float))
        titles.append(prev_title)
        ylabels.append("Predicted prob")
    if post_probs is not None and np.asarray(post_probs).size:
        panels.append(np.asarray(post_probs, dtype=float))
        titles.append(post_title)
        ylabels.append("Responsibility")
    if not panels:
        raise ValueError("Provide prev_probs and/or post_probs.")

    # Trim trailing all-zero context slots, but keep at least one column.
    width = max(p.shape[1] for p in panels)
    used = np.zeros(width, dtype=bool)
    for p in panels:
        used[: p.shape[1]] |= np.any(p > 0, axis=0)
    n_ctx = int(np.max(np.flatnonzero(used)) + 1) if used.any() else 1

    fig = new_figure(fig_name, figsize=(8.5, 3.4 * len(panels) + 0.8))
    axes = fig.subplots(len(panels), 1, squeeze=False)[:, 0]
    for ax, p, title, ylab in zip(axes, panels, titles, ylabels):
        c_k = min(n_ctx, p.shape[1])
        n_trials = p.shape[0]
        x = np.arange(1, n_trials + 1)
        cols = palette(c_k)
        labels = ["Context %d" % c for c in range(c_k)]
        if novel_context and c_k >= 1:
            cols[c_k - 1] = _NOVEL_COLOUR
            labels[c_k - 1] = "Novel context"
        if mode == "area":
            ax.stackplot(x, p[:, :c_k].T, colors=[tuple(c) for c in cols],
                         labels=labels, edgecolor="none")
        else:
            for c in range(c_k):
                ax.plot(x, p[:, c], color=cols[c], linewidth=1.2,
                        label=labels[c])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("Trial")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize="medium")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=min(c_k, 5),
                  fontsize="small", frameon=False)
    fig.tight_layout()
    return fig


def context_density_evolution(
    trials,
    grid,
    ctx_dens,
    *,
    fig_name="State|context evolution",
    title="Per-context posterior state density over trials",
    ylabel="State value",
    novel_dens=None,
    true_line=None,
    block_edges=None,
    ax=None,
):
    """Composite mixed-colour state-given-context evolution heat-map.

    Renders a single image (trial on x, state grid on y) of the per-context
    posterior state density recorded once per trial. Each context is tinted its
    own palette colour and the densities are additively blended into one
    truecolor image, so where contexts overlap their colours mix - a compact way
    to watch which context "owns" which part of state space as learning
    proceeds. Scalar-model only: a multi-dimensional per-trial density is
    already 2-D and cannot be shown as one trial-by-state image.

    Parameters
    ----------
    trials : array_like
        ``(T,)`` trial indices (x-axis).
    grid : array_like
        ``(G,)`` query grid (y-axis).
    ctx_dens : dict
        ``{global_context_label: (G, T) density}``; column ``t`` is that
        context's density at trial ``t``.
    fig_name, title, ylabel : str, optional
        Figure and axes text.
    novel_dens : array_like or None, optional
        ``(G, T)`` novel-context density, tinted grey.
    true_line : array_like or None, optional
        ``(T,)`` true latent target, overlaid as a white dashed line.
    block_edges : array_like or None, optional
        Contingency-switch trials, drawn as vertical guides.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on; a new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If no context density (and no novel density) is supplied.
    """
    trials = np.asarray(trials, dtype=float).reshape(-1)
    grid = np.asarray(grid, dtype=float).reshape(-1)
    keys = sorted(ctx_dens)

    # Tint by GLOBAL label (not enumeration order) so a context keeps its colour
    # across figures, matching context_bars.
    tints = palette(max(keys) + 1 if keys else 1)
    layers = [(np.asarray(ctx_dens[k], float), tints[k], "Context %d" % k)
              for k in keys]
    if novel_dens is not None:
        layers.append((np.asarray(novel_dens, float), np.array(_NOVEL_COLOUR),
                       "Novel context"))
    if not layers:
        raise ValueError("Provide at least one context density.")

    # Joint normalisation so brightness is comparable across contexts, then
    # additive colour mixing (overlaps brighten toward a blended hue).
    peak = max(float(np.max(d)) for d, _, _ in layers)
    if peak <= 0:
        peak = 1.0
    rgb = np.zeros((grid.size, trials.size, 3))
    for d, colour, _ in layers:
        w = np.clip(d, 0.0, None) / peak
        rgb += np.clip(w, 0.0, 1.0)[:, :, None] * np.asarray(colour)[None, None, :]
    rgb = np.clip(rgb, 0.0, 1.0)

    if ax is None:
        fig = new_figure(fig_name, figsize=(9.5, 4.5))
        ax = fig.subplots()
    else:
        fig = ax.figure

    ax.imshow(rgb, extent=_pixel_extent(trials, grid), origin="lower",
              aspect="auto")
    for e in np.asarray([] if block_edges is None else block_edges,
                        float).reshape(-1):
        ax.axvline(e, color="w", alpha=0.4, linewidth=1.0)
    if true_line is not None:
        ax.plot(trials, np.asarray(true_line, float).reshape(-1), "w--",
                linewidth=1.3)

    # An image carries no legend, so add off-screen marker proxies.
    handles = [ax.plot([], [], "s", markersize=9, color=tuple(colour),
                       linestyle="none")[0] for _, colour, _ in layers]
    ax.legend(handles, [lab for _, _, lab in layers], loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize="small", frameon=False)

    ax.set_xlabel("Trial")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(grid[0], grid[-1])
    fig.tight_layout()
    return fig


def trajectory_2d(
    path_xy,
    targets=None,
    *,
    fig_name="2-D trajectory",
    title="2-D trajectory",
    path_name="Path",
    target_name="Targets",
    observed_xy=None,
    ax=None,
):
    """2-D state / motor-output trajectory with target markers.

    Parameters
    ----------
    path_xy : array_like
        ``(2, T)`` path; row 0 is dim 1 and row 1 is dim 2.
    targets : array_like or None, optional
        ``(2, M)`` block-target marker locations.
    fig_name, title, path_name, target_name : str, optional
        Figure, axes and legend text.
    observed_xy : array_like or None, optional
        ``(2, T)`` observed points, overlaid as faint markers.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on; a new figure is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    path_xy = np.asarray(path_xy, dtype=float)
    cols = palette(2)

    if ax is None:
        fig = new_figure(fig_name, figsize=(6.0, 5.5))
        ax = fig.subplots()
    else:
        fig = ax.figure

    if observed_xy is not None:
        obs = np.asarray(observed_xy, dtype=float)
        ax.plot(obs[0], obs[1], "x", color=(0.75, 0.75, 0.75), markersize=4)
    ax.plot(path_xy[0], path_xy[1], "-", color=cols[0], linewidth=1.3,
            label=path_name)
    if targets is not None and np.asarray(targets).size:
        tgt = np.asarray(targets, dtype=float)
        ax.plot(tgt[0], tgt[1], "*", markersize=16, markerfacecolor=cols[1],
                markeredgecolor="k", linestyle="none", label=target_name)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def density_heatmaps(
    coin,
    diag,
    ctx_keys,
    ctx_targets,
    *,
    space="feedback",
    cue_labels=None,
    grid=120,
):
    """2-D per-context density heat-map grid, including the novel context.

    One image panel per instantiated context plus a final panel for the novel
    (not-yet-instantiated) context, for a ``state_dim == 2`` model. One call
    serves both the feedback-space and the state-space figure.

    Parameters
    ----------
    coin : realtimecoin.RealTimeCOIN
        The model, queried on a fresh grid per panel.
    diag : dict
        Its ``diagnostics()`` mapping, for the per-context means and biases.
    ctx_keys : sequence of int
        0-based global context labels to draw.
    ctx_targets : array_like
        ``(2, len(ctx_keys))`` true target of each context's cue.
    space : {'feedback', 'state'}, optional
        ``'feedback'`` draws ``p(y | context)`` = state + bias + noise and marks
        the true target (which lands on the peak); ``'state'`` draws
        ``p(state | context)`` and marks target-minus-bias.
    cue_labels : array_like or None, optional
        Per-global-context dominant cue label, used only in the panel titles.
    grid : int, optional
        Density grid resolution per axis.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``space`` is unknown or the model is not 2-D.
    """
    if space not in ("feedback", "state"):
        raise ValueError("space must be 'feedback' or 'state'; got %r" % (space,))
    if coin.state_dim != 2:
        raise ValueError("density_heatmaps requires a 2-D (state_dim == 2) model.")

    is_fb = space == "feedback"
    if is_fb:
        fig_name = "Per-context feedback densities"
        fig_title = "Per-context predictive feedback densities  p(y | context)"
        axlabel = "Feedback"
    else:
        fig_name = "Per-context state densities"
        fig_title = "Per-context posterior state densities  p(state | context)"
        axlabel = "State"

    ctx_keys = list(ctx_keys)
    ctx_targets = np.asarray(ctx_targets, dtype=float)
    n_inst = len(ctx_keys)
    n_panels = n_inst + 1                      # +1 for the novel context
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig = new_figure(fig_name, figsize=(4.2 * n_cols, 3.8 * n_rows))
    axes = fig.subplots(n_rows, n_cols, squeeze=False).ravel()

    for i, key in enumerate(ctx_keys):
        bias = np.asarray(diag["bias"], float)[key]
        if is_fb:
            mu = np.asarray(diag["state_mean"], float)[key] + bias
            mk = ctx_targets[:, i]                   # target lands on the peak
            mk_name = "true target"
        else:
            mu = np.asarray(diag["state_mean"], float)[key]
            mk = ctx_targets[:, i] - bias            # target - bias does
            mk_name = "true target - bias"

        m = 0.10                                     # margin so blob + marker fit
        ax_x = np.linspace(min(mu[0], mk[0]) - m, max(mu[0], mk[0]) + m, grid)
        ax_y = np.linspace(min(mu[1], mk[1]) - m, max(mu[1], mk[1]) + m, grid)
        gx, gy = np.meshgrid(ax_x, ax_y)
        pts = np.column_stack([gx.ravel(), gy.ravel()])   # one query point per ROW
        dmap = (coin.state_feedback_given_context_probability(pts) if is_fb
                else coin.state_given_context_probability(pts))
        z = np.asarray(dmap[key], float).reshape(gx.shape)

        ax = axes[i]
        im = ax.imshow(z, extent=_pixel_extent(ax_x, ax_y), origin="lower",
                       aspect="auto", cmap="viridis")
        ax.plot(mk[0], mk[1], "*", color="r", markersize=14, linestyle="none",
                label=mk_name)
        ax.plot(mu[0], mu[1], "+", color="w", markersize=10, markeredgewidth=1.5,
                linestyle="none", label="density mean")
        ax.set_xlabel("%s dim 1" % axlabel)
        ax.set_ylabel("%s dim 2" % axlabel)
        fig.colorbar(im, ax=ax)
        if cue_labels is None:
            ax.set_title("Context %d" % key)
        else:
            ax.set_title("Context %d (cue %d)" % (key, np.asarray(cue_labels)[key]))
        ax.legend(loc="best", fontsize="x-small")

    # Novel context: autoscale a grid to where its density is appreciable.
    broad = np.linspace(-1.5, 1.5, grid)
    bx, by = np.meshgrid(broad, broad)
    novel_fn = (coin.novel_state_feedback_probability if is_fb
                else coin.novel_state_probability)
    nd = np.asarray(
        novel_fn(np.column_stack([bx.ravel(), by.ravel()])), float
    ).reshape(bx.shape)
    if nd.max() > 0:
        mask = nd > 0.02 * nd.max()
        xr = broad[np.any(mask, axis=0)]
        yr = broad[np.any(mask, axis=1)]
        pad = 0.1
        ax_x = np.linspace(xr.min() - pad, xr.max() + pad, grid)
        ax_y = np.linspace(yr.min() - pad, yr.max() + pad, grid)
        gx, gy = np.meshgrid(ax_x, ax_y)
        zn = np.asarray(
            novel_fn(np.column_stack([gx.ravel(), gy.ravel()])), float
        ).reshape(gx.shape)
    else:
        ax_x, ax_y, zn = broad, broad, nd

    ax = axes[n_panels - 1]
    im = ax.imshow(zn, extent=_pixel_extent(ax_x, ax_y), origin="lower",
                   aspect="auto", cmap="viridis")
    ax.set_xlabel("%s dim 1" % axlabel)
    ax.set_ylabel("%s dim 2" % axlabel)
    fig.colorbar(im, ax=ax)
    ax.set_title("Novel context")

    for ax in axes[n_panels:]:                 # hide unused grid cells
        ax.set_visible(False)

    fig.suptitle(fig_title)
    fig.tight_layout()
    return fig

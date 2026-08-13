"""Match inferred context labels to known synthetic labels.

Translated from ``validation/validation_best_label_map.m``.

Context labels in a mixture model are exchangeable: the inferred context called
``2`` can represent the synthetic context called ``0`` without changing the
statistical model. Scoring context recovery therefore only makes sense AFTER
finding the relabelling of the inferred labels that maximises agreement with the
true labels; that separates genuine inference failure from harmless label
switching.

The optimal relabelling is a maximum-weight bipartite matching between inferred
labels and target labels ``0 .. K-1``, where the weight of pairing an inferred
label with a target is the number of trials on which that pairing agrees with
the true labels (i.e. the confusion matrix). That is a linear assignment
problem, exact for any ``K``.

Improvement over the MATLAB original
------------------------------------
``validation_best_label_map.m`` uses MATLAB's ``matchpairs`` when available but
falls back to an exhaustive search over all ``K!`` permutations for ``K <= 8``,
and for ``K > 8`` without ``matchpairs`` it merely WARNS and returns the
IDENTITY map - a silently suboptimal answer that understates the relabelled
accuracy (a defect, not a design choice). This translation uses
:func:`scipy.optimize.linear_sum_assignment` (Hungarian / Jonker-Volgenant) for
EVERY ``K``, so the result is exact and polynomial-time throughout; there is no
permutation fallback and no identity fallback, and no ``K`` at which the answer
degrades. ``scipy`` is already a hard dependency of ``realtimecoin``, so this
costs nothing.

Labels are 0-BASED here (as everywhere in this package), so the target labels
are ``0 .. K-1`` rather than MATLAB's ``1 .. K``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ["best_label_map"]


def best_label_map(true_labels, inferred_labels):
    """Relabel inferred contexts to best match known true contexts.

    Parameters
    ----------
    true_labels : array_like
        ``(T,)`` known (synthetic) 0-based context labels. Non-finite entries
        are ignored. Target columns are ``0 .. K-1``, so a true label outside
        that range gets an all-zero column and can never be matched - the same
        assumption ``validation_best_label_map.m`` makes with its ``g = 1:K``
        loop.
    inferred_labels : array_like
        ``(T,)`` inferred 0-based context labels, same length as
        ``true_labels``. Non-finite entries are ignored when scoring and pass
        through ``mapped`` unchanged.

    Returns
    -------
    mapped : numpy.ndarray
        ``(T,)`` float array of relabelled inferred contexts.
    accuracy : float
        Fraction of valid trials on which ``mapped`` equals ``true_labels``
        under the optimal relabelling; ``nan`` when there is nothing to score.
    mapping : dict
        ``{inferred label: target label}``, one entry per distinct inferred
        label. The mapping is injective (a partial permutation), so two inferred
        contexts are never merged.

    Notes
    -----
    ``K = max(#distinct true labels, #distinct inferred labels)`` and the
    confusion matrix has one row per distinct inferred label, so the number of
    rows never exceeds the number of target columns and the assignment always
    matches EVERY inferred label. MATLAB needs an extra "leftover target"
    fixup after ``matchpairs`` for exactly this reason; here it is unnecessary.

    Examples
    --------
    >>> mapped, accuracy, mapping = best_label_map([0, 0, 1, 1], [1, 1, 0, 0])
    >>> accuracy
    1.0
    >>> mapping == {0: 1, 1: 0}
    True
    """
    true_array = np.asarray(true_labels, dtype=float).reshape(-1)
    inferred_array = np.asarray(inferred_labels, dtype=float).reshape(-1)

    valid = np.isfinite(true_array) & np.isfinite(inferred_array)
    true_valid = true_array[valid].astype(int)
    inferred_valid = inferred_array[valid].astype(int)

    unique_true = np.unique(true_valid)
    unique_inferred = np.unique(inferred_valid)
    k = int(max(unique_true.size, unique_inferred.size))
    if k == 0:
        # Nothing scorable: hand back the input untouched, as the .m does.
        return inferred_array.copy(), float("nan"), {}

    # agreement[i, g] = number of trials where inferred label unique_inferred[i]
    # coincides with true label g, for target labels g = 0 .. K-1.
    agreement = np.zeros((unique_inferred.size, k))
    for i, label in enumerate(unique_inferred):
        inferred_mask = inferred_valid == label
        for g in range(k):
            agreement[i, g] = np.count_nonzero(inferred_mask & (true_valid == g))

    # Maximise total agreement == minimise the negated confusion matrix. Rows
    # (inferred labels) never outnumber columns (targets), so every row is
    # matched and the result is a valid injection into 0 .. K-1.
    rows, cols = linear_sum_assignment(-agreement)

    mapped = inferred_array.copy()
    mapping = {}
    for i, target in zip(rows, cols):
        source = int(unique_inferred[i])
        mapping[source] = int(target)
        mapped[inferred_array == source] = float(target)

    matched = float(agreement[rows, cols].sum())
    accuracy = matched / true_valid.size if true_valid.size else float("nan")
    return mapped, accuracy, mapping

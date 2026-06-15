"""
Shape-preserving blending basis functions.

This module implements paper-faithful SBS basis functions using smooth
piecewise-polynomial step functions. The blend weights are assembled directly
from a telescoping family of step differences, so the SBS path stays entirely
polynomial and non-rational.
"""

from __future__ import annotations

from math import comb, factorial

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Smooth piecewise-polynomial steps
# ---------------------------------------------------------------------------

def recursive_smooth_step(x: ArrayLike, order: int = 2) -> np.ndarray:
    """
    Recursive piecewise-polynomial smooth step in [0, 1].

    .. math::
       T_n(x) = \\frac{1}{(n+1)!}
       \\sum_{j=0}^{n+1}(-1)^j\\binom{n+1}{j}\\max((n+1)x-j,0)^{n+1}
    """
    n = int(order)
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    t = (n + 1) * x
    prefactor = 1.0 / factorial(n + 1)
    result = np.zeros_like(t)
    for j in range(n + 2):
        result += ((-1) ** j) * comb(n + 1, j) * np.maximum(t - j, 0.0) ** (n + 1)
    return np.clip(result * prefactor, 0.0, 1.0)


def smooth_step_at(
    t: ArrayLike,
    centre: float,
    half_width: float,
    order: int = 2,
    rising: bool = True,
) -> np.ndarray:
    """
    Centred smooth step over [centre-half_width, centre+half_width].
    """
    t = np.asarray(t, dtype=float)
    half_width = float(half_width)
    if half_width <= 0:
        raise ValueError("half_width must be positive.")
    u = (t - centre) / half_width
    x = np.clip(0.5 * (u + 1.0), 0.0, 1.0)
    s = recursive_smooth_step(x, order=order)
    return s if rising else (1.0 - s)


def sbs_basis(
    t: ArrayLike,
    a: float,
    b: float,
    half_width: float | None = None,
    order: int = 2,
) -> np.ndarray:
    """
    SBS basis over interval [a, b] using two smooth step functions.
    """
    if not b > a:
        raise ValueError("Expected interval endpoints with b > a.")
    if half_width is None:
        half_width = (b - a) / 2.0
    S_a = smooth_step_at(t, a, half_width, order=order, rising=False)
    S_b = smooth_step_at(t, b, half_width, order=order, rising=False)
    return np.clip(S_b - S_a, 0.0, None)


# ---------------------------------------------------------------------------
# Partition-of-unity blend weights
# ---------------------------------------------------------------------------

def _step_partition(
    t: np.ndarray,
    centers: np.ndarray,
    *,
    half_width: float,
    order: int,
) -> np.ndarray:
    """Direct polynomial partition built from neighbouring smooth steps."""
    k = len(centers)
    if k == 1:
        return np.ones((1, len(t)), dtype=float)

    boundaries = 0.5 * (centers[:-1] + centers[1:])
    steps = np.array(
        [
            smooth_step_at(t, boundary, half_width, order=order, rising=True)
            for boundary in boundaries
        ]
    )

    W = np.zeros((k, len(t)), dtype=float)
    W[0] = 1.0 - steps[0]
    for j in range(1, k - 1):
        W[j] = steps[j - 1] - steps[j]
    W[-1] = steps[-1]
    return np.clip(W, 0.0, 1.0)


def blend_weights(
    t: ArrayLike,
    centers: ArrayLike,
    locality: float = 1.0,
    width: float | None = None,
    *,
    periodic: bool = False,
    period: float = 1.0,
    order: int = 2,
) -> np.ndarray:
    """
    Compute direct non-rational SBS weights for all shape centres.

    For centres :math:`t_0 < t_1 < \\dots < t_{k-1}`, define midpoint
    boundaries and a rising smooth step at each boundary. The SBS weights are
    then assembled by a telescoping step-difference partition:

    .. math::
       W_0(t) = 1 - U_1(t), \\qquad
       W_j(t) = U_j(t) - U_{j+1}(t), \\qquad
       W_{k-1}(t) = U_{k-1}(t).

    The weights therefore satisfy :math:`\\sum_j W_j(t) = 1` identically
    without any rational normalisation.

    Parameters
    ----------
    t:
        Global curve parameter values, array of shape *(m,)*.
    centers:
        Centre parameter for each shape, array of shape *(k,)*.
    locality:
        Locality parameter α. Larger values narrow transition bands and
        increase locality.
    width:
        Optional base transition half-width before locality scaling. If
        omitted, half of the minimum inter-centre spacing is used so the step
        family remains ordered and the weights stay non-negative.
    periodic:
        If True, treat the parameter domain as periodic with wrap-around at
        ``period``. This is appropriate for closed curves.
    period:
        Period length used when ``periodic=True``.
    order:
        Smooth-step polynomial order used in the SBS basis.

    Returns
    -------
    np.ndarray
        Weight array of shape *(k, m)*. Each column sums to 1.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    centers = np.asarray(centers, dtype=float)
    k = len(centers)
    if k == 0:
        raise ValueError("At least one centre is required.")
    if k == 1:
        return np.ones((1, len(t)), dtype=float)
    if np.any(np.diff(centers) <= 0):
        raise ValueError("centers must be strictly increasing.")

    locality = float(locality)
    order = int(order)
    if locality < 0:
        raise ValueError("locality must be non-negative.")
    if locality == 0:
        return np.full((k, len(t)), 1.0 / k, dtype=float)

    if periodic:
        period = float(period)
        if period <= 0:
            raise ValueError("period must be positive when periodic=True.")
        centers = np.mod(centers, period)
        if np.any(np.diff(centers) <= 0):
            raise ValueError(
                "centers must be strictly increasing within one period when periodic=True."
            )

        wrapped_gaps = np.diff(np.concatenate([centers, centers[:1] + period]))
        min_gap = wrapped_gaps.min()
        base_half_width = 0.5 * min_gap if width is None else float(width)
        half_width = max(base_half_width / locality, 1e-12)

        extended = np.concatenate([centers - period, centers, centers + period])
        W_ext = _step_partition(
            np.mod(t, period),
            extended,
            half_width=half_width,
            order=order,
        )
        W = np.zeros((k, len(t)), dtype=float)
        for block in range(3):
            start = block * k
            W += W_ext[start:start + k]
        return W

    min_gap = np.diff(centers).min()
    base_half_width = 0.5 * min_gap if width is None else float(width)
    half_width = max(base_half_width / locality, 1e-12)
    return _step_partition(
        t,
        centers,
        half_width=half_width,
        order=order,
    )


# ---------------------------------------------------------------------------
# Legacy per-knot weighting hook
# ---------------------------------------------------------------------------

def apply_knot_weights(W: np.ndarray, knot_weights: ArrayLike) -> np.ndarray:
    """
    Per-knot renormalisation has been removed from the SBS path.
    """
    raise NotImplementedError(
        "Per-knot weight renormalisation has been removed: SBS evaluation is "
        "now strictly non-rational."
    )


# ---------------------------------------------------------------------------
# Standard cubic B-spline basis (for comparison / educational use)
# ---------------------------------------------------------------------------

def bspline_basis(i: int, p: int, t: ArrayLike, knots: ArrayLike) -> np.ndarray:
    """
    B-spline basis function N_{i,p}(t) via the Cox–de Boor recursion.

    Parameters
    ----------
    i:
        Basis function index.
    p:
        Polynomial degree (e.g. 3 for cubic).
    t:
        Parameter values, array of shape *(m,)*.
    knots:
        Knot vector, non-decreasing array of length n + p + 1 for n control
        points.

    Returns
    -------
    np.ndarray
        Basis function values of shape *(m,)*.
    """
    t = np.asarray(t, dtype=float)
    knots = np.asarray(knots, dtype=float)

    if p == 0:
        result = np.where(
            (t >= knots[i]) & (t < knots[i + 1]), 1.0, 0.0
        )
        # The rightmost non-zero basis function includes the right endpoint
        # (closed at t = knots[-1]) so the B-spline covers the full domain.
        if knots[i] < knots[i + 1]:  # non-empty span only
            result = np.where(
                (t == knots[-1]) & (knots[i + 1] == knots[-1]),
                1.0,
                result,
            )
        return result

    denom_left = knots[i + p] - knots[i]
    denom_right = knots[i + p + 1] - knots[i + 1]

    left = (
        (t - knots[i]) / denom_left * bspline_basis(i, p - 1, t, knots)
        if denom_left > 0
        else np.zeros_like(t)
    )
    right = (
        (knots[i + p + 1] - t) / denom_right * bspline_basis(i + 1, p - 1, t, knots)
        if denom_right > 0
        else np.zeros_like(t)
    )
    return left + right


def uniform_bspline_weights(t: ArrayLike, n: int, degree: int = 3) -> np.ndarray:
    """
    Evaluate all B-spline basis functions on a uniform clamped knot vector.

    Parameters
    ----------
    t:
        Parameter values in [0, 1], array of shape *(m,)*.
    n:
        Number of control points.
    degree:
        Polynomial degree (default 3 = cubic).

    Returns
    -------
    np.ndarray
        Weight array of shape *(n, m)*, columns summing to 1.
    """
    t = np.asarray(t, dtype=float)
    # Clamped uniform knot vector
    inner_knots = np.linspace(0.0, 1.0, n - degree + 1)
    knots = np.concatenate([
        np.zeros(degree),
        inner_knots,
        np.ones(degree),
    ])
    W = np.array([bspline_basis(i, degree, t, knots) for i in range(n)])
    # Enforce the standard clamped-B-spline convention at the right endpoint:
    # N_{n-1,p}(1) = 1, all others = 0.
    end_mask = t == 1.0
    if np.any(end_mask):
        W[:, end_mask] = 0.0
        W[-1, end_mask] = 1.0
    # Normalise to neutralise any residual floating-point error
    total = W.sum(axis=0, keepdims=True)
    total = np.where(total < 1e-14, 1.0, total)
    return W / total

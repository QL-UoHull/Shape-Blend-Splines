"""
Shape-preserving blending basis functions.

This module implements paper-faithful SBS basis functions using smooth
piecewise-polynomial step functions:

.. math::
   B_{a,b}(t) = S_b(t) - S_a(t)

where :math:`S_a` and :math:`S_b` are centred smooth step functions.
The construction uses polynomial arithmetic only (no trigonometric or
rational basis kernels).
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

def _circular_midpoint(a: float, b: float, period: float) -> float:
    """Midpoint on a periodic domain, returned modulo *period*."""
    a = float(a)
    b = float(b)
    if b < a:
        b += period
    return (0.5 * (a + b)) % period


def _circular_distance(values: np.ndarray, centers: np.ndarray, period: float) -> np.ndarray:
    """Pairwise circular distances for periodic nearest-centre fallback."""
    delta = np.abs(values[:, np.newaxis] - centers[np.newaxis, :])
    return np.minimum(delta, period - delta)


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
    Compute SBS partition-of-unity weights for all shape centres.

    For centres :math:`t_0 < t_1 < \\dots < t_{k-1}`, define midpoint bounds
    :math:`t_{j\\pm 1/2}` and unnormalised weights:

    .. math::
       w_j(t) = B_{t_{j-1/2}, t_{j+1/2}}(t)

    Then:

    .. math::
       W_j(t) = \\frac{w_j(t)}{\\sum_l w_l(t)}

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
        Optional base transition half-width for each endpoint step.
        If omitted, each interval uses half of its midpoint span.
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
        Weight array of shape *(k, m)*.  Each column sums to 1.

    Examples
    --------
    >>> import numpy as np
    >>> from shape_blend_splines.basis import blend_weights
    >>> t = np.linspace(0, 1, 5)
    >>> W = blend_weights(t, centers=[0.0, 0.5, 1.0], locality=1.0)
    >>> np.allclose(W.sum(axis=0), 1.0)
    True
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

    raw = np.zeros((k, len(t)), dtype=float)
    if periodic:
        period = float(period)
        if period <= 0:
            raise ValueError("period must be positive when periodic=True.")
        t_periodic = np.mod(t, period)
        centers_periodic = np.mod(centers, period)
        if np.any(np.diff(centers_periodic) <= 0):
            raise ValueError(
                "centers must be strictly increasing within one period when periodic=True."
            )

        for j in range(k):
            c_prev = centers_periodic[(j - 1) % k]
            c_curr = centers_periodic[j]
            c_next = centers_periodic[(j + 1) % k]

            left = _circular_midpoint(c_prev, c_curr, period=period)
            right = _circular_midpoint(c_curr, c_next, period=period)

            t_local = t_periodic.copy()
            if right <= left:
                right += period
                t_local = np.where(t_local < left, t_local + period, t_local)

            if width is None:
                base_half_width = 0.5 * (right - left)
            else:
                base_half_width = float(width)
            half_width = max(base_half_width / locality, 1e-12)
            raw[j] = sbs_basis(t_local, left, right, half_width=half_width, order=order)
    else:
        midpoints = 0.5 * (centers[:-1] + centers[1:])
        bounds = np.empty(k + 1, dtype=float)
        bounds[1:-1] = midpoints
        bounds[0] = centers[0] - (midpoints[0] - centers[0])
        bounds[-1] = centers[-1] + (centers[-1] - midpoints[-1])

        for j in range(k):
            a = bounds[j]
            b = bounds[j + 1]
            if width is None:
                base_half_width = 0.5 * (b - a)
            else:
                base_half_width = float(width)
            half_width = max(base_half_width / locality, 1e-12)
            raw[j] = sbs_basis(t, a, b, half_width=half_width, order=order)

    # Normalise column-wise (partition of unity).
    # If ALL kernels vanish at a parameter value (can happen at domain
    # boundaries when a centre coincides with an endpoint), fall back to
    # assigning unit weight to the nearest centre so the blended curve
    # still evaluates to a meaningful value.
    total = raw.sum(axis=0)                                   # (m,)
    zero_mask = total < 1e-14
    if np.any(zero_mask):
        t_zero = t[zero_mask]
        if periodic:
            dist = _circular_distance(np.mod(t_zero, period), centers_periodic, period)
        else:
            dist = np.abs(t_zero[:, np.newaxis] - centers[np.newaxis, :])  # (n0, k)
        nearest = np.argmin(dist, axis=1)                              # (n0,)
        for pos, nidx in zip(np.where(zero_mask)[0], nearest):
            raw[nidx, pos] = 1.0
        total = raw.sum(axis=0)
        total = np.where(total < 1e-14, 1.0, total)

    return raw / total


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

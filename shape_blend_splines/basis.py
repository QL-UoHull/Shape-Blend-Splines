"""
Partial Shape-Preserving (PSP) basis functions.

This module implements the paper-faithful PSP basis functions following:

    Q. Li, J. Tian, "Partial shape-preserving splines",
    Computer-Aided Design 43 (2011) 394-409.

The central construction (Eq. 17): the PSP basis on interval [a, b] is
the *difference of two smooth unit step functions*:

    B^{(n)}_{[a,b],delta}(x) = H_{n,delta}(x-a) - H_{n,delta}(x-b)

where H_{n,delta}(x) = H_n(n*x/delta) is the scaled smooth unit step (Eq. 11)
and H_n is defined by the closed-form expression (Eq. 6).

The flat-top [a+delta, b-delta] is where B = 1 exactly -- the shape-
preservation interval. When b-a < 2*delta the flat-top is empty (bump shape).

Backward compatibility
----------------------
The old ``recursive_smooth_step`` / ``blend_weights`` / ``sbs_basis`` API is
preserved but deprecated.  The old ``locality`` parameter maps to
``delta = (interval_half_width) / locality``.
"""

from __future__ import annotations

import warnings
from math import comb, factorial

import numpy as np
from numpy.typing import ArrayLike

# kept for backward compatibility
CUBIC_C2_ORDER = 2


# ---------------------------------------------------------------------------
# Smooth unit step function H_n  (Section 4, pages 395-396)
# ---------------------------------------------------------------------------

def heaviside_step(x: ArrayLike) -> np.ndarray:
    """
    Heaviside step function H_0 (Eq. 1).

    H_0(x) = 0 for x < 0,  1/2 for x = 0,  1 for x > 0.
    """
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 1.0, np.where(x == 0, 0.5, 0.0))


def smooth_unit_step(x: ArrayLike, n: int) -> np.ndarray:
    """
    Smooth unit step H_n(x) (Eq. 6, closed form).

    .. math::
       H_n(x) = \\frac{1}{n!\\,2^n}
         \\sum_{k=0}^{n} (-1)^k \\binom{n}{k}
         (x + n - 2k)^n H_0(x + n - 2k)

    Properties (Prop. 4.1):
      - C^{n-1} smooth for n >= 1; degree-n piecewise polynomial.
      - Monotone increasing.
      - H_n(x) = 1 for x >= n,  = 0 for x <= -n.
      - H_n(0) = 1/2.
      - H_n(-x) = 1 - H_n(x)  (antisymmetry).

    Explicit forms (Eqs. 7-10)::

        H_1(x) = 1/2 * ((x+1) H_0(x+1) - (x-1) H_0(x-1))
        H_2(x) = 1/8 * ((x+2)^2 H_0(x+2) - 2x^2 H_0(x) + (x-2)^2 H_0(x-2))
        H_3(x) = 1/48 * ((x+3)^3 H_0(x+3) - 3(x+1)^3 H_0(x+1)
                          + 3(x-1)^3 H_0(x-1) - (x-3)^3 H_0(x-3))

    Parameters
    ----------
    x : array-like
        Evaluation points.
    n : int
        Degree >= 0. n=0 returns the Heaviside step.

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x, dtype=float)
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n == 0:
        return heaviside_step(x)

    # Early exit for x outside the transition zone [-n, n] avoids catastrophic
    # cancellation in the summation (large terms with alternating signs).
    result = np.where(x >= n, 1.0, np.where(x <= -n, 0.0, np.nan))
    mask = np.isnan(result)  # only compute for x in (-n, n)
    if np.any(mask):
        xm = x[mask]
        prefactor = 1.0 / (factorial(n) * (2 ** n))
        part = np.zeros_like(xm)
        for k in range(n + 1):
            shift = xm + float(n - 2 * k)
            coeff = ((-1) ** k) * comb(n, k)
            # Only H_0(shift)=1 when shift>0, =0.5 at 0 (measure zero), =0 else
            part = part + coeff * (shift ** n) * heaviside_step(shift)
        result = result.copy()
        result[mask] = np.clip(part * prefactor, 0.0, 1.0)
    return result


def smooth_unit_step_delta(x: ArrayLike, n: int, delta: float) -> np.ndarray:
    """
    Scaled smooth unit step H_{n,delta}(x) = H_n(n*x/delta)  (Eq. 11).

    delta is the *rising/blending range*:
      - H_{n,delta}(x) = 1 for x >= delta
      - H_{n,delta}(x) = 0 for x <= -delta
      - H_{n,delta}(0) = 1/2

    A smaller delta gives a narrower transition (wider flat-top on the basis
    function); a larger delta gives a wider transition (bump-shaped basis).

    Parameters
    ----------
    x : array-like
    n : int
        Degree.
    delta : float
        Rising/blending range (> 0).

    Returns
    -------
    np.ndarray
    """
    delta = float(delta)
    if delta <= 0:
        raise ValueError("delta must be positive.")
    x = np.asarray(x, dtype=float)
    return smooth_unit_step(n * x / delta, n)


def smooth_unit_step_deriv(x: ArrayLike, n: int, i: int) -> np.ndarray:
    """
    i-th derivative of H_n(x)  (Eqs. 12-13).

    .. math::
       H_n^{(i)}(x) = \\frac{1}{(n-i)!\\,2^n}
         \\sum_{k=0}^{n}(-1)^k\\binom{n}{k}
         (x+n-2k)^{n-i} H_0(x+n-2k)

    Parameters
    ----------
    x : array-like
    n : int
        Degree.
    i : int
        Derivative order, 0 <= i < n.

    Returns
    -------
    np.ndarray
    """
    x = np.asarray(x, dtype=float)
    n, i = int(n), int(i)
    if not (0 <= i < n):
        raise ValueError(f"Derivative order i={i} must satisfy 0 <= i < n={n}.")

    prefactor = 1.0 / (factorial(n - i) * (2 ** n))
    result = np.zeros_like(x)
    for k in range(n + 1):
        shift = x + float(n - 2 * k)
        coeff = ((-1) ** k) * comb(n, k)
        result = result + coeff * (shift ** (n - i)) * heaviside_step(shift)
    return result * prefactor


# ---------------------------------------------------------------------------
# PSP basis function (Section 5, Eq. 17)
# ---------------------------------------------------------------------------

def psp_basis(
    x: ArrayLike, a: float, b: float, n: int, delta: float
) -> np.ndarray:
    """
    PSP basis function B^{(n)}_{[a,b],delta}(x)  (Eq. 17).

    .. math::
       B^{(n)}_{[a,b],\\delta}(x) = H_{n,\\delta}(x-a) - H_{n,\\delta}(x-b)

    Properties:
      - Non-negative: 0 <= B <= 1.
      - C^{n-1} smooth.
      - **Flat-top**: B = 1 exactly on [a+delta, b-delta]  (shape preservation).
        Flat-top is empty (bump shape) when b - a < 2*delta.
      - Additivity: B_{[a,c]}(x) + B_{[c,b]}(x) = B_{[a,b]}(x).

    Parameters
    ----------
    x : array-like
    a, b : float
        Interval endpoints (a <= b).
    n : int
        Degree.
    delta : float
        Rising/blending range.

    Returns
    -------
    np.ndarray
    """
    a, b = float(a), float(b)
    if b < a:
        raise ValueError(f"Expected a <= b, got a={a}, b={b}.")
    x = np.asarray(x, dtype=float)
    ha = smooth_unit_step_delta(x - a, n, delta)
    hb = smooth_unit_step_delta(x - b, n, delta)
    return np.clip(ha - hb, 0.0, 1.0)


def psp_basis_asymmetric(
    x: ArrayLike,
    a: float,
    b: float,
    n: int,
    delta_a: float,
    delta_b: float,
    warn: bool = True,
) -> np.ndarray:
    """
    Asymmetric PSP basis (Eq. 19).

    .. math::
       B = H_{n,\\delta_a}(x-a) - H_{n,\\delta_b}(x-b)

    May go slightly negative unless the non-negativity condition holds:
        0 <= (b - a - delta_b) <= delta_a

    Parameters
    ----------
    warn : bool
        Emit a UserWarning when the non-negativity condition is not met.

    Returns
    -------
    np.ndarray
    """
    a, b = float(a), float(b)
    delta_a, delta_b = float(delta_a), float(delta_b)
    condition = 0.0 <= (b - a - delta_b) <= delta_a
    if warn and not condition:
        warnings.warn(
            f"Asymmetric PSP basis: non-negativity condition not satisfied "
            f"(a={a}, b={b}, delta_a={delta_a}, delta_b={delta_b}). "
            "Basis may go slightly negative.",
            UserWarning,
            stacklevel=2,
        )
    x = np.asarray(x, dtype=float)
    ha = smooth_unit_step_delta(x - a, n, delta_a)
    hb = smooth_unit_step_delta(x - b, n, delta_b)
    return ha - hb


# ---------------------------------------------------------------------------
# Partition of unity (Section 5, Eq. 18)
# ---------------------------------------------------------------------------

def psp_partition(
    x: ArrayLike,
    knots: ArrayLike,
    n: int,
    delta: float,
    periodic: bool = False,
    period: float | None = None,
) -> np.ndarray:
    """
    PSP partition-of-unity basis matrix (Eq. 18).

    For m+1 knots t_0 <= ... <= t_m, the i-th basis function is::

        B_i(x) = B^{(n)}_{[t_i, t_{i+1}], delta}(x),   i = 0, ..., m-1

    The columns sum to::

        sum_i B_i(x) = H_{n,delta}(x - t_0) - H_{n,delta}(x - t_m)

    which equals 1 for x in [t_0 + delta, t_m - delta]  (design domain).

    Parameters
    ----------
    x : array-like
        Evaluation points.
    knots : array-like
        Knot vector of length m+1 (gives m basis functions).
    n : int
        Degree.
    delta : float
        Rising/blending range.
    periodic : bool
        If True, wrap x modulo the period (= knots[-1] - knots[0] if not
        supplied explicitly).
    period : float, optional
        Period length (only used when periodic=True).

    Returns
    -------
    np.ndarray, shape (m, len(x))
        B[i, j] = value of i-th basis function at x[j].
        Columns sum to 1 within the design domain.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    knots = np.asarray(knots, dtype=float)
    m = len(knots) - 1
    if m < 1:
        raise ValueError("Need at least 2 knots (for at least 1 basis function).")

    if periodic:
        if period is None:
            period = float(knots[-1] - knots[0])
        # Wrap x into [knots[0], knots[0]+period)
        x = np.mod(x - knots[0], period) + knots[0]

    B = np.zeros((m, len(x)))
    for i in range(m):
        B[i] = psp_basis(x, knots[i], knots[i + 1], n, delta)
    return B


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def shape_preserving_interval(a: float, b: float, delta: float) -> tuple[float, float]:
    """
    Return the shape-preserving (flat-top) interval [a+delta, b-delta].

    This is where B^{(n)}_{[a,b],delta} = 1 exactly.
    If b - a < 2*delta the flat-top is degenerate (left > right means empty).

    Returns
    -------
    tuple (left, right)
    """
    return (float(a) + delta, float(b) - delta)


def knots_from_weights(weights: ArrayLike, a0: float = 0.0) -> np.ndarray:
    """
    Build a knot vector from interval widths (weights as knot spacings).

    w_i = a_{i+1} - a_i  (Eq. 20 of Li & Tian 2011).

    A larger weight gives a wider interval and hence stronger pull toward
    the corresponding control point -- equivalent to a NURBS weight but
    without any rational denominator.

    Parameters
    ----------
    weights : array-like
        Non-negative interval widths [w_0, w_1, ..., w_{n}].
    a0 : float
        Starting knot value (default 0).

    Returns
    -------
    np.ndarray, shape (len(weights)+1,)
        Knot vector [a_0, a_1, ..., a_{n+1}].
    """
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("All weights (interval widths) must be non-negative.")
    return np.concatenate([[float(a0)], float(a0) + np.cumsum(weights)])


def interpolated_indices(knots: ArrayLike, delta: float) -> list[int]:
    """
    Indices of basis functions with a non-empty flat-top (selective interpolation).

    Basis function i has a non-empty flat-top iff:
        knots[i+1] - knots[i] >= 2 * delta

    When the flat-top is non-empty, the corresponding control point is
    reproduced *exactly* by the PSP curve (selective interpolation, Fig. 11).

    Parameters
    ----------
    knots : array-like
        Knot vector of length n+2.
    delta : float
        Rising/blending range.

    Returns
    -------
    list of int
    """
    knots = np.asarray(knots, dtype=float)
    intervals = np.diff(knots)
    return [int(i) for i, w in enumerate(intervals) if w >= 2.0 * delta]


# ---------------------------------------------------------------------------
# B-spline special case (page 398 of Li & Tian 2011)
# ---------------------------------------------------------------------------

def bspline_basis(i: int, p: int, t: ArrayLike, knots: ArrayLike) -> np.ndarray:
    """
    Standard B-spline basis function N_{i,p}(t) via Cox-de Boor recursion.

    This is kept for the *B-spline special case* demonstration:
    when knots are equally spaced with unit spacing, the degree-p B-spline
    basis function equals the PSP basis::

        N_{i,p}(t) == psp_basis(t, a_i, a_{i+1}, n=p, delta=p/2)

    with a_i = i + p/2, a_{i+1} = a_i + 1  (page 398 of Li & Tian 2011).

    Parameters
    ----------
    i : int
        Basis function index.
    p : int
        Polynomial degree.
    t : array-like
        Parameter values.
    knots : array-like
        Non-decreasing knot vector.

    Returns
    -------
    np.ndarray
    """
    t = np.asarray(t, dtype=float)
    knots = np.asarray(knots, dtype=float)

    if p == 0:
        result = np.where(
            (t >= knots[i]) & (t < knots[i + 1]), 1.0, 0.0
        )
        if knots[i] < knots[i + 1]:
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
    All B-spline basis functions on a uniform clamped knot vector.

    Provided for the B-spline special case comparison. For uniform knots the
    degree-p B-spline equals the PSP basis; for non-equal knots they differ.

    Returns
    -------
    np.ndarray, shape (n, len(t))
    """
    t = np.asarray(t, dtype=float)
    inner_knots = np.linspace(0.0, 1.0, n - degree + 1)
    knots = np.concatenate([
        np.zeros(degree),
        inner_knots,
        np.ones(degree),
    ])
    W = np.array([bspline_basis(i, degree, t, knots) for i in range(n)])
    end_mask = t == 1.0
    if np.any(end_mask):
        W[:, end_mask] = 0.0
        W[-1, end_mask] = 1.0
    total = W.sum(axis=0, keepdims=True)
    total = np.where(total < 1e-14, 1.0, total)
    return W / total


# ---------------------------------------------------------------------------
# Backward-compatible deprecated API
# ---------------------------------------------------------------------------

def recursive_smooth_step(x: ArrayLike, order: int = CUBIC_C2_ORDER) -> np.ndarray:
    """
    Deprecated. Use :func:`smooth_unit_step` instead.

    This was an ad-hoc smoothstep on [0,1]; it is *not* H_n from the paper.
    Retained for backward compatibility only.
    """
    warnings.warn(
        "recursive_smooth_step() is deprecated and does not implement H_n from "
        "Li & Tian (2011). Use smooth_unit_step(x, n) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    n = int(order)
    if n < 0:
        raise ValueError("order must be non-negative.")
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
    order: int = CUBIC_C2_ORDER,
    rising: bool = True,
) -> np.ndarray:
    """Deprecated. Centred smooth step (old SBS API)."""
    warnings.warn(
        "smooth_step_at() is deprecated (old SBS API). "
        "Use smooth_unit_step_delta() and psp_basis() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    t = np.asarray(t, dtype=float)
    u = (t - centre) / float(half_width)
    x = np.clip(0.5 * (u + 1.0), 0.0, 1.0)
    s = recursive_smooth_step.__wrapped__(x, order=order)
    return s if rising else (1.0 - s)


def _recursive_smooth_step_unwrapped(x: ArrayLike, order: int = CUBIC_C2_ORDER) -> np.ndarray:
    """Internal, non-deprecated version for backward-compat helpers."""
    n = int(order)
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    t = (n + 1) * x
    prefactor = 1.0 / factorial(n + 1)
    result = np.zeros_like(t)
    for j in range(n + 2):
        result += ((-1) ** j) * comb(n + 1, j) * np.maximum(t - j, 0.0) ** (n + 1)
    return np.clip(result * prefactor, 0.0, 1.0)


smooth_step_at.__wrapped__ = smooth_step_at  # sentinel


def sbs_basis(
    t: ArrayLike,
    a: float,
    b: float,
    half_width: float | None = None,
    order: int = CUBIC_C2_ORDER,
) -> np.ndarray:
    """Deprecated. Use :func:`psp_basis` instead."""
    warnings.warn(
        "sbs_basis() is deprecated. Use psp_basis(x, a, b, n, delta) instead. "
        "The delta parameter corresponds to the rising/blending range from the paper.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not b > a:
        raise ValueError("Expected b > a.")
    if half_width is None:
        half_width = (b - a) / 2.0
    t = np.asarray(t, dtype=float)

    def _ss(tv, centre, hw, rising=True):
        u = (tv - centre) / hw
        x = np.clip(0.5 * (u + 1.0), 0.0, 1.0)
        s = _recursive_smooth_step_unwrapped(x, order=order)
        return s if rising else 1.0 - s

    S_a = _ss(t, a, half_width, rising=False)
    S_b = _ss(t, b, half_width, rising=False)
    return np.clip(S_b - S_a, 0.0, None)


def blend_weights(
    t: ArrayLike,
    centers: ArrayLike,
    locality: float = 1.0,
    width: float | None = None,
    *,
    periodic: bool = False,
    period: float = 1.0,
    order: int = CUBIC_C2_ORDER,
) -> np.ndarray:
    """
    Deprecated. Old SBS blend weights.

    This is the midpoint-boundary telescoping construction from the old SBS
    code. It is **not** the paper's PSP partition (Eq. 18). Use
    :func:`psp_partition` for the paper-faithful basis.

    Retained for backward compatibility with existing code using
    ``ShapeBlendSpline`` / ``PeriodicShapeBlendSpline``.
    """
    warnings.warn(
        "blend_weights() is deprecated (old SBS API, not paper-faithful PSP). "
        "Use psp_partition(x, knots, n, delta) for the Li & Tian (2011) basis.",
        DeprecationWarning,
        stacklevel=2,
    )
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

    def _step_partition_inner(tv, ctrs, hw):
        ki = len(ctrs)
        if ki == 1:
            return np.ones((1, len(tv)), dtype=float)
        boundaries = 0.5 * (ctrs[:-1] + ctrs[1:])
        steps = []
        for bnd in boundaries:
            u = (tv - bnd) / hw
            x = np.clip(0.5 * (u + 1.0), 0.0, 1.0)
            steps.append(_recursive_smooth_step_unwrapped(x, order=order))
        steps = np.array(steps)
        W2 = np.zeros((ki, len(tv)), dtype=float)
        W2[0] = 1.0 - steps[0]
        for j in range(1, ki - 1):
            W2[j] = steps[j - 1] - steps[j]
        W2[-1] = steps[-1]
        return np.clip(W2, 0.0, 1.0)

    if locality == 0:
        return np.full((k, len(t)), 1.0 / k, dtype=float)

    if periodic:
        period = float(period)
        centers = np.mod(centers, period)
        wrapped_gaps = np.diff(np.concatenate([centers, centers[:1] + period]))
        min_gap = wrapped_gaps.min()
        base_half_width = 0.5 * min_gap if width is None else float(width)
        half_width = max(base_half_width / locality, 1e-12)
        extended = np.concatenate([centers - period, centers, centers + period])
        W_ext = _step_partition_inner(np.mod(t, period), extended, half_width)
        W = np.zeros((k, len(t)), dtype=float)
        for block in range(3):
            W += W_ext[block * k:(block + 1) * k]
        return W

    min_gap = np.diff(centers).min()
    base_half_width = 0.5 * min_gap if width is None else float(width)
    half_width = max(base_half_width / locality, 1e-12)
    return _step_partition_inner(t, centers, half_width)


def apply_knot_weights(W: np.ndarray, knot_weights: ArrayLike) -> np.ndarray:
    """Deprecated. Per-knot renormalisation was removed from the SBS path."""
    raise NotImplementedError(
        "Per-knot weight renormalisation has been removed: PSP evaluation is "
        "strictly non-rational. Use knots_from_weights() to encode weights as "
        "knot spacings (Li & Tian 2011, Eq. 20-21)."
    )

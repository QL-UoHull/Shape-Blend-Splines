"""
Shape-preserving blending basis functions.

This module implements the core weight / basis-function machinery for the
Shape Blend Spline (SBS) technique described in:

    Q. Li, "Shape Blend Splines", Computer-Aided Design, 2011.
    DOI: 10.1016/j.cad.2011.01.006

The central idea is a family of *shape-preserving partition-of-unity* (PU)
weight functions that can be tuned via a *locality parameter* α:

  • α = 0  →  uniform (global) blending: all shapes contribute equally.
  • α = 1  →  standard raised-cosine (smooth local) blending.
  • α > 1  →  more localised; each shape dominates near its own centre
               parameter, producing stronger local shape preservation.

Because the weights always sum to 1, the blended curve interpolates and
blends between the constituent shapes rather than pulling toward zero.

.. note::
   Exact formulae from the paper were not available during development.
   This implementation provides a faithful, documented approximation based
   on the published repository description and standard shape-preserving
   spline literature.  See README.md for full transparency notes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Primitive bump / kernel functions
# ---------------------------------------------------------------------------

def raised_cosine_bump(u: ArrayLike) -> np.ndarray:
    """
    Smooth raised-cosine bump function.

    .. math::
        \\phi(u) =
        \\begin{cases}
          \\tfrac{1}{2}\\bigl(1 + \\cos(\\pi u)\\bigr) & |u| \\le 1 \\\\
          0 & |u| > 1
        \\end{cases}

    The function is non-negative, symmetric, has value 1 at u = 0,
    value 0 at |u| = 1, and zero first derivative at the boundary —
    making it ideal as a smooth partition-of-unity kernel.

    Parameters
    ----------
    u:
        Normalised distance from the centre.  Scalar or array.

    Returns
    -------
    np.ndarray
        Same shape as *u*.
    """
    u = np.asarray(u, dtype=float)
    result = np.zeros_like(u)
    mask = np.abs(u) <= 1.0
    result[mask] = 0.5 * (1.0 + np.cos(np.pi * u[mask]))
    return result


def cubic_hermite_bump(u: ArrayLike) -> np.ndarray:
    """
    Smooth cubic-Hermite bump function (alternative kernel).

    .. math::
        \\phi(u) =
        \\begin{cases}
          1 - 3u^2 + 2|u|^3 & |u| \\le 1 \\\\
          0 & |u| > 1
        \\end{cases}

    This is the standard C¹ cubic Hermite weight used in meshless methods.
    """
    u = np.asarray(u, dtype=float)
    au = np.abs(u)
    result = np.zeros_like(u)
    mask = au <= 1.0
    result[mask] = 1.0 - 3.0 * au[mask] ** 2 + 2.0 * au[mask] ** 3
    return result


# ---------------------------------------------------------------------------
# Shape-preserving weight with locality control
# ---------------------------------------------------------------------------

def shape_blend_kernel(u: ArrayLike, locality: float = 1.0) -> np.ndarray:
    """
    Compute the (unnormalised) shape-blend weight for normalised distance *u*.

    The locality-controlled weight is defined as:

    .. math::
        w(u; \\alpha) = \\phi(u)^\\alpha

    where φ is the raised-cosine bump and α ≥ 0 is the *locality parameter*.

    Parameters
    ----------
    u:
        Normalised distance (scalar or array).
    locality:
        Locality parameter α ≥ 0.

        * α = 0 → constant weight (uniform / global blending).
        * α = 1 → raised-cosine (smooth local blending, default).
        * α > 1 → more concentrated; stronger shape preservation.

    Returns
    -------
    np.ndarray
        Non-negative weights, same shape as *u*.
    """
    phi = raised_cosine_bump(u)
    if locality == 0:
        # Degenerate case: constant weight everywhere the kernel is non-zero
        return np.where(np.abs(np.asarray(u, dtype=float)) <= 1.0, 1.0, 0.0)
    return phi ** float(locality)


# ---------------------------------------------------------------------------
# Partition-of-unity blend weights
# ---------------------------------------------------------------------------

def blend_weights(
    t: ArrayLike,
    centers: ArrayLike,
    locality: float = 1.0,
    width: float | None = None,
) -> np.ndarray:
    """
    Compute normalised shape-blend weights for all shape centres.

    For *k* shapes with centre parameters t₀, t₁, …, t_{k-1}, the weight of
    shape *j* at parameter *t* is:

    .. math::
        W_j(t) = \\frac{w\\!\\left(\\frac{t - t_j}{\\sigma};\\,\\alpha\\right)}
                       {\\sum_{l=0}^{k-1} w\\!\\left(\\frac{t - t_l}{\\sigma};\\,\\alpha\\right)}

    where σ is the support half-width and the denominator ensures
    :math:`\\sum_j W_j(t) = 1` (partition of unity).

    Parameters
    ----------
    t:
        Global curve parameter values, array of shape *(m,)*.
    centers:
        Centre parameter for each shape, array of shape *(k,)*.
    locality:
        Shape-preservation locality parameter α (see :func:`shape_blend_kernel`).
    width:
        Support half-width σ for each weight function.  Defaults to the
        average spacing between consecutive centres, which ensures full
        coverage with mild overlap.

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
    t = np.asarray(t, dtype=float)
    centers = np.asarray(centers, dtype=float)
    k = len(centers)

    if width is None:
        if k > 1:
            # Use average gap (so neighbouring supports just touch)
            width = (centers[-1] - centers[0]) / (k - 1)
        else:
            width = 1.0

    # u[j, i] = (t[i] - centers[j]) / width
    u = (t[np.newaxis, :] - centers[:, np.newaxis]) / width  # (k, m)
    raw = shape_blend_kernel(u, locality)                     # (k, m)

    # Normalise column-wise (partition of unity).
    # If ALL kernels vanish at a parameter value (can happen at domain
    # boundaries when a centre coincides with an endpoint), fall back to
    # assigning unit weight to the nearest centre so the blended curve
    # still evaluates to a meaningful value.
    total = raw.sum(axis=0)                                   # (m,)
    zero_mask = total < 1e-14
    if np.any(zero_mask):
        t_zero = t[zero_mask]
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

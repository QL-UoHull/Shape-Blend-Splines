"""
Parametric shape / primitive definitions.

Each function returns a 2-D point array for parameter values t in a given
range.  These are the *primitives* for Partial Shape-Preserving (PSP) splines:

- In :class:`~shape_blend_splines.curve.BlendedPrimitivePSPSpline` (Eq. 22),
  each primitive is preserved **exactly** on the flat-top of its PSP basis
  function and blended smoothly across the rising-range bands.
- In :class:`~shape_blend_splines.curve.WeightedControlPolygonPSPSpline`,
  control-point positions are the degenerate (constant) case.
- In :class:`~shape_blend_splines.curve.HermitePSPSpline`, the primitive for
  node i is the tangent line P_i + (t - t_i)*v_i (Eq. 23).

Shapes included
---------------
- :func:`line_segment`
- :func:`circle_arc`
- :func:`ellipse_arc`
- :func:`superellipse_arc`
- :func:`rectangle_arc`
- :func:`polyline`
- :func:`star_arc`
- :func:`sine_wave` — sinusoidal primitive for primitive-blending demos
- :func:`helix_2d` — 2-D projection of a helix / cosine-sine primitive
- :func:`from_control_points` — cubic Hermite shape from user control points

All functions accept scalar or 1-D array *t* and return an *(m, 2)* array of
(x, y) coordinates.

Reference
---------
Q. Li, J. Tian, "Partial shape-preserving splines",
Computer-Aided Design 43 (2011) 394-409.  Section 6.2 / Property 6.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_param(t: ArrayLike) -> np.ndarray:
    """Ensure *t* is a 1-D float array."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return t


def closed_polygon_edges(
    vertices: ArrayLike,
    knots: ArrayLike | None = None,
) -> list[Callable[[ArrayLike], np.ndarray]]:
    """
    Return callable edge primitives for a closed polygon.

    Edge ``i`` maps local ``s in [0,1]`` to
    ``(1-s)*V_i + s*V_{(i+1) mod k}``, but each returned callable accepts the
    global parameter ``t`` and normalizes it over that edge's knot interval
    ``[t_i, t_{i+1}]``.

    Parameters
    ----------
    vertices:
        Polygon vertices of shape ``(k, 2)``.
    knots:
        Knot vector of length ``k+1`` defining the edge intervals.
        If omitted, unit intervals ``[0,1], [1,2], ...`` are used.
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or len(verts) < 3:
        raise ValueError("vertices must have shape (k, d) with k >= 3.")

    k = len(verts)
    if knots is None:
        knot_arr = np.arange(k + 1, dtype=float)
    else:
        knot_arr = np.asarray(knots, dtype=float)
        if len(knot_arr) != k + 1:
            raise ValueError("len(knots) must equal len(vertices)+1.")
    if np.any(np.diff(knot_arr) <= 0):
        raise ValueError("knots must be strictly increasing.")

    period = float(knot_arr[-1] - knot_arr[0])
    edges: list[Callable[[ArrayLike], np.ndarray]] = []
    for i in range(k):
        p0 = verts[i]
        p1 = verts[(i + 1) % k]
        a = float(knot_arr[i])
        b = float(knot_arr[i + 1])
        span = b - a

        def _edge(t, p0=p0, p1=p1, a=a, span=span, period=period):
            tt = _to_param(t)
            # For periodic blends, map to the equivalent local interval.
            c = a + 0.5 * span
            local_t = tt - np.round((tt - c) / period) * period if period > 0 else tt
            s = np.clip((local_t - a) / span, 0.0, 1.0)
            return (1.0 - s)[:, np.newaxis] * p0 + s[:, np.newaxis] * p1

        edges.append(_edge)

    return edges


# ---------------------------------------------------------------------------
# Elementary shapes
# ---------------------------------------------------------------------------

def line_segment(
    t: ArrayLike,
    p0: ArrayLike = (0.0, 0.0),
    p1: ArrayLike = (1.0, 0.0),
) -> np.ndarray:
    """
    Straight line segment from *p0* to *p1*.

    Parameters
    ----------
    t:
        Parameter values in [0, 1].
    p0:
        Start point (x, y).
    p1:
        End point (x, y).

    Returns
    -------
    np.ndarray, shape *(m, 2)*
    """
    t = _to_param(t)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    return p0 + t[:, np.newaxis] * (p1 - p0)


def circle_arc(
    t: ArrayLike,
    center: ArrayLike = (0.0, 0.0),
    radius: float = 1.0,
    theta_start: float = 0.0,
    theta_end: float = 2.0 * np.pi,
) -> np.ndarray:
    """
    Circular arc.

    Parameters
    ----------
    t:
        Parameter values in [0, 1].
    center:
        Centre (cx, cy).
    radius:
        Radius r > 0.
    theta_start, theta_end:
        Start / end angles in radians (default: full circle).

    Returns
    -------
    np.ndarray, shape *(m, 2)*
    """
    t = _to_param(t)
    cx, cy = float(center[0]), float(center[1])
    theta = theta_start + t * (theta_end - theta_start)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    return np.column_stack([x, y])


def ellipse_arc(
    t: ArrayLike,
    center: ArrayLike = (0.0, 0.0),
    a: float = 1.5,
    b: float = 1.0,
    theta_start: float = 0.0,
    theta_end: float = 2.0 * np.pi,
) -> np.ndarray:
    """
    Elliptic arc with semi-axes *a* (x) and *b* (y).

    Parameters
    ----------
    t:
        Parameter values in [0, 1].
    center:
        Centre (cx, cy).
    a:
        Semi-major axis (along x).
    b:
        Semi-minor axis (along y).
    theta_start, theta_end:
        Angle range in radians.

    Returns
    -------
    np.ndarray, shape *(m, 2)*
    """
    t = _to_param(t)
    cx, cy = float(center[0]), float(center[1])
    theta = theta_start + t * (theta_end - theta_start)
    x = cx + a * np.cos(theta)
    y = cy + b * np.sin(theta)
    return np.column_stack([x, y])


def superellipse_arc(
    t: ArrayLike,
    center: ArrayLike = (0.0, 0.0),
    a: float = 1.0,
    b: float = 1.0,
    exponent: float = 2.0,
    theta_start: float = 0.0,
    theta_end: float = 2.0 * np.pi,
) -> np.ndarray:
    """
    Superellipse (Lamé curve) arc.

    The parametric equations are:

    .. math::
        x(\\theta) = a \\cdot \\mathrm{sgn}(\\cos\\theta)|\\cos\\theta|^{2/n}

        y(\\theta) = b \\cdot \\mathrm{sgn}(\\sin\\theta)|\\sin\\theta|^{2/n}

    * n = 2 → standard ellipse.
    * n > 2 → rounded rectangle (larger n → more square).
    * 0 < n < 2 → star-like (concave sides).

    Parameters
    ----------
    t:
        Parameter values in [0, 1].
    center:
        Centre (cx, cy).
    a, b:
        Semi-axes.
    exponent:
        Shape exponent n.
    theta_start, theta_end:
        Angle range in radians.

    Returns
    -------
    np.ndarray, shape *(m, 2)*
    """
    t = _to_param(t)
    cx, cy = float(center[0]), float(center[1])
    theta = theta_start + t * (theta_end - theta_start)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = cx + a * np.sign(cos_t) * np.abs(cos_t) ** (2.0 / exponent)
    y = cy + b * np.sign(sin_t) * np.abs(sin_t) ** (2.0 / exponent)
    return np.column_stack([x, y])


def rectangle_arc(
    t: ArrayLike,
    center: ArrayLike = (0.0, 0.0),
    width: float = 2.0,
    height: float = 2.0,
) -> np.ndarray:
    """
    Closed rectangle traversed counter-clockwise at uniform arc-length speed.

    Parameters
    ----------
    t:
        Parameter values in [0, 1] (0 and 1 both map to the bottom-left
        corner).
    center:
        Centre (cx, cy).
    width, height:
        Full width and height.

    Returns
    -------
    np.ndarray, shape *(m, 2)*
    """
    t = _to_param(t)
    cx, cy = float(center[0]), float(center[1])
    hw, hh = width / 2.0, height / 2.0
    perimeter = 2.0 * (width + height)

    # Arc-length parameterisation along the four sides
    # Side lengths as fractions of perimeter
    s = t * perimeter
    x = np.zeros(len(t))
    y = np.zeros(len(t))

    # Bottom side (left → right)
    m = s < width
    x[m] = cx - hw + s[m]
    y[m] = cy - hh

    # Right side (bottom → top)
    m = (s >= width) & (s < width + height)
    x[m] = cx + hw
    y[m] = cy - hh + (s[m] - width)

    # Top side (right → left)
    m = (s >= width + height) & (s < 2 * width + height)
    x[m] = cx + hw - (s[m] - width - height)
    y[m] = cy + hh

    # Left side (top → bottom)
    m = s >= 2 * width + height
    x[m] = cx - hw
    y[m] = cy + hh - (s[m] - 2 * width - height)

    return np.column_stack([x, y])


def polyline(
    t: ArrayLike,
    vertices: ArrayLike,
    closed: bool = False,
) -> np.ndarray:
    """
    Piecewise-linear polyline through *vertices* at uniform speed.

    Parameters
    ----------
    t:
        Parameter values in [0, 1].
    vertices:
        Array of shape *(k, 2)* giving the vertex coordinates.
    closed:
        If True, an extra segment from the last vertex back to the first
        is added.

    Returns
    -------
    np.ndarray, shape *(m, 2)*
    """
    t = _to_param(t)
    vertices = np.asarray(vertices, dtype=float)
    if closed:
        vertices = np.vstack([vertices, vertices[:1]])

    # Cumulative arc-length fractions
    diffs = np.diff(vertices, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total = seg_lengths.sum()
    if total < 1e-14:
        return np.tile(vertices[0], (len(t), 1))

    cum = np.concatenate([[0.0], np.cumsum(seg_lengths) / total])

    # Map t to segment index and local parameter
    idx = np.searchsorted(cum, t, side="right") - 1
    idx = np.clip(idx, 0, len(diffs) - 1)
    span = seg_lengths[idx] / total
    local = (t - cum[idx]) / np.where(span > 0, span, 1.0)
    local = np.clip(local, 0.0, 1.0)

    x = vertices[idx, 0] + local * diffs[idx, 0]
    y = vertices[idx, 1] + local * diffs[idx, 1]
    return np.column_stack([x, y])


def star_arc(
    t: ArrayLike,
    center: ArrayLike = (0.0, 0.0),
    outer_radius: float = 1.0,
    inner_radius: float = 0.4,
    n_points: int = 5,
) -> np.ndarray:
    """
    Regular star polygon (via polyline).

    Parameters
    ----------
    t:
        Parameter values in [0, 1].
    center:
        Centre (cx, cy).
    outer_radius:
        Radius of the outer (tip) vertices.
    inner_radius:
        Radius of the inner (notch) vertices.
    n_points:
        Number of star tips.

    Returns
    -------
    np.ndarray, shape *(m, 2)*
    """
    cx, cy = float(center[0]), float(center[1])
    angles_outer = np.linspace(np.pi / 2.0, np.pi / 2.0 + 2 * np.pi, n_points, endpoint=False)
    angles_inner = angles_outer + np.pi / n_points
    xs = []
    ys = []
    for ao, ai in zip(angles_outer, angles_inner):
        xs.extend([cx + outer_radius * np.cos(ao), cx + inner_radius * np.cos(ai)])
        ys.extend([cy + outer_radius * np.sin(ao), cy + inner_radius * np.sin(ai)])
    verts = np.column_stack([xs, ys])
    return polyline(t, verts, closed=True)


def from_control_points(
    t: ArrayLike,
    control_pts: ArrayLike,
    closed: bool = False,
) -> np.ndarray:
    """
    Smooth cubic Hermite curve through user-supplied control points.

    Tangents are estimated using centripetal Catmull–Rom rules, which give
    visually pleasing results for irregularly spaced points.

    Parameters
    ----------
    t:
        Parameter values in [0, 1].
    control_pts:
        Array of shape *(k, 2)*.  At least 2 points required.
    closed:
        If True, the curve forms a smooth closed loop.

    Returns
    -------
    np.ndarray, shape *(m, 2)*
    """
    t = _to_param(t)
    pts = np.asarray(control_pts, dtype=float)
    k = len(pts)
    if k < 2:
        raise ValueError("Need at least 2 control points.")

    # Centripetal Catmull–Rom tangents
    def _tangent(p_prev, p_curr, p_next):
        d0 = np.linalg.norm(p_curr - p_prev)
        d1 = np.linalg.norm(p_next - p_curr)
        d0 = max(d0, 1e-10)
        d1 = max(d1, 1e-10)
        return (p_next - p_prev) / (d0 + d1) * 2.0

    if closed:
        tangents = np.zeros_like(pts)
        for i in range(k):
            tangents[i] = _tangent(pts[(i - 1) % k], pts[i], pts[(i + 1) % k])

        seg_t = np.mod(t, 1.0) * k
        seg_idx = np.floor(seg_t).astype(int) % k
        u = seg_t - np.floor(seg_t)

        p0 = pts[seg_idx]
        p1 = pts[(seg_idx + 1) % k]
        m0 = tangents[seg_idx]
        m1 = tangents[(seg_idx + 1) % k]
    else:
        n_ctrl = len(pts)
        tangents = np.zeros_like(pts)
        for i in range(1, n_ctrl - 1):
            tangents[i] = _tangent(pts[i - 1], pts[i], pts[i + 1])
        tangents[0] = tangents[1]
        tangents[-1] = tangents[-2]

        # Map global t to segment index and local parameter
        n_segs_actual = n_ctrl - 1
        seg_t = t * n_segs_actual
        seg_idx = np.clip(seg_t.astype(int), 0, n_segs_actual - 1)
        u = seg_t - seg_idx  # local parameter in [0, 1]

        p0 = pts[seg_idx]
        p1 = pts[seg_idx + 1]
        m0 = tangents[seg_idx]
        m1 = tangents[seg_idx + 1]

    # Cubic Hermite basis
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2

    x = h00 * p0[:, 0] + h10 * m0[:, 0] + h01 * p1[:, 0] + h11 * m1[:, 0]
    y = h00 * p0[:, 1] + h10 * m0[:, 1] + h01 * p1[:, 1] + h11 * m1[:, 1]
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Convenience dict for notebook / example use
# ---------------------------------------------------------------------------

SHAPE_REGISTRY = {
    "line": line_segment,
    "circle": circle_arc,
    "ellipse": ellipse_arc,
    "superellipse": superellipse_arc,
    "rectangle": rectangle_arc,
    "polyline": polyline,
    "star": star_arc,
    "control_points": from_control_points,
}


# ---------------------------------------------------------------------------
# Additional primitives for PSP blending demos
# ---------------------------------------------------------------------------

def sine_wave(
    t: ArrayLike,
    x_start: float = 0.0,
    x_end: float = 2.0 * np.pi,
    amplitude: float = 1.0,
    y_offset: float = 0.0,
) -> np.ndarray:
    """
    Sinusoidal primitive for use in :class:`BlendedPrimitivePSPSpline`.

    Parameters
    ----------
    t : array-like
        Parameter values (used as the x coordinate linearly mapped to
        [x_start, x_end]).
    x_start, x_end : float
        Horizontal extent.
    amplitude : float
        Peak amplitude.
    y_offset : float
        Vertical offset.

    Returns
    -------
    np.ndarray, shape (m, 2)
    """
    t = _to_param(t)
    x = x_start + t * (x_end - x_start)
    y = y_offset + amplitude * np.sin(x)
    return np.column_stack([x, y])


def helix_2d(
    t: ArrayLike,
    cx: float = 0.0,
    cy: float = 0.0,
    radius: float = 1.0,
    turns: float = 1.0,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> np.ndarray:
    """
    2-D projection of a helix: (cos, sin) with linearly growing angle.

    Useful as a parametric primitive for :class:`BlendedPrimitivePSPSpline`
    to demonstrate smooth blending between helical and other geometric shapes
    (Fig. 12-style).

    Parameters
    ----------
    t : array-like
        Parameter values in [t_start, t_end].
    cx, cy : float
        Centre of the helix circle.
    radius : float
        Helix radius.
    turns : float
        Number of full revolutions over [t_start, t_end].
    t_start, t_end : float
        Parameter range.

    Returns
    -------
    np.ndarray, shape (m, 2)
    """
    t = _to_param(t)
    fraction = (t - t_start) / max(t_end - t_start, 1e-12)
    theta = fraction * turns * 2.0 * np.pi
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    return np.column_stack([x, y])

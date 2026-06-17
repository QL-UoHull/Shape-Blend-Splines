r"""
Partial Shape-Preserving (PSP) spline curves.

This module implements the paper-faithful PSP curve classes following:

    Q. Li, J. Tian, "Partial shape-preserving splines",
    Computer-Aided Design 43 (2011) 394-409.

Three design modes from the paper (Section 6):

1. **Weighted control-polygon** (Eq. 21; Figs. 9, 10):
   ``WeightedControlPolygonPSPSpline`` — weights encoded as knot spacings;
   purely non-rational.

2. **Primitive blending** (Eq. 22):
   ``BlendedPrimitivePSPSpline`` — blend whole parametric primitives
   (lines, arcs, helix, …); each is reproduced exactly on its flat-top.

3. **Hermite position + velocity** (Eq. 23):
   ``HermitePSPSpline`` — interpolate position *and* velocity at each node
   using the quadratic (n=2) PSP basis.

Plus:
- ``PSPSpline`` — generic base class.
- ``PeriodicPSPSpline`` — closed-loop variant.
- Deprecated aliases ``ShapeBlendSpline``, ``PeriodicShapeBlendSpline``,
  ``ControlPointSpline``, ``ShapeBlender`` for backward compatibility.

Value proposition vs B-spline / NURBS
--------------------------------------
- B-spline: polynomial, partition of unity, C^{n-1}, local control.
  *Cannot* reach basis value = 1 (no flat-top), no selective interpolation.
- NURBS: everything above + exact primitive reproduction via rational weights.
  Rational denominator, extra algebraic complexity.
- PSP spline: everything B-splines have, PLUS flat-top shape preservation,
  weights as knot spacings (non-rational), and an extra design dimension
  delta.  Achieves what NURBS achieves with PURE POLYNOMIALS.
"""

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .basis import (
    CUBIC_C2_ORDER,
    blend_weights,
    interpolated_indices,
    knots_from_weights,
    psp_basis,
    psp_partition,
    shape_preserving_interval,
    smooth_unit_step_delta,
)


# ---------------------------------------------------------------------------
# Generic PSP spline base
# ---------------------------------------------------------------------------

class PSPSpline:
    """
    Generic PSP spline: P(t) = sum_i f_i(t) * B_i(t).

    Each ``f_i`` may be a fixed control point (array) or a callable
    parametric primitive.  When all ``f_i`` are constant points this
    reduces to the weighted control-polygon design (Eq. 21).

    Parameters
    ----------
    primitives : sequence of array-like or callable
        Control points or parametric functions.  A callable must accept a
        1-D float array and return an (m, d) array.
    knots : array-like
        Knot vector of length ``len(primitives) + 1``.
    n : int
        Polynomial degree (default 3 = cubic).
    delta : float
        Rising/blending range.  Smaller delta → wider flat-tops (stronger
        shape preservation).  Larger delta → bump-shaped basis.
    """

    def __init__(
        self,
        primitives: Sequence,
        knots: ArrayLike,
        n: int = 3,
        delta: float = 0.5,
    ) -> None:
        self.primitives = list(primitives)
        self.knots = np.asarray(knots, dtype=float)
        self.n = int(n)
        self.delta = float(delta)

        m = len(self.primitives)
        if len(self.knots) != m + 1:
            raise ValueError(
                f"Need len(knots) = len(primitives)+1 = {m+1}, "
                f"got {len(self.knots)}."
            )
        if np.any(np.diff(self.knots) < 0):
            raise ValueError("knots must be non-decreasing.")

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, t: ArrayLike) -> np.ndarray:
        """
        Evaluate the PSP curve at parameter values *t*.

        Returns
        -------
        np.ndarray, shape (m, d)
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        B = psp_partition(t, self.knots, self.n, self.delta)  # (m, len_t)

        # Determine output dimension from first primitive
        first = self.primitives[0]
        sample = self._eval_prim(first, t[:1])
        d = sample.shape[1]

        result = np.zeros((len(t), d))
        for i, prim in enumerate(self.primitives):
            pts = self._eval_prim(prim, t)  # (len_t, d)
            result += B[i, :, np.newaxis] * pts
        return result

    @staticmethod
    def _eval_prim(prim, t: np.ndarray) -> np.ndarray:
        """Evaluate a primitive (constant array or callable) at parameter t."""
        if callable(prim):
            pts = np.atleast_2d(prim(t))
        else:
            pts = np.asarray(prim, dtype=float)
            if pts.ndim == 1:
                pts = np.tile(pts, (len(t), 1))
        return pts

    # ------------------------------------------------------------------
    # Shape-preservation analysis
    # ------------------------------------------------------------------

    def shape_preserving_intervals(self) -> list[tuple[float, float]]:
        """
        Return the flat-top interval for every basis function.

        A flat-top is non-empty when the knot span >= 2 * delta, meaning
        the basis function equals 1 exactly there and the corresponding
        primitive is reproduced exactly (selective interpolation).

        Returns
        -------
        list of (left, right)
            left > right indicates an empty flat-top (bump shape).
        """
        return [
            shape_preserving_interval(self.knots[i], self.knots[i + 1], self.delta)
            for i in range(len(self.primitives))
        ]

    def interpolated_control_points(self) -> list[int]:
        """
        Indices of control points / primitives interpolated exactly.

        A primitive is interpolated exactly when its knot span >= 2*delta
        (non-empty flat-top).
        """
        return interpolated_indices(self.knots, self.delta)

    def weights_at(self, t: ArrayLike) -> np.ndarray:
        """Return the PSP basis matrix (m, len_t) at parameter values t."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return psp_partition(t, self.knots, self.n, self.delta)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        ax=None,
        n_points: int = 500,
        show_control_polygon: bool = True,
        show_flat_tops: bool = True,
        show_basis: bool = False,
        color: str = "steelblue",
        **kwargs,
    ):
        """
        Plot the PSP curve.

        Parameters
        ----------
        show_flat_tops : bool
            Shade the flat-top (shape-preservation) regions.
        show_basis : bool
            Add a stacked basis-function panel above the curve (Fig. 11 style).
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        t = np.linspace(self.knots[0], self.knots[-1], n_points)
        pts = self.evaluate(t)

        if show_basis:
            fig, (ax_b, ax_c) = plt.subplots(
                2, 1, figsize=(9, 7),
                gridspec_kw={"height_ratios": [1, 2]},
            )
            ax = ax_c
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 5))
            ax_b = None

        line_kw = dict(color=color, lw=2.0, label="PSP curve")
        line_kw.update(kwargs)
        ax.plot(pts[:, 0], pts[:, 1], **line_kw)

        if show_control_polygon:
            ctrl = [
                self._eval_prim(p, np.array([
                    0.5 * (self.knots[i] + self.knots[i + 1])
                ])).flatten()
                for i, p in enumerate(self.primitives)
                if not callable(p)
            ]
            if ctrl:
                ctrl = np.array(ctrl)
                ax.plot(ctrl[:, 0], ctrl[:, 1], "o--", color="tomato",
                        lw=1.2, ms=5, label="Control polygon")

        if show_flat_tops:
            colors_ft = cm.Pastel1.colors
            for idx, (left, right) in enumerate(self.shape_preserving_intervals()):
                if left < right:
                    mask = (t >= left) & (t <= right)
                    if np.any(mask):
                        ax.fill_between(
                            pts[mask, 0], pts[mask, 1] - 0.02,
                            pts[mask, 1] + 0.02,
                            color=colors_ft[idx % len(colors_ft)],
                            alpha=0.5, linewidth=0,
                            label=f"Flat-top {idx}" if idx == 0 else None,
                        )

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"PSP spline  n={self.n}, delta={self.delta:.3g}  "
            f"({len(self.primitives)} control points)"
        )
        ax.legend(fontsize=8)

        if show_basis and ax_b is not None:
            B = self.weights_at(t)
            colors_b = cm.tab10.colors
            interp_idx = self.interpolated_control_points()
            for i in range(len(self.primitives)):
                lbl = f"B_{i}"
                if i in interp_idx:
                    lbl += " ★"
                ax_b.plot(t, B[i], color=colors_b[i % len(colors_b)],
                          lw=1.5, label=lbl)
            ax_b.axhline(1.0, color="k", lw=0.5, ls=":")
            ax_b.set_ylabel("Basis value")
            ax_b.set_title(
                f"PSP basis functions (n={self.n}, delta={self.delta:.3g}); "
                "★ = interpolated"
            )
            ax_b.legend(fontsize=7, ncol=4)
            ax_b.set_ylim(-0.05, 1.1)
            fig.tight_layout()
            return ax_b, ax

        return ax

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_primitives={len(self.primitives)}, "
            f"n={self.n}, delta={self.delta}, "
            f"knots=[{self.knots[0]:.3g}..{self.knots[-1]:.3g}])"
        )


# ---------------------------------------------------------------------------
# Weighted control-polygon design (Eqs. 20-21; Figs. 9, 10)
# ---------------------------------------------------------------------------

class WeightedControlPolygonPSPSpline(PSPSpline):
    """
    PSP curve via weighted control polygon (Eqs. 20-21 of Li & Tian 2011).

    Curve formula (Eq. 21):

    .. math::
       P(t) = \\sum_{i=0}^{N} P_i \\, B^{(n)}_{[a_i,a_{i+1}],\\delta}(t)

    Weights are encoded as **knot spacings** (Eq. 20):

    .. math::
       w_i = a_{i+1} - a_i \\geq 0

    A larger weight means a wider interval, and the curve is pulled closer to
    the corresponding control point — exactly the NURBS weight effect but
    without any rational denominator.

    Parameters
    ----------
    control_pts : array-like, shape (N+1, d)
        Control points.
    weights : array-like of length N+1, optional
        Interval widths w_i >= 0.  Default: equal weights (uniform spacing).
    n : int
        Polynomial degree (default 3).
    delta : float
        Rising/blending range.
    a0 : float
        Starting knot value (default 0.0).
    """

    def __init__(
        self,
        control_pts: ArrayLike,
        weights: ArrayLike | None = None,
        n: int = 3,
        delta: float = 0.5,
        a0: float = 0.0,
    ) -> None:
        control_pts = np.asarray(control_pts, dtype=float)
        if control_pts.ndim == 1:
            control_pts = control_pts[:, np.newaxis]
        N = len(control_pts)

        if weights is None:
            weights = np.ones(N)
        weights = np.asarray(weights, dtype=float)
        if len(weights) != N:
            raise ValueError(
                f"len(weights)={len(weights)} must equal len(control_pts)={N}."
            )

        self.control_pts = control_pts
        self.interval_weights = weights.copy()
        knots = knots_from_weights(weights, a0=a0)

        super().__init__(
            primitives=list(control_pts),
            knots=knots,
            n=n,
            delta=delta,
        )

    def plot(
        self,
        ax=None,
        n_points: int = 500,
        show_control_polygon: bool = True,
        show_flat_tops: bool = True,
        show_basis: bool = False,
        color: str = "steelblue",
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        t = np.linspace(self.knots[0], self.knots[-1], n_points)
        pts = self.evaluate(t)

        if show_basis:
            fig, (ax_b, ax_c) = plt.subplots(
                2, 1, figsize=(9, 7),
                gridspec_kw={"height_ratios": [1, 2]},
            )
            ax = ax_c
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 5))
            ax_b = None

        line_kw = dict(color=color, lw=2.0, label="PSP curve")
        line_kw.update(kwargs)
        ax.plot(pts[:, 0], pts[:, 1], **line_kw)

        if show_control_polygon and self.control_pts.shape[1] >= 2:
            cp = self.control_pts
            ax.plot(cp[:, 0], cp[:, 1], "o--", color="tomato",
                    lw=1.2, ms=5, label="Control polygon")

        if show_flat_tops:
            colors_ft = cm.Pastel1.colors
            for idx, (left, right) in enumerate(self.shape_preserving_intervals()):
                if left < right:
                    mask = (t >= left) & (t <= right)
                    if np.any(mask):
                        seg = pts[mask]
                        ax.fill_between(
                            seg[:, 0],
                            seg[:, 1] - 0.015 * (pts[:, 1].max() - pts[:, 1].min() + 1e-9),
                            seg[:, 1] + 0.015 * (pts[:, 1].max() - pts[:, 1].min() + 1e-9),
                            color=colors_ft[idx % len(colors_ft)],
                            alpha=0.5, linewidth=0,
                        )

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"PSP weighted control polygon  n={self.n}, delta={self.delta:.3g}"
        )
        ax.legend(fontsize=8)

        if show_basis and ax_b is not None:
            B = self.weights_at(t)
            colors_b = cm.tab10.colors
            interp_idx = self.interpolated_control_points()
            for i in range(len(self.primitives)):
                lbl = f"B_{i} (w={self.interval_weights[i]:.2g})"
                if i in interp_idx:
                    lbl += " ★"
                ax_b.plot(t, B[i], color=colors_b[i % len(colors_b)],
                          lw=1.5, label=lbl)
            ax_b.axhline(1.0, color="k", lw=0.5, ls=":")
            ax_b.set_ylabel("Basis value")
            ax_b.set_title("Basis functions")
            ax_b.legend(fontsize=7, ncol=4)
            ax_b.set_ylim(-0.05, 1.1)
            fig.tight_layout()
            return ax_b, ax

        return ax


# ---------------------------------------------------------------------------
# Blended primitive design (Eq. 22)
# ---------------------------------------------------------------------------

class BlendedPrimitivePSPSpline(PSPSpline):
    """
    PSP curve blending parametric primitives (Eq. 22 of Li & Tian 2011).

    Curve formula:

    .. math::
       P(t) = \\sum_i P_i(t) \\, B^{(m)}_{[t_i,t_{i+1}],\\delta}(t)

    Each primitive ``P_i(t)`` is preserved exactly on its flat-top.

    Parameters
    ----------
    primitives : sequence of callable
        Parametric functions ``P_i(t) -> (m, d)`` array.
    knots : array-like
        Knot vector of length len(primitives)+1.
    n : int
        Polynomial degree.
    delta : float
        Rising/blending range.
    """

    def __init__(
        self,
        primitives: Sequence[Callable],
        knots: ArrayLike,
        n: int = 3,
        delta: float = 0.5,
    ) -> None:
        if not all(callable(p) for p in primitives):
            raise ValueError(
                "BlendedPrimitivePSPSpline requires all primitives to be callable."
            )
        super().__init__(primitives=primitives, knots=knots, n=n, delta=delta)


# ---------------------------------------------------------------------------
# Hermite position + velocity design (Eq. 23)
# ---------------------------------------------------------------------------

class HermitePSPSpline:
    """
    PSP Hermite spline: interpolate position *and* velocity (Eq. 23).

    Using the quadratic (n=2) PSP basis:

    .. math::
       P(t) = \\sum_{i=0}^{N} \\bigl(P_i + (t - t_i)\\,v_i\\bigr)
              \\, B^{(2)}_{[a_i,a_{i+1}],\\delta}(t)

    At node ``t_i`` (inside the flat-top): P(t_i) = P_i, P'(t_i) = v_i.

    Each primitive ``(P_i + (t-t_i)*v_i)`` is a line through P_i with
    velocity v_i; on the flat-top it is reproduced exactly (straight
    segment), giving natural embedded straight-line sections with smooth
    joins.

    Parameters
    ----------
    points : array-like, shape (N+1, d)
        Interpolation positions P_i.
    velocities : array-like, shape (N+1, d)
        Tangent velocities v_i at each node.
    knots : array-like, optional
        Knot vector of length N+2.  Defaults to uniform spacing [0..N].
    delta : float
        Rising/blending range for the quadratic (n=2) basis.
        Must satisfy: each knot span >= 2*delta for interpolation to hold.
    """

    def __init__(
        self,
        points: ArrayLike,
        velocities: ArrayLike,
        knots: ArrayLike | None = None,
        delta: float = 0.4,
    ) -> None:
        self.points = np.asarray(points, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        N = len(self.points)

        if len(self.velocities) != N:
            raise ValueError("len(velocities) must equal len(points).")
        if self.points.ndim == 1:
            self.points = self.points[:, np.newaxis]
        if self.velocities.ndim == 1:
            self.velocities = self.velocities[:, np.newaxis]

        if knots is None:
            knots = np.arange(float(N + 1))
        self.knots = np.asarray(knots, dtype=float)
        if len(self.knots) != N + 1:
            raise ValueError(
                f"len(knots) must be {N + 1}, got {len(self.knots)}."
            )

        self.delta = float(delta)
        self.n = 2  # quadratic basis

        # Node parameters: midpoint of each knot span
        self.t_nodes = 0.5 * (self.knots[:-1] + self.knots[1:])

    def evaluate(self, t: ArrayLike) -> np.ndarray:
        """Evaluate the Hermite PSP curve."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        B = psp_partition(t, self.knots, self.n, self.delta)  # (N, len_t)

        d = self.points.shape[1]
        result = np.zeros((len(t), d))
        for i in range(len(self.points)):
            # Primitive: line through P_i with velocity v_i at t_i
            dt = t - self.t_nodes[i]  # (len_t,)
            prim_i = self.points[i] + dt[:, np.newaxis] * self.velocities[i]
            result += B[i, :, np.newaxis] * prim_i
        return result

    def evaluate_deriv(self, t: ArrayLike) -> np.ndarray:
        """Evaluate the derivative P'(t) of the Hermite PSP curve."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        B = psp_partition(t, self.knots, self.n, self.delta)

        # d/dt B^{(2)}_{[a,b],delta}(t):
        # = d/dt H_{2,delta}(t-a) - d/dt H_{2,delta}(t-b)
        # = (2/delta) * H_2'(2*(t-a)/delta) - (2/delta) * H_2'(2*(t-b)/delta)
        # where H_2'(x) = (1/8)*(2(x+2) H_0(x+2) - 4x H_0(x) + 2(x-2) H_0(x-2))
        # For simplicity, use finite differences for the derivative
        eps = 1e-6
        B_fwd = psp_partition(t + eps, self.knots, self.n, self.delta)
        B_bwd = psp_partition(t - eps, self.knots, self.n, self.delta)
        dB = (B_fwd - B_bwd) / (2 * eps)  # (N, len_t)

        d = self.points.shape[1]
        result = np.zeros((len(t), d))
        for i in range(len(self.points)):
            dt = t - self.t_nodes[i]
            prim_i = self.points[i] + dt[:, np.newaxis] * self.velocities[i]
            dprim_i = self.velocities[i] * np.ones((len(t), 1))
            result += B[i, :, np.newaxis] * dprim_i + dB[i, :, np.newaxis] * prim_i
        return result

    def shape_preserving_intervals(self) -> list[tuple[float, float]]:
        return [
            shape_preserving_interval(self.knots[i], self.knots[i + 1], self.delta)
            for i in range(len(self.points))
        ]

    def interpolated_control_points(self) -> list[int]:
        return interpolated_indices(self.knots, self.delta)

    def weights_at(self, t: ArrayLike) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return psp_partition(t, self.knots, self.n, self.delta)

    def plot(
        self,
        ax=None,
        n_points: int = 500,
        show_control_pts: bool = True,
        show_flat_tops: bool = True,
        show_basis: bool = False,
        color: str = "steelblue",
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        t = np.linspace(self.knots[0], self.knots[-1], n_points)
        pts = self.evaluate(t)

        if show_basis:
            fig, (ax_b, ax_c) = plt.subplots(
                2, 1, figsize=(9, 7),
                gridspec_kw={"height_ratios": [1, 2]},
            )
            ax = ax_c
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 5))
            ax_b = None

        line_kw = dict(color=color, lw=2.0, label="Hermite PSP curve")
        line_kw.update(kwargs)
        ax.plot(pts[:, 0], pts[:, 1], **line_kw)

        if show_control_pts and self.points.shape[1] >= 2:
            ax.scatter(self.points[:, 0], self.points[:, 1],
                       color="tomato", s=40, zorder=5, label="Nodes P_i")

        if show_flat_tops:
            colors_ft = plt.cm.Pastel1.colors
            for idx, (left, right) in enumerate(self.shape_preserving_intervals()):
                if left < right:
                    mask = (t >= left) & (t <= right)
                    if np.any(mask):
                        seg = pts[mask]
                        ax.fill_between(
                            seg[:, 0],
                            seg[:, 1] - 0.015,
                            seg[:, 1] + 0.015,
                            color=colors_ft[idx % len(colors_ft)],
                            alpha=0.5, linewidth=0,
                        )

        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Hermite PSP spline  n=2, delta={self.delta:.3g}")
        ax.legend(fontsize=8)

        if show_basis and ax_b is not None:
            B = self.weights_at(t)
            colors_b = cm.tab10.colors
            interp_idx = self.interpolated_control_points()
            for i in range(len(self.points)):
                lbl = f"B_{i}"
                if i in interp_idx:
                    lbl += " ★"
                ax_b.plot(t, B[i], color=colors_b[i % len(colors_b)],
                          lw=1.5, label=lbl)
            ax_b.axhline(1.0, color="k", lw=0.5, ls=":")
            ax_b.set_ylabel("Basis value")
            ax_b.set_title("Basis functions (n=2)")
            ax_b.legend(fontsize=7, ncol=4)
            ax_b.set_ylim(-0.05, 1.1)
            fig.tight_layout()
            return ax_b, ax

        return ax

    def __repr__(self) -> str:
        return (
            f"HermitePSPSpline(N={len(self.points)}, "
            f"delta={self.delta}, "
            f"knots=[{self.knots[0]:.3g}..{self.knots[-1]:.3g}])"
        )


# ---------------------------------------------------------------------------
# Periodic (closed) PSP spline
# ---------------------------------------------------------------------------

class PeriodicPSPSpline(PSPSpline):
    """
    Closed-loop periodic PSP spline.

    The parameter domain is periodic: the last knot span connects back to
    the first.  Suitable for closed curves.

    Parameters
    ----------
    primitives : sequence
        Control points (array) or callable parametric primitives.
    knots : array-like
        Knot vector of length len(primitives)+1; the period is
        knots[-1] - knots[0].
    n : int
        Polynomial degree.
    delta : float
        Rising/blending range.
    """

    def evaluate(self, t: ArrayLike) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        B = psp_partition(
            t, self.knots, self.n, self.delta,
            periodic=True,
            period=float(self.knots[-1] - self.knots[0]),
        )
        first = self.primitives[0]
        sample = self._eval_prim(first, t[:1])
        d = sample.shape[1]
        result = np.zeros((len(t), d))
        for i, prim in enumerate(self.primitives):
            pts = self._eval_prim(prim, t)
            result += B[i, :, np.newaxis] * pts
        return result


# ---------------------------------------------------------------------------
# Backward-compatible deprecated aliases
# ---------------------------------------------------------------------------

class ShapeBlendSpline:
    """
    Deprecated. Old SBS curve class.

    Use :class:`WeightedControlPolygonPSPSpline` or
    :class:`BlendedPrimitivePSPSpline` instead.

    This class forwards to the old midpoint-boundary blend_weights
    implementation for backward compatibility.  It is **not** the paper's
    PSP technique.
    """

    def __init__(
        self,
        shapes: Sequence[Callable],
        t_centers: ArrayLike | None = None,
        locality: float = 1.0,
        blend_width: float | None = None,
        smooth_order: int = CUBIC_C2_ORDER,
        closed: bool = False,
        period: float = 1.0,
        knot_weights: ArrayLike | None = None,
    ) -> None:
        warnings.warn(
            "ShapeBlendSpline is deprecated. "
            "Use WeightedControlPolygonPSPSpline or BlendedPrimitivePSPSpline "
            "for the paper-faithful PSP implementation "
            "(Li & Tian, CAD 43, 394-409, 2011).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.shapes = list(shapes)
        k = len(self.shapes)
        if k == 0:
            raise ValueError("At least one shape is required.")

        self.closed = bool(closed)
        self.period = float(period)
        if self.period <= 0:
            raise ValueError("period must be positive.")

        if t_centers is None:
            t_centers = np.linspace(
                0.0, self.period, k, endpoint=not self.closed
            )
        self.t_centers = np.asarray(t_centers, dtype=float)
        if len(self.t_centers) != k:
            raise ValueError(
                f"len(t_centers)={len(self.t_centers)} must equal len(shapes)={k}."
            )

        self.locality = float(locality)
        self.blend_width = blend_width
        self.smooth_order = int(smooth_order)

        if knot_weights is not None:
            raise ValueError(
                "knot_weights are no longer supported. Use "
                "WeightedControlPolygonPSPSpline with the 'weights' parameter."
            )

    def evaluate(self, t: ArrayLike) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            W = blend_weights(
                t, self.t_centers, self.locality, self.blend_width,
                periodic=self.closed, period=self.period,
                order=self.smooth_order,
            )
        result = np.zeros((len(t), 2))
        for j, shape_fn in enumerate(self.shapes):
            pts = np.atleast_2d(shape_fn(t))
            result += W[j, :, np.newaxis] * pts
        return result

    def weights_at(self, t: ArrayLike) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return blend_weights(
                t, self.t_centers, self.locality, self.blend_width,
                periodic=self.closed, period=self.period,
                order=self.smooth_order,
            )

    def evaluate_shape(self, j: int, t: ArrayLike) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.atleast_2d(self.shapes[j](t))

    def __len__(self) -> int:
        return len(self.shapes)

    def __repr__(self) -> str:
        return (
            f"ShapeBlendSpline(n_shapes={len(self.shapes)}, "
            f"locality={self.locality}, smooth_order={self.smooth_order}, "
            f"closed={self.closed}) [DEPRECATED]"
        )

    def plot(self, ax=None, n_points: int = 500, show_components: bool = True,
             show_weights: bool = False, blend_color: str = "steelblue",
             component_alpha: float = 0.35, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        t = np.linspace(0.0, self.period, n_points, endpoint=not self.closed)
        pts = self.evaluate(t)

        if show_weights:
            fig, (ax_curve, ax_w) = plt.subplots(
                1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]}
            )
            ax = ax_curve
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(7, 5))

        colors = cm.tab10.colors
        if show_components:
            for j, shape_fn in enumerate(self.shapes):
                sp = np.atleast_2d(shape_fn(t))
                label = getattr(shape_fn, "__name__", f"Shape {j}")
                ax.plot(sp[:, 0], sp[:, 1], "--",
                        color=colors[j % len(colors)],
                        alpha=component_alpha, lw=1.2,
                        label=f"{label} (component)")

        line_kwargs = dict(color=blend_color, lw=2.0, label="Shape Blend Spline")
        line_kwargs.update(kwargs)
        ax.plot(pts[:, 0], pts[:, 1], **line_kwargs)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"{'Closed' if self.closed else 'Open'} SBS (deprecated)  "
            f"(alpha={self.locality:.2f}, k={len(self.shapes)} shapes)"
        )

        if show_weights:
            W = self.weights_at(t)
            for j in range(len(self.shapes)):
                name = getattr(self.shapes[j], "__name__", f"Shape {j}")
                ax_w.plot(t, W[j], color=colors[j % len(colors)], label=name)
            ax_w.set_xlabel("t")
            ax_w.set_ylabel("Weight W_j(t)")
            ax_w.set_title("Blend weights (partition of unity)")
            ax_w.legend(fontsize=8)
            ax_w.set_ylim(-0.05, 1.05)
            return ax, ax_w

        return ax


class PeriodicShapeBlendSpline(ShapeBlendSpline):
    """Deprecated. Closed periodic SBS curve. Use PeriodicPSPSpline instead."""

    def __init__(
        self,
        shapes: Sequence[Callable],
        t_centers: ArrayLike | None = None,
        locality: float = 1.0,
        blend_width: float | None = None,
        smooth_order: int = CUBIC_C2_ORDER,
        period: float = 1.0,
        knot_weights: ArrayLike | None = None,
    ) -> None:
        # Suppress the parent's DeprecationWarning so we can issue our own
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            super().__init__(
                shapes=shapes,
                t_centers=t_centers,
                locality=locality,
                blend_width=blend_width,
                smooth_order=smooth_order,
                closed=True,
                period=period,
                knot_weights=knot_weights,
            )
        warnings.warn(
            "PeriodicShapeBlendSpline is deprecated. "
            "Use PeriodicPSPSpline for the paper-faithful closed PSP curve "
            "(Li & Tian, CAD 43, 394-409, 2011).",
            DeprecationWarning,
            stacklevel=2,
        )


class ControlPointSpline(ShapeBlendSpline):
    """
    Deprecated. SBS control-point convenience class.
    Use WeightedControlPolygonPSPSpline instead.
    """

    def __init__(
        self,
        control_pts: ArrayLike,
        shape_fn=None,
        locality: float = 1.0,
        blend_width: float | None = None,
        smooth_order: int = CUBIC_C2_ORDER,
        closed: bool = False,
    ) -> None:
        from .shapes import from_control_points

        control_pts = np.asarray(control_pts, dtype=float)
        k = len(control_pts)
        if k < 2:
            raise ValueError("Need at least 2 control points.")

        if shape_fn is None:
            def _default_shape(t, _pts=control_pts):
                return from_control_points(t, _pts, closed=closed)
            shapes = [_default_shape]
            t_centers = np.array([0.0 if closed else 0.5])
        elif callable(shape_fn):
            shapes = [shape_fn]
            t_centers = np.array([0.0 if closed else 0.5])
        else:
            shapes = list(shape_fn)
            t_centers = np.linspace(0.0, 1.0, len(shapes), endpoint=not closed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            super().__init__(
                shapes=shapes,
                t_centers=t_centers,
                locality=locality,
                blend_width=blend_width,
                smooth_order=smooth_order,
                closed=closed,
            )
        warnings.warn(
            "ControlPointSpline is deprecated. "
            "Use WeightedControlPolygonPSPSpline instead "
            "(Li & Tian, CAD 43, 394-409, 2011).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.control_pts = control_pts

r"""
Shape Blend Spline — main curve class.

This module provides :class:`ShapeBlendSpline`, the primary user-facing class
for the Shape Blend Spline (SBS) technique, together with a periodic variant
for closed curves.

A Shape Blend Spline is a parametric planar curve defined as a
*weighted blend* of *k* constituent shapes:

.. math::
    \\mathbf{C}(t) = \\sum_{j=0}^{k-1} W_j(t)\\, \\mathbf{S}_j(t)

where:

* :math:`\\mathbf{S}_j(t)` is the *j*-th parametric shape (e.g. circle arc,
  line segment, free-form Hermite curve …).
* :math:`W_j(t)` is the shape-preserving partition-of-unity weight of shape
  *j* (see :mod:`shape_blend_splines.basis`).
* The locality parameter α controls how tightly each :math:`W_j` is
  concentrated around its centre parameter :math:`t_j`.

When α is large, :math:`\\mathbf{C}(t)` is nearly identical to
:math:`\\mathbf{S}_j(t)` near :math:`t = t_j` (strong shape preservation),
while transitions between shapes are smooth.

Reference
---------
Q. Li, "Shape Blend Splines", *Computer-Aided Design*, 2011.
DOI: 10.1016/j.cad.2011.01.006

.. note::
   The central implementation is non-rational:

   .. math::
      \mathbf{C}(t) = \sum_j W_j(t)\,\mathbf{S}_j(t)

   The basis weights form a partition of unity on either an open or periodic
   parameter domain.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .basis import blend_weights


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ShapeBlendSpline:
    """
    A Shape Blend Spline defined by blending *k* parametric shapes.

    Parameters
    ----------
    shapes:
        Sequence of callables, each with signature ``shape(t) → np.ndarray``
        returning an *(m, 2)* array for parameter array *t* ∈ [0, 1].
        See :mod:`shape_blend_splines.shapes` for a catalogue.
    t_centers:
        Centre parameter values t₀ < t₁ < … < t_{k-1} for each shape,
        within the global parameter range [0, 1].  Defaults to uniform
        spacing.
    locality:
        Shape-preservation locality parameter α ≥ 0 (default 1.0).
        Higher values preserve individual shapes more strongly.
    blend_width:
        Support half-width σ for the weight functions.  Defaults to the
        mean inter-centre spacing.
    closed:
        If True, evaluate weights on a periodic domain so the first and last
        shapes are neighbours. This is the standard closed-curve SBS setting.
    period:
        Period length for the global parameter domain when ``closed=True``.

    Examples
    --------
    >>> import numpy as np
    >>> from shape_blend_splines.shapes import circle_arc, ellipse_arc
    >>> from shape_blend_splines.curve import ShapeBlendSpline
    >>> sbs = ShapeBlendSpline(
    ...     shapes=[circle_arc, ellipse_arc],
    ...     locality=2.0,
    ... )
    >>> pts = sbs.evaluate(np.linspace(0, 1, 100))
    >>> pts.shape
    (100, 2)
    """

    def __init__(
        self,
        shapes: Sequence[Callable],
        t_centers: ArrayLike | None = None,
        locality: float = 1.0,
        blend_width: float | None = None,
        closed: bool = False,
        period: float = 1.0,
    ) -> None:
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
                0.0,
                self.period,
                k,
                endpoint=not self.closed,
            )
        self.t_centers = np.asarray(t_centers, dtype=float)
        if len(self.t_centers) != k:
            raise ValueError(
                f"len(t_centers)={len(self.t_centers)} must equal len(shapes)={k}."
            )

        self.locality = float(locality)
        self.blend_width = blend_width

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, t: ArrayLike) -> np.ndarray:
        """
        Evaluate the blended curve at parameter values *t*.

        Parameters
        ----------
        t:
            Parameter values in [0, 1], scalar or 1-D array.

        Returns
        -------
        np.ndarray, shape *(m, 2)*
            (x, y) coordinates of the blended curve.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        k = len(self.shapes)

        # Compute normalised blend weights  (k, m)
        W = blend_weights(
            t,
            self.t_centers,
            self.locality,
            self.blend_width,
            periodic=self.closed,
            period=self.period,
        )

        # Weighted sum of shape evaluations
        result = np.zeros((len(t), 2))
        for j, shape_fn in enumerate(self.shapes):
            pts = np.atleast_2d(shape_fn(t))  # (m, 2)
            result += W[j, :, np.newaxis] * pts

        return result

    def evaluate_shape(self, j: int, t: ArrayLike) -> np.ndarray:
        """
        Evaluate the *j*-th constituent shape without blending.

        Useful for visualising individual shapes.

        Parameters
        ----------
        j:
            Shape index (0-based).
        t:
            Parameter values in [0, 1].

        Returns
        -------
        np.ndarray, shape *(m, 2)*
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.atleast_2d(self.shapes[j](t))

    def weights_at(self, t: ArrayLike) -> np.ndarray:
        """
        Return the blend weight matrix at parameter values *t*.

        Parameters
        ----------
        t:
            Parameter values in [0, 1].

        Returns
        -------
        np.ndarray, shape *(k, m)*
            W[j, i] is the weight of shape *j* at parameter t[i].
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return blend_weights(
            t,
            self.t_centers,
            self.locality,
            self.blend_width,
            periodic=self.closed,
            period=self.period,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.shapes)

    def __repr__(self) -> str:
        return (
            f"ShapeBlendSpline(n_shapes={len(self.shapes)}, "
            f"locality={self.locality}, "
            f"closed={self.closed}, "
            f"t_centers={self.t_centers})"
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        ax=None,
        n_points: int = 500,
        show_components: bool = True,
        show_weights: bool = False,
        blend_color: str = "steelblue",
        component_alpha: float = 0.35,
        **kwargs,
    ):
        """
        Plot the blended curve and optionally the individual component shapes.

        Parameters
        ----------
        ax:
            Matplotlib ``Axes`` object.  Created automatically if None.
        n_points:
            Number of sample points for plotting.
        show_components:
            If True, draw each constituent shape as a dashed background curve.
        show_weights:
            If True, add a second axes showing the weight functions W_j(t).
        blend_color:
            Line colour for the blended curve.
        component_alpha:
            Opacity of component shape lines.
        **kwargs:
            Forwarded to ``ax.plot`` for the main blended curve.

        Returns
        -------
        ax : matplotlib Axes (or tuple of Axes when show_weights=True)
        """
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
                ax.plot(
                    sp[:, 0],
                    sp[:, 1],
                    "--",
                    color=colors[j % len(colors)],
                    alpha=component_alpha,
                    lw=1.2,
                    label=f"{label} (component)",
                )
            ax.plot(
                self.t_centers * (pts[:, 0].max() - pts[:, 0].min())
                + pts[:, 0].min()
                if False
                else [],
                [],
                " ",
            )  # spacer for legend

        line_kwargs = dict(color=blend_color, lw=2.0, label="Shape Blend Spline")
        line_kwargs.update(kwargs)
        ax.plot(pts[:, 0], pts[:, 1], **line_kwargs)

        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"{'Closed' if self.closed else 'Open'} Shape Blend Spline  "
            f"(α = {self.locality:.2f}, "
            f"k = {len(self.shapes)} shapes)"
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


# ---------------------------------------------------------------------------
# Explicit periodic alias
# ---------------------------------------------------------------------------

class PeriodicShapeBlendSpline(ShapeBlendSpline):
    """Convenience subclass for closed, periodic SBS curves."""

    def __init__(
        self,
        shapes: Sequence[Callable],
        t_centers: ArrayLike | None = None,
        locality: float = 1.0,
        blend_width: float | None = None,
        period: float = 1.0,
    ) -> None:
        super().__init__(
            shapes=shapes,
            t_centers=t_centers,
            locality=locality,
            blend_width=blend_width,
            closed=True,
            period=period,
        )


# ---------------------------------------------------------------------------
# Convenience constructor: spline through control points
# ---------------------------------------------------------------------------

class ControlPointSpline(ShapeBlendSpline):
    """
    Shape Blend Spline defined via a sequence of (x, y) control points and
    an optional shape type per segment.

    This is a higher-level interface that automatically sets up the shapes and
    centre parameters based on the supplied control points.

    Each *segment* between consecutive control points is represented by the
    shape function associated with that segment.  When ``shape_fn`` is None,
    a straight line segment is used.

    Parameters
    ----------
    control_pts:
        Array-like of shape *(k, 2)* giving k ≥ 2 control points.
    shape_fn:
        A single callable or a list of k-1 callables (one per segment).
        Each callable has signature ``fn(t) → (m, 2)`` where t ∈ [0, 1].
        Defaults to :func:`~shape_blend_splines.shapes.line_segment` for
        each segment.
    locality:
        Shape-preservation locality parameter α.
    blend_width:
        Support half-width σ.

    Examples
    --------
    >>> import numpy as np
    >>> from shape_blend_splines.curve import ControlPointSpline
    >>> pts = np.array([[0,0],[1,0.5],[2,0],[3,0.5],[4,0]])
    >>> sbs = ControlPointSpline(pts, locality=2.0)
    >>> curve = sbs.evaluate(np.linspace(0, 1, 200))
    """

    def __init__(
        self,
        control_pts: ArrayLike,
        shape_fn=None,
        locality: float = 1.0,
        blend_width: float | None = None,
        closed: bool = False,
    ) -> None:
        from .shapes import from_control_points

        control_pts = np.asarray(control_pts, dtype=float)
        k = len(control_pts)
        if k < 2:
            raise ValueError("Need at least 2 control points.")

        if shape_fn is None:
            # Default: smooth Catmull-Rom through the control points
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

        super().__init__(
            shapes=shapes,
            t_centers=t_centers,
            locality=locality,
            blend_width=blend_width,
            closed=closed,
        )
        self.control_pts = control_pts

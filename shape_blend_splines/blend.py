"""
Higher-level shape blending utilities.

This module builds on :class:`~shape_blend_splines.curve.ShapeBlendSpline` to
provide convenience helpers for common blending tasks:

* :func:`blend_two_shapes`   — morph smoothly from one shape to another.
* :func:`blend_shape_series` — blend along an ordered sequence of shapes.
* :func:`shape_morph`        — compute a series of intermediate blended curves.
* :class:`ShapeBlender`      — interactive / programmatic multi-shape blender.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .curve import ShapeBlendSpline
from .shapes import line_segment, circle_arc, ellipse_arc


# ---------------------------------------------------------------------------
# Two-shape blending
# ---------------------------------------------------------------------------

def blend_two_shapes(
    shape_a: Callable,
    shape_b: Callable,
    blend: float = 0.5,
    locality: float = 1.0,
) -> "ShapeBlender":
    """
    Create a :class:`ShapeBlender` that blends between *shape_a* and *shape_b*.

    The blended curve is:

    .. math::
        \\mathbf{C}(t) = (1 - \\beta)\\,\\mathbf{S}_a(t) + \\beta\\,\\mathbf{S}_b(t)

    where β = *blend*.

    Parameters
    ----------
    shape_a:
        First shape callable  ``shape_a(t) → (m, 2)``.
    shape_b:
        Second shape callable ``shape_b(t) → (m, 2)``.
    blend:
        Blend parameter β ∈ [0, 1].

        * β = 0 → result = *shape_a* exactly.
        * β = 1 → result = *shape_b* exactly.
        * β = 0.5 → equal mix (default).
    locality:
        Locality parameter α (reserved for future spatial blending;
        not used by :class:`ShapeBlender` which applies uniform weights).

    Returns
    -------
    ShapeBlender

    Examples
    --------
    >>> from shape_blend_splines.shapes import circle_arc, ellipse_arc
    >>> from shape_blend_splines.blend import blend_two_shapes
    >>> sbs = blend_two_shapes(circle_arc, ellipse_arc, blend=0.3)
    >>> pts = sbs.evaluate(np.linspace(0, 1, 100))
    """
    beta = float(np.clip(blend, 0.0, 1.0))
    return ShapeBlender([shape_a, shape_b], weights=[1.0 - beta, beta])


# ---------------------------------------------------------------------------
# Series blending
# ---------------------------------------------------------------------------

def blend_shape_series(
    shapes: Sequence[Callable],
    locality: float = 1.0,
    t_centers: ArrayLike | None = None,
) -> ShapeBlendSpline:
    """
    Blend an ordered sequence of shapes with uniformly spaced centre parameters.

    Parameters
    ----------
    shapes:
        Ordered list of shape callables.
    locality:
        Locality parameter α.
    t_centers:
        Optional explicit centre parameters.  Defaults to uniform spacing.

    Returns
    -------
    ShapeBlendSpline

    Examples
    --------
    >>> from shape_blend_splines.shapes import circle_arc, ellipse_arc, star_arc
    >>> from shape_blend_splines.blend import blend_shape_series
    >>> sbs = blend_shape_series([circle_arc, ellipse_arc, star_arc], locality=2.0)
    """
    return ShapeBlendSpline(shapes=shapes, t_centers=t_centers, locality=locality)


# ---------------------------------------------------------------------------
# Shape morphing
# ---------------------------------------------------------------------------

def shape_morph(
    shape_a: Callable,
    shape_b: Callable,
    n_frames: int = 5,
    locality: float = 2.0,
    n_points: int = 300,
) -> list[np.ndarray]:
    """
    Compute *n_frames* intermediate curves morphing from *shape_a* to *shape_b*.

    Parameters
    ----------
    shape_a:
        Source shape callable.
    shape_b:
        Target shape callable.
    n_frames:
        Number of frames (including start and end).
    locality:
        Locality parameter α used for each intermediate blend.
    n_points:
        Number of points per curve.

    Returns
    -------
    list of np.ndarray, each of shape *(n_points, 2)*
        Intermediate blended curves, from β = 0 (shape_a) to β = 1 (shape_b).
    """
    t = np.linspace(0.0, 1.0, n_points)
    frames = []
    for beta in np.linspace(0.0, 1.0, n_frames):
        sbs = blend_two_shapes(shape_a, shape_b, blend=beta, locality=locality)
        frames.append(sbs.evaluate(t))
    return frames


# ---------------------------------------------------------------------------
# ShapeBlender — interactive multi-shape blender
# ---------------------------------------------------------------------------

class ShapeBlender:
    """
    Multi-shape blender with adjustable per-shape weights and locality.

    Unlike :class:`~shape_blend_splines.curve.ShapeBlendSpline`, which uses
    automatic centre-based weights, :class:`ShapeBlender` lets the caller
    directly specify the *unnormalised* weight for each shape.  The weights
    are then normalised to a partition of unity.

    This is useful for interactive exploration where users drag sliders to
    vary how much each shape contributes.

    Parameters
    ----------
    shapes:
        List of shape callables.
    weights:
        Initial per-shape weights ≥ 0.  Defaults to equal weights.

    Examples
    --------
    >>> from shape_blend_splines.shapes import circle_arc, ellipse_arc, star_arc
    >>> from shape_blend_splines.blend import ShapeBlender
    >>> blender = ShapeBlender([circle_arc, ellipse_arc, star_arc])
    >>> blender.set_weights([1.0, 0.0, 0.0])
    >>> pts = blender.evaluate(100)
    """

    def __init__(
        self,
        shapes: Sequence[Callable],
        weights: Sequence[float] | None = None,
    ) -> None:
        self.shapes = list(shapes)
        k = len(self.shapes)
        if weights is None:
            self._weights = np.ones(k)
        else:
            self._weights = np.asarray(weights, dtype=float)
        if len(self._weights) != k:
            raise ValueError("len(weights) must equal len(shapes).")

    @property
    def weights(self) -> np.ndarray:
        """Normalised per-shape weights (sum to 1)."""
        total = self._weights.sum()
        if total < 1e-12:
            return np.ones(len(self.shapes)) / len(self.shapes)
        return self._weights / total

    def set_weights(self, weights: Sequence[float]) -> None:
        """Set per-shape weights (unnormalised; will be normalised on access)."""
        self._weights = np.asarray(weights, dtype=float)

    def evaluate(self, t: ArrayLike | None = None, n_points: int = 300) -> np.ndarray:
        """
        Evaluate the globally-weighted blend.

        Parameters
        ----------
        t:
            Parameter values in [0, 1].  If None, *n_points* uniformly-spaced
            values are used.
        n_points:
            Number of uniformly-spaced parameter samples (used only when *t*
            is None).

        Returns
        -------
        np.ndarray, shape *(m, 2)*
        """
        if t is None:
            t = np.linspace(0.0, 1.0, n_points)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        w = self.weights
        result = np.zeros((len(t), 2))
        for j, shape_fn in enumerate(self.shapes):
            pts = np.atleast_2d(shape_fn(t))
            result += w[j] * pts
        return result

    def plot(self, ax=None, n_points: int = 400, **kwargs):
        """
        Plot the currently blended curve.

        Parameters
        ----------
        ax:
            Matplotlib Axes (created if None).
        n_points:
            Sample count.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        pts = self.evaluate(n_points)
        line_kw = dict(color="steelblue", lw=2, label="Blended shape")
        line_kw.update(kwargs)
        ax.plot(pts[:, 0], pts[:, 1], **line_kw)
        ax.set_aspect("equal")
        wstr = ", ".join(f"{w:.2f}" for w in self.weights)
        ax.set_title(f"Blended shape  weights=[{wstr}]")
        return ax

    def __repr__(self) -> str:
        wstr = ", ".join(f"{w:.3f}" for w in self.weights)
        return f"ShapeBlender(shapes={len(self.shapes)}, weights=[{wstr}])"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def circle_to_ellipse(
    blend: float = 0.5,
    center=(0, 0),
    radius: float = 1.0,
    a: float = 1.5,
    b: float = 0.8,
    locality: float = 1.0,
) -> ShapeBlendSpline:
    """
    Convenience: blend a circle and an ellipse.

    Parameters
    ----------
    blend:
        Blend factor β ∈ [0, 1].  0 → circle, 1 → ellipse.
    center:
        Shared centre for both shapes.
    radius:
        Circle radius.
    a, b:
        Ellipse semi-axes.
    locality:
        Locality parameter α.

    Returns
    -------
    ShapeBlendSpline
    """
    s_circle = partial(circle_arc, center=center, radius=radius)
    s_ellipse = partial(ellipse_arc, center=center, a=a, b=b)
    return blend_two_shapes(s_circle, s_ellipse, blend=blend, locality=locality)

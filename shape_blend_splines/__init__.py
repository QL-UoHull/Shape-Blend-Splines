"""
Partial Shape-Preserving Splines (PSP Splines)
===============================================

A Python implementation of the PSP spline technique from:

    Q. Li, J. Tian, "Partial shape-preserving splines",
    Computer-Aided Design 43 (2011) 394-409.

The central idea (Eq. 17): the PSP basis on interval [a, b] is the
*difference of two smooth unit step functions*:

    B^{(n)}_{[a,b],delta}(x) = H_{n,delta}(x-a) - H_{n,delta}(x-b)

Value proposition vs B-spline / NURBS
--------------------------------------
- **B-spline**: polynomial, partition of unity, C^{n-1}, local control.
  Cannot reach basis value = 1 (no flat-top); no selective interpolation.
- **NURBS**: adds exact primitive reproduction, but via rational weights.
- **PSP spline**: achieves everything NURBS does with **pure polynomials**:

  1. *Flat-top shape preservation*: B = 1 on [a+delta, b-delta] exactly.
  2. *Weights as knot spacings*: w_i = a_{i+1} - a_i (non-rational).
  3. *Extra design dimension delta*: same control polygon, different delta
     => different curve family (Figs. 9b, 11a vs 11b).
  4. *Selective interpolation*: control points with long intervals are
     reproduced exactly while others are only approached (Fig. 11).

Quick start
-----------
>>> import numpy as np
>>> from shape_blend_splines import WeightedControlPolygonPSPSpline
>>> ctrl = np.array([[0,0],[1,1],[2,0],[3,1],[4,0]], dtype=float)
>>> spline = WeightedControlPolygonPSPSpline(ctrl, n=3, delta=0.4)
>>> pts = spline.evaluate(np.linspace(spline.knots[0], spline.knots[-1], 200))

Deprecated API (backward compatible)
--------------------------------------
>>> from shape_blend_splines import PeriodicShapeBlendSpline  # deprecated
>>> from shape_blend_splines.shapes import circle_arc, star_arc
>>> sbs = PeriodicShapeBlendSpline([circle_arc, star_arc], locality=2.0)

Public API
----------
.. autosummary::
   :toctree: api

   PSPSpline
   WeightedControlPolygonPSPSpline
   BlendedPrimitivePSPSpline
   HermitePSPSpline
   PeriodicPSPSpline
   ShapeBlendSpline
   PeriodicShapeBlendSpline
   ControlPointSpline
   ShapeBlender
   blend_two_shapes
   blend_shape_series
   shape_morph
"""

from importlib.metadata import version, PackageNotFoundError

from .curve import (
    PSPSpline,
    WeightedControlPolygonPSPSpline,
    BlendedPrimitivePSPSpline,
    HermitePSPSpline,
    PeriodicPSPSpline,
    # deprecated aliases
    ShapeBlendSpline,
    PeriodicShapeBlendSpline,
    ControlPointSpline,
)
from .blend import ShapeBlender, blend_two_shapes, blend_shape_series, shape_morph
from . import shapes, basis, curve, blend

try:
    __version__ = version("shape_blend_splines")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    # New paper-faithful PSP API
    "PSPSpline",
    "WeightedControlPolygonPSPSpline",
    "BlendedPrimitivePSPSpline",
    "HermitePSPSpline",
    "PeriodicPSPSpline",
    # Deprecated aliases (backward compatible)
    "ShapeBlendSpline",
    "PeriodicShapeBlendSpline",
    "ControlPointSpline",
    "ShapeBlender",
    "blend_two_shapes",
    "blend_shape_series",
    "shape_morph",
    # Sub-modules
    "shapes",
    "basis",
    "curve",
    "blend",
]

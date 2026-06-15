"""
Shape-Blend-Splines
===================

A Python implementation of the non-rational Shape Blend Spline (SBS) technique
described
in:

    Q. Li, "Shape Blend Splines",
    *Computer-Aided Design*, vol. 43, no. 8, pp. 990–1001, 2011.
    DOI: https://doi.org/10.1016/j.cad.2011.01.006
    PII: S0010-4485(11)00008-X

The framework blends whole parametric shapes (circle arcs, ellipses,
rectangles, stars, line segments, control-point curves, …) into complex open
or closed planar geometries using shape-preserving partition-of-unity basis
functions whose locality can be tuned via a single parameter α.

Quick start
-----------
>>> from shape_blend_splines import PeriodicShapeBlendSpline
>>> from shape_blend_splines.shapes import circle_arc, star_arc
>>> import numpy as np
>>> sbs = PeriodicShapeBlendSpline([circle_arc, star_arc], locality=2.0)
>>> pts = sbs.evaluate(np.linspace(0, 1, 300))

Public API
----------
.. autosummary::
   :toctree: api

   ShapeBlendSpline
   PeriodicShapeBlendSpline
   ControlPointSpline
   ShapeBlender
   blend_two_shapes
   blend_shape_series
   shape_morph

"""

from importlib.metadata import version, PackageNotFoundError

from .curve import ShapeBlendSpline, PeriodicShapeBlendSpline, ControlPointSpline
from .blend import ShapeBlender, blend_two_shapes, blend_shape_series, shape_morph
from . import shapes, basis, curve, blend

try:
    __version__ = version("shape_blend_splines")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "ShapeBlendSpline",
    "PeriodicShapeBlendSpline",
    "ControlPointSpline",
    "ShapeBlender",
    "blend_two_shapes",
    "blend_shape_series",
    "shape_morph",
    "shapes",
    "basis",
    "curve",
    "blend",
]

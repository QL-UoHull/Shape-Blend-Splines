"""
tests/test_smoke.py — Lightweight smoke tests for shape_blend_splines.

These tests verify that the package can be imported, that core functions return
results with correct shapes/types, and that the partition-of-unity property
holds for the blend weights.

Run with:
    pytest tests/test_smoke.py -v
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

def test_package_import():
    import shape_blend_splines  # noqa: F401


def test_version_attribute():
    import shape_blend_splines
    assert hasattr(shape_blend_splines, "__version__")


# ---------------------------------------------------------------------------
# basis module
# ---------------------------------------------------------------------------

class TestBasisFunctions:
    def test_recursive_smooth_step_range(self):
        from shape_blend_splines.basis import recursive_smooth_step
        x = np.linspace(-1.0, 2.0, 300)
        y = recursive_smooth_step(x, order=2)
        assert np.all(y >= 0.0)
        assert np.all(y <= 1.0)
        assert np.isclose(y[0], 0.0)
        assert np.isclose(y[-1], 1.0)

    def test_smooth_step_at_orientations(self):
        from shape_blend_splines.basis import smooth_step_at
        t = np.array([-10.0, 10.0])
        up = smooth_step_at(t, centre=0.0, half_width=1.0, rising=True)
        down = smooth_step_at(t, centre=0.0, half_width=1.0, rising=False)
        assert np.allclose(up, [0.0, 1.0])
        assert np.allclose(down, [1.0, 0.0])

    def test_sbs_basis_non_negative(self):
        from shape_blend_splines.basis import sbs_basis
        t = np.linspace(-1.0, 2.0, 300)
        B = sbs_basis(t, a=0.0, b=1.0, order=2)
        assert np.all(B >= 0.0)
        assert np.isclose(B[0], 0.0)
        assert np.isclose(B[-1], 0.0)

    def test_blend_weights_partition_of_unity(self):
        from shape_blend_splines.basis import blend_weights
        t = np.linspace(0.0, 1.0, 100)
        centers = np.array([0.0, 0.5, 1.0])
        W = blend_weights(t, centers, locality=1.0)
        assert W.shape == (3, 100)
        assert np.allclose(W.sum(axis=0), 1.0, atol=1e-10)

    def test_blend_weights_non_negative(self):
        from shape_blend_splines.basis import blend_weights
        t = np.linspace(0.0, 1.0, 50)
        centers = np.linspace(0.0, 1.0, 4)
        W = blend_weights(t, centers, locality=2.0)
        assert np.all(W >= 0.0)

    def test_blend_weights_locality_high(self):
        """High locality → weights are very peaked at their centres."""
        from shape_blend_splines.basis import blend_weights
        centers = np.array([0.0, 0.5, 1.0])
        # At t = 0.5, shape 1 (index 1) should dominate for high locality
        W = blend_weights(np.array([0.5]), centers, locality=10.0)
        assert W[1, 0] > 0.9

    def test_blend_weights_single_shape(self):
        from shape_blend_splines.basis import blend_weights
        t = np.linspace(0.0, 1.0, 20)
        W = blend_weights(t, np.array([0.5]), locality=1.0)
        assert np.allclose(W.sum(axis=0), 1.0)
        assert W.shape == (1, 20)

    def test_blend_weights_periodic_partition_of_unity(self):
        from shape_blend_splines.basis import blend_weights
        t = np.linspace(0.0, 1.0, 120, endpoint=False)
        centers = np.array([0.0, 0.25, 0.5, 0.75])
        W = blend_weights(t, centers, locality=2.0, periodic=True)
        assert W.shape == (4, 120)
        assert np.allclose(W.sum(axis=0), 1.0, atol=1e-10)

    def test_blend_weights_periodic_wraps_edges(self):
        from shape_blend_splines.basis import blend_weights
        centers = np.array([0.0, 0.25, 0.5, 0.75])
        W = blend_weights(np.array([0.0, 1.0]), centers, locality=3.0, periodic=True)
        assert np.allclose(W[:, 0], W[:, 1], atol=1e-10)

    def test_bspline_basis_partition_of_unity(self):
        from shape_blend_splines.basis import uniform_bspline_weights
        t = np.linspace(0.0, 1.0, 50)
        W = uniform_bspline_weights(t, n=6, degree=3)
        assert W.shape == (6, 50)
        assert np.allclose(W.sum(axis=0), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# shapes module
# ---------------------------------------------------------------------------

class TestShapes:
    @pytest.mark.parametrize("fn_name, kwargs", [
        ("line_segment", {}),
        ("circle_arc", {}),
        ("ellipse_arc", {}),
        ("superellipse_arc", {}),
        ("rectangle_arc", {}),
    ])
    def test_shape_output_shape(self, fn_name, kwargs):
        from shape_blend_splines import shapes
        fn = getattr(shapes, fn_name)
        t = np.linspace(0.0, 1.0, 50)
        pts = fn(t, **kwargs)
        assert pts.shape == (50, 2), f"{fn_name} returned shape {pts.shape}"

    def test_polyline(self):
        from shape_blend_splines.shapes import polyline
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        t = np.linspace(0.0, 1.0, 100)
        pts = polyline(t, verts)
        assert pts.shape == (100, 2)

    def test_star_arc(self):
        from shape_blend_splines.shapes import star_arc
        t = np.linspace(0.0, 1.0, 200)
        pts = star_arc(t)
        assert pts.shape == (200, 2)

    def test_from_control_points(self):
        from shape_blend_splines.shapes import from_control_points
        ctrl = np.array([[0, 0], [1, 1], [2, 0], [3, 1]], dtype=float)
        t = np.linspace(0.0, 1.0, 80)
        pts = from_control_points(t, ctrl)
        assert pts.shape == (80, 2)

    def test_from_control_points_too_few(self):
        from shape_blend_splines.shapes import from_control_points
        with pytest.raises(ValueError):
            from_control_points(np.array([0.5]), np.array([[0, 0]]))

    def test_circle_arc_full_circle_start_end(self):
        """t=0 and t=1 should be the same point on a full circle."""
        from shape_blend_splines.shapes import circle_arc
        t = np.array([0.0, 1.0])
        pts = circle_arc(t)
        assert np.allclose(pts[0], pts[1], atol=1e-10)

    def test_line_segment_endpoints(self):
        from shape_blend_splines.shapes import line_segment
        p0, p1 = (1.0, 2.0), (5.0, -3.0)
        pts = line_segment(np.array([0.0, 1.0]), p0=p0, p1=p1)
        assert np.allclose(pts[0], p0)
        assert np.allclose(pts[1], p1)


# ---------------------------------------------------------------------------
# curve module
# ---------------------------------------------------------------------------

class TestShapeBlendSpline:
    def test_basic_construction(self):
        from shape_blend_splines.curve import ShapeBlendSpline
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        sbs = ShapeBlendSpline([circle_arc, ellipse_arc], locality=1.0)
        assert len(sbs) == 2

    def test_periodic_construction(self):
        from shape_blend_splines.curve import PeriodicShapeBlendSpline
        from shape_blend_splines.shapes import circle_arc, ellipse_arc, star_arc
        sbs = PeriodicShapeBlendSpline([circle_arc, ellipse_arc, star_arc], locality=2.0)
        assert len(sbs) == 3
        assert sbs.closed is True

    def test_evaluate_shape(self):
        from shape_blend_splines.curve import ShapeBlendSpline
        from shape_blend_splines.shapes import circle_arc, star_arc
        sbs = ShapeBlendSpline([circle_arc, star_arc])
        t = np.linspace(0.0, 1.0, 100)
        pts = sbs.evaluate(t)
        assert pts.shape == (100, 2)
        assert np.all(np.isfinite(pts))

    def test_single_shape(self):
        """With one shape, the SBS should reproduce that shape exactly."""
        from shape_blend_splines.curve import ShapeBlendSpline
        from shape_blend_splines.shapes import circle_arc
        sbs = ShapeBlendSpline([circle_arc])
        t = np.linspace(0.0, 1.0, 50)
        pts_sbs = sbs.evaluate(t)
        pts_ref = circle_arc(t)
        assert np.allclose(pts_sbs, pts_ref, atol=1e-10)

    def test_weights_at_partition_of_unity(self):
        from shape_blend_splines.curve import ShapeBlendSpline
        from shape_blend_splines.shapes import circle_arc, ellipse_arc, star_arc
        sbs = ShapeBlendSpline([circle_arc, ellipse_arc, star_arc])
        t = np.linspace(0.0, 1.0, 40)
        W = sbs.weights_at(t)
        assert W.shape == (3, 40)
        assert np.allclose(W.sum(axis=0), 1.0)

    def test_no_shapes_raises(self):
        from shape_blend_splines.curve import ShapeBlendSpline
        with pytest.raises(ValueError):
            ShapeBlendSpline([])

    def test_mismatched_centers_raises(self):
        from shape_blend_splines.curve import ShapeBlendSpline
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        with pytest.raises(ValueError):
            ShapeBlendSpline([circle_arc, ellipse_arc], t_centers=[0.5])

    def test_evaluate_shape_j(self):
        from shape_blend_splines.curve import ShapeBlendSpline
        from shape_blend_splines.shapes import circle_arc, star_arc
        sbs = ShapeBlendSpline([circle_arc, star_arc])
        t = np.linspace(0.0, 1.0, 30)
        pts = sbs.evaluate_shape(0, t)
        assert pts.shape == (30, 2)

    def test_control_point_spline(self):
        from shape_blend_splines.curve import ControlPointSpline
        ctrl = np.array([[0, 0], [1, 2], [3, 1], [4, 3]], dtype=float)
        sbs = ControlPointSpline(ctrl, locality=1.5)
        t = np.linspace(0, 1, 100)
        pts = sbs.evaluate(t)
        assert pts.shape == (100, 2)
        assert np.all(np.isfinite(pts))

    def test_periodic_curve_matches_at_domain_edges(self):
        from shape_blend_splines.curve import PeriodicShapeBlendSpline
        from shape_blend_splines.shapes import circle_arc, ellipse_arc, rectangle_arc, star_arc
        sbs = PeriodicShapeBlendSpline(
            [circle_arc, ellipse_arc, rectangle_arc, star_arc],
            locality=3.0,
        )
        pts = sbs.evaluate(np.array([0.0, 1.0]))
        assert np.allclose(pts[0], pts[1], atol=1e-10)

    def test_control_point_spline_closed_loop(self):
        from shape_blend_splines.curve import ControlPointSpline
        ctrl = np.array([[0, 0], [1, 2], [3, 1], [4, 3]], dtype=float)
        sbs = ControlPointSpline(ctrl, locality=1.5, closed=True)
        pts = sbs.evaluate(np.array([0.0, 1.0]))
        assert np.allclose(pts[0], pts[1], atol=1e-10)


# ---------------------------------------------------------------------------
# blend module
# ---------------------------------------------------------------------------

class TestBlend:
    def test_blend_two_shapes(self):
        from shape_blend_splines.blend import blend_two_shapes
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        sbs = blend_two_shapes(circle_arc, ellipse_arc, blend=0.5)
        pts = sbs.evaluate(np.linspace(0, 1, 80))
        assert pts.shape == (80, 2)

    def test_blend_zero_gives_first_shape(self):
        """blend=0 → exactly the first shape in the global weighted blend."""
        from shape_blend_splines.blend import blend_two_shapes
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        sbs = blend_two_shapes(circle_arc, ellipse_arc, blend=0.0)
        t = np.linspace(0.0, 1.0, 100)
        pts_blend = sbs.evaluate(t)
        pts_circle = circle_arc(t)
        assert np.allclose(pts_blend, pts_circle, atol=1e-10)

    def test_blend_two_shapes_warns_when_locality_is_provided(self):
        from shape_blend_splines.blend import blend_two_shapes
        from shape_blend_splines.shapes import circle_arc, ellipse_arc

        with pytest.warns(UserWarning, match="locality parameter is ignored"):
            blender = blend_two_shapes(circle_arc, ellipse_arc, blend=0.25, locality=3.0)

        pts = blender.evaluate(np.linspace(0, 1, 80))
        assert pts.shape == (80, 2)

    def test_blend_shape_series(self):
        from shape_blend_splines.blend import blend_shape_series
        from shape_blend_splines.shapes import circle_arc, ellipse_arc, star_arc
        sbs = blend_shape_series([circle_arc, ellipse_arc, star_arc], locality=2.0)
        pts = sbs.evaluate(np.linspace(0, 1, 100))
        assert pts.shape == (100, 2)

    def test_shape_morph_returns_list(self):
        from shape_blend_splines.blend import shape_morph
        from shape_blend_splines.shapes import circle_arc, star_arc
        frames = shape_morph(circle_arc, star_arc, n_frames=4, n_points=50)
        assert len(frames) == 4
        for f in frames:
            assert f.shape == (50, 2)

    def test_shape_morph_warns_when_locality_is_provided(self):
        from shape_blend_splines.blend import shape_morph
        from shape_blend_splines.shapes import circle_arc, star_arc

        with pytest.warns(UserWarning, match="locality parameter is ignored"):
            frames = shape_morph(circle_arc, star_arc, n_frames=3, locality=3.0, n_points=30)

        assert len(frames) == 3

    def test_shape_blender_weights_sum_to_one(self):
        from shape_blend_splines.blend import ShapeBlender
        from shape_blend_splines.shapes import circle_arc, ellipse_arc, star_arc
        blender = ShapeBlender([circle_arc, ellipse_arc, star_arc],
                               weights=[2.0, 1.0, 0.5])
        w = blender.weights
        assert np.isclose(w.sum(), 1.0)

    def test_shape_blender_evaluate(self):
        from shape_blend_splines.blend import ShapeBlender
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        blender = ShapeBlender([circle_arc, ellipse_arc])
        pts = blender.evaluate(n_points=60)
        assert pts.shape == (60, 2)
        assert np.all(np.isfinite(pts))

    def test_circle_to_ellipse_factory(self):
        from shape_blend_splines.blend import circle_to_ellipse
        sbs = circle_to_ellipse(blend=0.4)
        pts = sbs.evaluate(np.linspace(0, 1, 100))
        assert pts.shape == (100, 2)


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_no_error(self):
        """End-to-end: build SBS, evaluate, no exceptions."""
        from shape_blend_splines import ShapeBlendSpline
        from shape_blend_splines.shapes import (
            circle_arc, ellipse_arc, superellipse_arc,
            rectangle_arc, star_arc,
        )
        shapes = [circle_arc, ellipse_arc, superellipse_arc, rectangle_arc, star_arc]
        sbs = ShapeBlendSpline(shapes, locality=2.0)
        t = np.linspace(0.0, 1.0, 1000)
        pts = sbs.evaluate(t)
        assert pts.shape == (1000, 2)
        assert np.all(np.isfinite(pts))

    def test_locality_monotone_effect(self):
        """Higher locality → weights at centre param closer to 1."""
        from shape_blend_splines.basis import blend_weights
        centers = np.array([0.0, 0.5, 1.0])
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            W = blend_weights(np.array([0.5]), centers, locality=alpha)
            # W[1] is the centre shape weight at t=0.5
            assert W[1, 0] >= W[0, 0], f"Locality monotone failed at alpha={alpha}"


# ---------------------------------------------------------------------------
# Rational reweighting removed
# ---------------------------------------------------------------------------

class TestNonRationalSBS:
    """Regression tests guarding the non-rational SBS path."""

    def test_apply_knot_weights_is_disabled(self):
        from shape_blend_splines.basis import blend_weights, apply_knot_weights
        t = np.linspace(0, 1, 60, endpoint=False)
        centers = np.array([0.0, 0.25, 0.5, 0.75])
        W = blend_weights(t, centers, locality=2.0, periodic=True)
        with pytest.raises(NotImplementedError, match="non-rational"):
            apply_knot_weights(W, [1.0, 1.0, 1.0, 1.0])

    def test_shape_blend_spline_rejects_knot_weights(self):
        from shape_blend_splines import PeriodicShapeBlendSpline
        from shape_blend_splines.shapes import line_segment
        from functools import partial

        corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        edges = [
            partial(line_segment, p0=corners[0], p1=corners[1]),
            partial(line_segment, p0=corners[1], p1=corners[2]),
            partial(line_segment, p0=corners[2], p1=corners[3]),
            partial(line_segment, p0=corners[3], p1=corners[0]),
        ]
        with pytest.raises(ValueError, match="no longer supported"):
            PeriodicShapeBlendSpline(edges, knot_weights=[1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# 4-edge square closed SBS curve construction
# ---------------------------------------------------------------------------

class TestSquareEdgeSBS:
    """Tests for the 4-corner-square → 4-edge-line → closed SBS demo pattern."""

    def _build_square_sbs(self, locality=2.0):
        from shape_blend_splines.shapes import line_segment
        from shape_blend_splines import PeriodicShapeBlendSpline
        corners = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
        centers = np.linspace(0.0, 1.0, len(corners), endpoint=False)
        edges = []
        for j, center in enumerate(centers):
            p0 = tuple(corners[j])
            p1 = tuple(corners[(j + 1) % len(corners)])

            def _edge_shape(t, *, p0=p0, p1=p1, center=center):
                t = np.asarray(t, dtype=float)
                local_t = np.mod(t - center + 0.5, 1.0)
                return line_segment(local_t, p0=p0, p1=p1)

            edges.append(_edge_shape)
        return PeriodicShapeBlendSpline(edges, t_centers=centers, locality=locality)

    def test_square_sbs_is_closed(self):
        """The periodic SBS over 4 edges should close cleanly at the period seam."""
        sbs = self._build_square_sbs()
        assert sbs.closed is True
        t = np.array([0.0, 1.0])
        W = sbs.weights_at(t)
        pts = sbs.evaluate(t)
        assert np.allclose(W[:, 0], W[:, 1], atol=1e-10)
        assert np.allclose(pts[0], pts[1], atol=1e-10)

    def test_square_sbs_output_shape(self):
        """Evaluate returns correct array shape."""
        sbs = self._build_square_sbs()
        pts = sbs.evaluate(np.linspace(0, 1, 400, endpoint=False))
        assert pts.shape == (400, 2)
        assert np.all(np.isfinite(pts))

    def test_square_sbs_partition_of_unity(self):
        """Weights must sum to 1 at every t."""
        sbs = self._build_square_sbs(locality=3.0)
        t = np.linspace(0, 1, 100, endpoint=False)
        W = sbs.weights_at(t)
        assert np.allclose(W.sum(axis=0), 1.0, atol=1e-10)

    def test_square_sbs_passes_through_side_midpoints(self):
        """The 4-edge closed family should align with the side midpoints."""
        sbs = self._build_square_sbs(locality=2.0)
        t = np.array([0.0, 0.25, 0.5, 0.75])
        pts = sbs.evaluate(t)
        expected = np.array([
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        assert np.allclose(pts, expected, atol=5e-3)

    def test_square_sbs_progresses_from_square_like_to_ellipse_like(self):
        """Lower locality should make the 4-point closed curve more ellipse-like."""
        t = np.linspace(0, 1, 1200, endpoint=False)
        square_like = self._build_square_sbs(locality=2.0).evaluate(t)
        intermediate = self._build_square_sbs(locality=0.8).evaluate(t)
        ellipse_like = self._build_square_sbs(locality=0.4).evaluate(t)

        square_spread = np.ptp(np.linalg.norm(square_like, axis=1))
        intermediate_spread = np.ptp(np.linalg.norm(intermediate, axis=1))
        ellipse_spread = np.ptp(np.linalg.norm(ellipse_like, axis=1))

        assert square_spread > intermediate_spread > ellipse_spread
        assert ellipse_spread < 0.02


# ---------------------------------------------------------------------------
# B-spline step-difference identity (numerical verification)
# ---------------------------------------------------------------------------

class TestBSplineStepDifference:
    """
    Verify that each B-spline basis function N_{i,p}(t) can be reproduced as a
    difference of two smooth step functions (the SBS step-difference identity).

    For a clamped uniform knot vector the exact match requires a normalisation
    factor (τ_{i+p+1} - τ_i) that comes from the divided-difference
    representation.  We verify the *normalised* form: after renaming,
    N_{i,p}(t) ∝  S_b(t) - S_a(t)  on each span.

    Here we use a concrete cubic example (p=3, n=5 control points) and compare
    the full basis array from bspline_basis against the step-difference
    reconstruction from smooth_step_at.
    """

    def _step_diff_bspline(self, i: int, p: int, t, knots):
        """
        Approximate N_{i,p}(t) as a normalised SBS step difference.

        Uses smooth_step_at centred at the left and right knot boundaries of
        the basis function's primary support [knots[i], knots[i+p+1]].
        """
        from shape_blend_splines.basis import smooth_step_at
        a = float(knots[i])
        b = float(knots[i + p + 1])
        if np.isclose(a, b):
            return np.zeros_like(t)
        half_width = (b - a) / 2.0
        # Rising step at a (0 → 1 over [a - hw, a + hw])
        S_a = smooth_step_at(t, centre=a, half_width=half_width, rising=True)
        # Rising step at b (0 → 1 over [b - hw, b + hw])
        S_b = smooth_step_at(t, centre=b, half_width=half_width, rising=True)
        raw = np.clip(S_b - S_a, 0.0, None)
        # Normalise to unit peak so we can compare the *shape* of the function.
        peak = raw.max()
        return raw / peak if peak > 1e-14 else raw

    def test_bspline_step_diff_agreement_cubic(self):
        """
        For a uniform clamped B-spline, each basis function N_{i,p}(t) is
        non-negative and supported on [knots[i], knots[i+p+1]].  The
        sbs_basis function constructed at the same support endpoints shares
        the non-negativity property, demonstrating the step-difference
        parallel between B-spline bases and smooth SBS bases.  (The SBS
        smooth step functions have soft tails, so exact hard-support equality
        is not expected — the theoretical connection is through the
        step-difference structure, not numerical equality.)
        """
        from shape_blend_splines.basis import bspline_basis, sbs_basis

        p = 3
        n = 6  # control points
        inner = np.linspace(0.0, 1.0, n - p + 1)
        knots = np.concatenate([np.zeros(p), inner, np.ones(p)])

        t = np.linspace(0.0, 1.0, 400)

        for i in range(n):
            N_exact = bspline_basis(i, p, t, knots)
            a = float(knots[i])
            b = float(knots[i + p + 1])

            # B-spline basis must be non-negative everywhere
            assert np.all(N_exact >= -1e-10), \
                f"bspline_basis N_{{{i},{p}}} should be non-negative"

            if not np.isclose(a, b):
                B = sbs_basis(t, a, b)
                # SBS step-diff basis must also be non-negative everywhere
                assert np.all(B >= -1e-10), \
                    f"sbs_basis is negative for i={i}"

    def test_bspline_step_diff_exact_sbs_basis(self):
        """
        The sbs_basis(t, a, b) function exactly equals the step-difference
        B_{a,b}(t) = S_b(t) - S_a(t) for smooth_step_at calls at a and b.
        """
        from shape_blend_splines.basis import sbs_basis, smooth_step_at
        t = np.linspace(-0.5, 1.5, 300)
        a, b = 0.2, 0.8
        # sbs_basis uses smooth_step_at(..., rising=False) internally
        B = sbs_basis(t, a, b)
        # Reconstruct manually: falling step at a minus falling step at b
        S_a = smooth_step_at(t, centre=a, half_width=(b - a) / 2.0, rising=False)
        S_b = smooth_step_at(t, centre=b, half_width=(b - a) / 2.0, rising=False)
        B_manual = np.clip(S_b - S_a, 0.0, None)
        assert np.allclose(B, B_manual, atol=1e-12), \
            "sbs_basis should equal the manual step-difference reconstruction"

    def test_bspline_basis_partition_of_unity_cubic(self):
        """Sanity check: all B-spline basis functions for n=6, p=3 sum to 1."""
        from shape_blend_splines.basis import uniform_bspline_weights
        t = np.linspace(0.0, 1.0, 100)
        W = uniform_bspline_weights(t, n=6, degree=3)
        assert np.allclose(W.sum(axis=0), 1.0, atol=1e-10)

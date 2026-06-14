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

"""
tests/test_smoke.py — Regression tests for the PSP spline implementation.

Tests verify the paper-faithful mathematics from:
    Q. Li, J. Tian, "Partial shape-preserving splines",
    Computer-Aided Design 43 (2011) 394-409.

Run with:
    pytest tests/test_smoke.py -v
"""

import warnings

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

def test_package_imports():
    import shape_blend_splines as s
    assert hasattr(s, "PSPSpline")
    assert hasattr(s, "WeightedControlPolygonPSPSpline")
    assert hasattr(s, "BlendedPrimitivePSPSpline")
    assert hasattr(s, "HermitePSPSpline")
    assert hasattr(s, "PeriodicPSPSpline")
    # deprecated aliases still importable
    assert hasattr(s, "ShapeBlendSpline")
    assert hasattr(s, "PeriodicShapeBlendSpline")
    assert hasattr(s, "ControlPointSpline")
    assert hasattr(s, "ShapeBlender")


# ---------------------------------------------------------------------------
# H_0: Heaviside step
# ---------------------------------------------------------------------------

class TestHeavisideStep:
    from shape_blend_splines.basis import heaviside_step

    def test_negative(self):
        from shape_blend_splines.basis import heaviside_step
        assert heaviside_step(-1.0) == 0.0
        assert heaviside_step(-0.001) == 0.0

    def test_zero(self):
        from shape_blend_splines.basis import heaviside_step
        assert heaviside_step(0.0) == 0.5

    def test_positive(self):
        from shape_blend_splines.basis import heaviside_step
        assert heaviside_step(1.0) == 1.0
        assert heaviside_step(0.001) == 1.0

    def test_array(self):
        from shape_blend_splines.basis import heaviside_step
        x = np.array([-2.0, 0.0, 2.0])
        h = heaviside_step(x)
        np.testing.assert_array_equal(h, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# H_n: smooth unit step (Eqs. 6-10)
# ---------------------------------------------------------------------------

class TestSmoothUnitStep:

    def test_h1_explicit(self):
        """H_1(x) = 1/2 * ((x+1) H_0(x+1) - (x-1) H_0(x-1))  (Eq. 7)."""
        from shape_blend_splines.basis import smooth_unit_step, heaviside_step
        x = np.linspace(-2, 2, 50)
        h1_eq6 = smooth_unit_step(x, 1)
        h1_eq7 = 0.5 * ((x + 1) * heaviside_step(x + 1) - (x - 1) * heaviside_step(x - 1))
        np.testing.assert_allclose(h1_eq6, h1_eq7, atol=1e-12)

    def test_h2_explicit(self):
        """H_2 via Eq. 8."""
        from shape_blend_splines.basis import smooth_unit_step, heaviside_step
        x = np.linspace(-3, 3, 60)
        h2_eq6 = smooth_unit_step(x, 2)
        h0 = heaviside_step
        h2_eq8 = (1.0 / 8) * (
            (x + 2) ** 2 * h0(x + 2)
            - 2 * x ** 2 * h0(x)
            + (x - 2) ** 2 * h0(x - 2)
        )
        np.testing.assert_allclose(h2_eq6, h2_eq8, atol=1e-12)

    def test_h3_explicit(self):
        """H_3 via Eq. 9."""
        from shape_blend_splines.basis import smooth_unit_step, heaviside_step
        x = np.linspace(-4, 4, 80)
        h3_eq6 = smooth_unit_step(x, 3)
        h0 = heaviside_step
        h3_eq9 = (1.0 / 48) * (
            (x + 3) ** 3 * h0(x + 3)
            - 3 * (x + 1) ** 3 * h0(x + 1)
            + 3 * (x - 1) ** 3 * h0(x - 1)
            - (x - 3) ** 3 * h0(x - 3)
        )
        np.testing.assert_allclose(h3_eq6, h3_eq9, atol=1e-12)

    def test_h3_piecewise_crosscheck(self):
        """Piecewise H_3 from page 396."""
        from shape_blend_splines.basis import smooth_unit_step
        # Piece 1: H_3(x) = 0 for x < -3
        x_neg = np.array([-5.0, -4.0, -3.5, -3.0 - 1e-9])
        np.testing.assert_allclose(smooth_unit_step(x_neg, 3), 0.0, atol=1e-10)

        # Piece 2: (1/48)(3+x)^3 for -3 <= x < -1
        x_p2 = np.linspace(-3, -1, 20, endpoint=False)
        expected = (1.0 / 48) * (3 + x_p2) ** 3
        np.testing.assert_allclose(smooth_unit_step(x_p2, 3), expected, atol=1e-12)

        # Piece 3: (1/24)(12 + 9x - x^3) for -1 <= x < 0
        x_p3 = np.linspace(-1, 0, 20, endpoint=False)
        expected3 = (1.0 / 24) * (12 + 9 * x_p3 - x_p3 ** 3)
        np.testing.assert_allclose(smooth_unit_step(x_p3, 3), expected3, atol=1e-12)

        # Antisymmetry: H_3(x) = 1 - H_3(-x) for x >= 0
        x_pos = np.linspace(0, 5, 50)
        h3_pos = smooth_unit_step(x_pos, 3)
        h3_neg = smooth_unit_step(-x_pos, 3)
        np.testing.assert_allclose(h3_pos + h3_neg, 1.0, atol=1e-12)

    def test_boundary_values(self):
        """H_n = 0 at x=-n, H_n = 1 at x=n (Prop. 4.1)."""
        from shape_blend_splines.basis import smooth_unit_step
        for n in [1, 2, 3, 4]:
            assert smooth_unit_step(-float(n), n) == 0.0
            assert smooth_unit_step(float(n), n) == 1.0

    def test_midpoint(self):
        """H_n(0) = 1/2 (Prop. 4.1)."""
        from shape_blend_splines.basis import smooth_unit_step
        for n in [1, 2, 3, 4]:
            assert abs(smooth_unit_step(0.0, n) - 0.5) < 1e-12

    def test_zero_outside(self):
        """H_n = 0 for x <= -n, = 1 for x >= n."""
        from shape_blend_splines.basis import smooth_unit_step
        for n in [1, 2, 3]:
            assert smooth_unit_step(-float(n) - 1e-6, n) == 0.0
            assert smooth_unit_step(float(n) + 1e-6, n) == 1.0

    def test_antisymmetry(self):
        """H_n(-x) = 1 - H_n(x)."""
        from shape_blend_splines.basis import smooth_unit_step
        for n in [1, 2, 3]:
            x = np.linspace(-float(n), float(n), 50)
            np.testing.assert_allclose(
                smooth_unit_step(-x, n) + smooth_unit_step(x, n), 1.0, atol=1e-12
            )

    def test_monotone(self):
        """H_n is monotone increasing."""
        from shape_blend_splines.basis import smooth_unit_step
        for n in [1, 2, 3]:
            x = np.linspace(-float(n), float(n), 100)
            h = smooth_unit_step(x, n)
            assert np.all(np.diff(h) >= -1e-12)

    def test_nonneg_range(self):
        """H_n in [0, 1]."""
        from shape_blend_splines.basis import smooth_unit_step
        for n in [1, 2, 3]:
            x = np.linspace(-float(n) - 1, float(n) + 1, 200)
            h = smooth_unit_step(x, n)
            assert h.min() >= 0.0 - 1e-12
            assert h.max() <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# H_{n,delta}: scaled smooth unit step (Eq. 11)
# ---------------------------------------------------------------------------

class TestSmoothUnitStepDelta:

    def test_one_at_delta(self):
        """H_{n,delta}(x) = 1 for x >= delta."""
        from shape_blend_splines.basis import smooth_unit_step_delta
        for n in [1, 2, 3]:
            for delta in [0.1, 0.5, 1.0, 2.0]:
                x = np.array([delta, delta + 0.1, delta + 1.0])
                np.testing.assert_allclose(
                    smooth_unit_step_delta(x, n, delta), 1.0, atol=1e-10
                )

    def test_zero_at_neg_delta(self):
        """H_{n,delta}(x) = 0 for x <= -delta."""
        from shape_blend_splines.basis import smooth_unit_step_delta
        for n in [1, 2, 3]:
            for delta in [0.1, 0.5, 1.0]:
                x = np.array([-delta, -delta - 0.1, -delta - 1.0])
                np.testing.assert_allclose(
                    smooth_unit_step_delta(x, n, delta), 0.0, atol=1e-10
                )

    def test_half_at_zero(self):
        """H_{n,delta}(0) = 1/2."""
        from shape_blend_splines.basis import smooth_unit_step_delta
        for n in [1, 2, 3]:
            assert abs(smooth_unit_step_delta(0.0, n, 0.5) - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# PSP basis (Eq. 17)
# ---------------------------------------------------------------------------

class TestPSPBasis:

    def test_nonneg(self):
        """0 <= B <= 1."""
        from shape_blend_splines.basis import psp_basis
        x = np.linspace(0, 8, 300)
        B = psp_basis(x, 2.0, 6.0, 3, 0.5)
        assert B.min() >= 0.0 - 1e-10
        assert B.max() <= 1.0 + 1e-10

    def test_flat_top_exact_one(self):
        """B = 1 exactly on [a+delta, b-delta] when b-a >= 2*delta."""
        from shape_blend_splines.basis import psp_basis
        a, b, delta = 1.0, 5.0, 0.5
        x_flat = np.linspace(a + delta + 1e-6, b - delta - 1e-6, 100)
        B_flat = psp_basis(x_flat, a, b, 3, delta)
        np.testing.assert_allclose(B_flat, 1.0, atol=1e-10)

    def test_flat_top_small_delta(self):
        """Smaller delta → wider flat-top (Fig. 5 delta=0.1)."""
        from shape_blend_splines.basis import psp_basis
        a, b, n = 2.0, 6.0, 3
        x = np.linspace(2, 6, 500)
        B_small = psp_basis(x, a, b, n, 0.1)
        B_large = psp_basis(x, a, b, n, 1.5)
        # Small delta: B=1 on most of [2,6]
        flat_small = np.sum(np.isclose(B_small, 1.0, atol=1e-8))
        flat_large = np.sum(np.isclose(B_large, 1.0, atol=1e-8))
        assert flat_small > flat_large

    def test_additivity(self):
        """B_{[a,c]}(x) + B_{[c,b]}(x) = B_{[a,b]}(x)."""
        from shape_blend_splines.basis import psp_basis
        x = np.linspace(0, 10, 200)
        a, c, b, n, delta = 1.0, 4.0, 7.0, 3, 0.5
        Bac = psp_basis(x, a, c, n, delta)
        Bcb = psp_basis(x, c, b, n, delta)
        Bab = psp_basis(x, a, b, n, delta)
        np.testing.assert_allclose(Bac + Bcb, Bab, atol=1e-12)

    def test_zero_outside_support(self):
        """B = 0 outside [a-delta, b+delta]."""
        from shape_blend_splines.basis import psp_basis
        a, b, delta = 2.0, 6.0, 0.5
        x_out = np.array([a - delta - 0.1, b + delta + 0.1])
        B_out = psp_basis(x_out, a, b, 3, delta)
        np.testing.assert_allclose(B_out, 0.0, atol=1e-10)

    def test_symmetric_around_midpoint(self):
        """Symmetric basis: B(m-x) = B(m+x) for symmetric [a,b]."""
        from shape_blend_splines.basis import psp_basis
        a, b, n, delta = 1.0, 5.0, 3, 0.8
        m = (a + b) / 2
        dx = np.linspace(0, 2, 50)
        B_left = psp_basis(m - dx, a, b, n, delta)
        B_right = psp_basis(m + dx, a, b, n, delta)
        np.testing.assert_allclose(B_left, B_right, atol=1e-12)

    def test_empty_flat_top_when_short(self):
        """When b-a < 2*delta there is no flat-top (bump shape)."""
        from shape_blend_splines.basis import psp_basis, shape_preserving_interval
        a, b, delta = 1.0, 2.0, 0.8
        left, right = shape_preserving_interval(a, b, delta)
        assert left > right  # empty flat-top

        x = np.linspace(a - delta, b + delta, 200)
        B = psp_basis(x, a, b, 3, delta)
        assert B.max() < 1.0 - 1e-6  # never reaches 1

    def test_error_on_invalid_interval(self):
        from shape_blend_splines.basis import psp_basis
        with pytest.raises(ValueError):
            psp_basis(0.0, 5.0, 2.0, 3, 0.5)


# ---------------------------------------------------------------------------
# Asymmetric basis (Eq. 19)
# ---------------------------------------------------------------------------

class TestAsymmetricBasis:

    def test_nonneg_when_condition_holds(self):
        """B >= 0 when 0 <= b-a-delta_b <= delta_a."""
        from shape_blend_splines.basis import psp_basis_asymmetric
        a, b, delta_a, delta_b = 0.0, 2.0, 1.0, 1.0  # symmetric → condition holds
        x = np.linspace(-1, 3, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            B = psp_basis_asymmetric(x, a, b, 3, delta_a, delta_b, warn=False)
        assert B.min() >= -1e-12

    def test_warns_when_condition_fails(self):
        from shape_blend_splines.basis import psp_basis_asymmetric
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            psp_basis_asymmetric(0.0, 0.0, 2.0, 3, 0.1, 3.0, warn=True)
            assert any(issubclass(x.category, UserWarning) for x in w)


# ---------------------------------------------------------------------------
# Partition of unity (Eq. 18)
# ---------------------------------------------------------------------------

class TestPSPPartition:

    def test_partition_sums_to_one_in_design_domain(self):
        """sum_i B_i(x) = 1 for x in [knots[0]+delta, knots[-1]-delta]."""
        from shape_blend_splines.basis import psp_partition
        knots = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
        delta = 0.4
        x = np.linspace(knots[0] + delta + 1e-6, knots[-1] - delta - 1e-6, 200)
        B = psp_partition(x, knots, 3, delta)
        sums = B.sum(axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_partition_nonneg(self):
        from shape_blend_splines.basis import psp_partition
        knots = np.arange(6, dtype=float)
        x = np.linspace(0, 5, 200)
        B = psp_partition(x, knots, 3, 0.4)
        assert B.min() >= -1e-12

    def test_partition_periodic(self):
        """Periodic partition of unity sums to 1."""
        from shape_blend_splines.basis import psp_partition
        knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        delta = 0.3
        x = np.linspace(0, 4, 200)
        B = psp_partition(x, knots, 3, delta, periodic=True)
        sums = B.sum(axis=0)
        # Within design domain should be 1
        mask = (x >= knots[0] + delta) & (x <= knots[-1] - delta)
        np.testing.assert_allclose(sums[mask], 1.0, atol=1e-10)

    def test_output_shape(self):
        from shape_blend_splines.basis import psp_partition
        knots = np.linspace(0, 4, 5)
        x = np.linspace(0, 4, 100)
        B = psp_partition(x, knots, 3, 0.3)
        assert B.shape == (4, 100)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_shape_preserving_interval(self):
        from shape_blend_splines.basis import shape_preserving_interval
        left, right = shape_preserving_interval(1.0, 5.0, 0.5)
        assert abs(left - 1.5) < 1e-12
        assert abs(right - 4.5) < 1e-12

    def test_shape_preserving_interval_empty(self):
        from shape_blend_splines.basis import shape_preserving_interval
        left, right = shape_preserving_interval(1.0, 2.0, 0.8)
        assert left > right  # empty

    def test_knots_from_weights(self):
        from shape_blend_splines.basis import knots_from_weights
        knots = knots_from_weights([1.0, 2.0, 1.5, 0.5])
        expected = np.array([0.0, 1.0, 3.0, 4.5, 5.0])
        np.testing.assert_allclose(knots, expected, atol=1e-12)

    def test_knots_from_weights_equal(self):
        from shape_blend_splines.basis import knots_from_weights
        knots = knots_from_weights([1, 1, 1, 1], a0=2.0)
        expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(knots, expected, atol=1e-12)

    def test_knots_from_weights_negative_raises(self):
        from shape_blend_splines.basis import knots_from_weights
        with pytest.raises(ValueError):
            knots_from_weights([-1.0, 1.0])

    def test_interpolated_indices_none(self):
        """Short intervals: no flat-top."""
        from shape_blend_splines.basis import interpolated_indices
        knots = [0, 0.5, 1.0, 1.5]
        # delta=0.4 → need width >= 0.8; width=0.5 < 0.8
        assert interpolated_indices(knots, 0.4) == []

    def test_interpolated_indices_some(self):
        """Long intervals give flat-tops."""
        from shape_blend_splines.basis import interpolated_indices
        knots = [0, 0.5, 2.5, 3.0, 5.0, 5.5]
        # delta=0.4 → need width >= 0.8
        # widths: 0.5, 2.0, 0.5, 2.0, 0.5
        result = interpolated_indices(knots, 0.4)
        assert result == [1, 3]


# ---------------------------------------------------------------------------
# B-spline special case (page 398)
# ---------------------------------------------------------------------------

class TestBSplineSpecialCase:

    def test_cubic_bspline_equals_psp_basis(self):
        """
        For equal-spaced unit knots, B-spline N_{i,3} equals
        psp_basis(x, i+1.5, i+2.5, n=3, delta=1.5).
        """
        from shape_blend_splines.basis import bspline_basis, psp_basis
        # Build uniform B-spline with integer knots
        n_ctrl = 6
        degree = 3
        knots = np.arange(n_ctrl + degree + 1, dtype=float)
        t = np.linspace(knots[degree], knots[-degree - 1], 300)

        for i in range(n_ctrl):
            Nip = bspline_basis(i, degree, t, knots)
            # PSP equivalent: a_i = i + n/2, interval width = 1, delta = n/2
            n_psp = degree
            delta_psp = n_psp / 2.0  # = 1.5
            a_i = float(i) + delta_psp
            b_i = a_i + 1.0
            Bpsp = psp_basis(t, a_i, b_i, n_psp, delta_psp)
            np.testing.assert_allclose(
                Nip, Bpsp, atol=1e-10,
                err_msg=f"B-spline N_{i},{degree} mismatch with PSP basis"
            )

    def test_bspline_partition_of_unity(self):
        from shape_blend_splines.basis import uniform_bspline_weights
        t = np.linspace(0, 1, 200)
        W = uniform_bspline_weights(t, n=6, degree=3)
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# WeightedControlPolygonPSPSpline
# ---------------------------------------------------------------------------

class TestWeightedControlPolygonPSPSpline:

    def _make_spline(self, weights=None, delta=0.4):
        from shape_blend_splines.curve import WeightedControlPolygonPSPSpline
        ctrl = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]], dtype=float)
        return WeightedControlPolygonPSPSpline(ctrl, weights=weights, n=3, delta=delta)

    def test_evaluate_shape(self):
        spl = self._make_spline()
        t = np.linspace(spl.knots[0], spl.knots[-1], 100)
        pts = spl.evaluate(t)
        assert pts.shape == (100, 2)

    def test_partition_of_unity(self):
        """Basis functions sum to 1 in the design domain."""
        from shape_blend_splines.basis import psp_partition
        spl = self._make_spline()
        delta = spl.delta
        x = np.linspace(spl.knots[0] + delta + 0.01,
                        spl.knots[-1] - delta - 0.01, 200)
        B = psp_partition(x, spl.knots, spl.n, delta)
        np.testing.assert_allclose(B.sum(axis=0), 1.0, atol=1e-10)

    def test_larger_weight_wider_influence(self):
        """Larger weight → wider flat-top → stronger pull."""
        from shape_blend_splines.basis import shape_preserving_interval, knots_from_weights
        weights_eq = [1.0, 1.0, 1.0, 1.0, 1.0]
        weights_big2 = [1.0, 3.0, 1.0, 1.0, 1.0]
        knots_eq = knots_from_weights(weights_eq)
        knots_big2 = knots_from_weights(weights_big2)
        delta = 0.4
        left_eq, right_eq = shape_preserving_interval(knots_eq[1], knots_eq[2], delta)
        left_b2, right_b2 = shape_preserving_interval(knots_big2[1], knots_big2[2], delta)
        # Bigger weight → wider interval → wider flat-top
        assert (right_b2 - left_b2) > (right_eq - left_eq)

    def test_selective_interpolation_long_interval(self):
        """Control point with long interval is interpolated exactly."""
        from shape_blend_splines.curve import WeightedControlPolygonPSPSpline
        # P_1 has a very long interval → should be interpolated
        ctrl = np.array([[0, 0], [1, 2], [3, 1], [4, 0]], dtype=float)
        weights = [0.3, 3.0, 0.3, 0.3]  # P_1 gets long interval
        delta = 0.4
        spl = WeightedControlPolygonPSPSpline(ctrl, weights=weights, n=3, delta=delta)
        interp = spl.interpolated_control_points()
        assert 1 in interp

        # At the midpoint of P_1's interval, curve should equal P_1
        a1, b1 = spl.knots[1], spl.knots[2]
        t_mid = 0.5 * (a1 + b1)
        pt = spl.evaluate(np.array([t_mid]))[0]
        np.testing.assert_allclose(pt, ctrl[1], atol=1e-8)

    def test_selective_interpolation_short_interval_not_interpolated(self):
        """Control point with short interval is NOT interpolated."""
        from shape_blend_splines.curve import WeightedControlPolygonPSPSpline
        ctrl = np.array([[0, 0], [1, 2], [3, 1], [4, 0]], dtype=float)
        weights = [3.0, 0.3, 3.0, 3.0]  # P_1 has short interval
        delta = 0.4
        spl = WeightedControlPolygonPSPSpline(ctrl, weights=weights, n=3, delta=delta)
        interp = spl.interpolated_control_points()
        assert 1 not in interp

        a1, b1 = spl.knots[1], spl.knots[2]
        t_mid = 0.5 * (a1 + b1)
        pt = spl.evaluate(np.array([t_mid]))[0]
        # Should NOT be exactly at P_1
        dist = np.linalg.norm(pt - ctrl[1])
        assert dist > 1e-3

    def test_equal_weights_gives_equal_knots(self):
        from shape_blend_splines.curve import WeightedControlPolygonPSPSpline
        ctrl = np.ones((5, 2))
        spl = WeightedControlPolygonPSPSpline(ctrl, weights=None)
        np.testing.assert_allclose(np.diff(spl.knots), 1.0, atol=1e-12)

    def test_shape_preserving_intervals(self):
        spl = self._make_spline()
        spi = spl.shape_preserving_intervals()
        assert len(spi) == 5


# ---------------------------------------------------------------------------
# BlendedPrimitivePSPSpline
# ---------------------------------------------------------------------------

class TestBlendedPrimitivePSPSpline:

    def test_primitive_reproduced_on_flat_top(self):
        """Primitive is reproduced exactly on its flat-top."""
        from shape_blend_splines.curve import BlendedPrimitivePSPSpline
        from functools import partial
        from shape_blend_splines.shapes import line_segment, circle_arc

        # Build two primitives with long intervals so each has a flat-top
        # Primitive 0: line from (0,0) to (3,0)
        # Primitive 1: circle arc
        def prim0(t):
            return np.column_stack([t, np.zeros_like(t)])

        def prim1(t):
            return np.column_stack([t, np.ones_like(t)])

        knots = [0.0, 3.0, 6.0]
        delta = 1.0
        spl = BlendedPrimitivePSPSpline([prim0, prim1], knots=knots, n=3, delta=delta)

        # On flat-top of prim0: [0+delta, 3-delta] = [1, 2]
        t_ft0 = np.linspace(1.01, 1.99, 20)
        pts = spl.evaluate(t_ft0)
        # prim0(t) = (t, 0); at t in [1,2] spl should be prim0(t)
        np.testing.assert_allclose(pts[:, 1], 0.0, atol=1e-8)

    def test_output_shape(self):
        from shape_blend_splines.curve import BlendedPrimitivePSPSpline

        def p0(t):
            return np.column_stack([t, np.zeros_like(t)])

        def p1(t):
            return np.column_stack([t, np.ones_like(t)])

        spl = BlendedPrimitivePSPSpline([p0, p1], knots=[0, 3, 6], n=3, delta=0.5)
        pts = spl.evaluate(np.linspace(0, 6, 100))
        assert pts.shape == (100, 2)

    def test_rejects_non_callable(self):
        from shape_blend_splines.curve import BlendedPrimitivePSPSpline
        with pytest.raises(ValueError):
            BlendedPrimitivePSPSpline(
                [np.array([0, 0]), np.array([1, 1])],
                knots=[0, 1, 2], n=3, delta=0.3
            )

    def test_edge_flat_top_is_straight_and_nonstalled(self):
        """Edge primitive stays collinear and moves along edge on flat-top."""
        from shape_blend_splines.curve import BlendedPrimitivePSPSpline

        A = np.array([1.0, -0.5])
        B = np.array([4.0, 2.5])

        def left_prim(t):
            t = np.atleast_1d(t)
            return np.column_stack([np.zeros_like(t), np.ones_like(t)])

        def edge_prim(t):
            t = np.atleast_1d(np.asarray(t, dtype=float))
            s = np.clip((t - 1.0) / 4.0, 0.0, 1.0)  # interval [1, 5]
            return (1.0 - s)[:, np.newaxis] * A + s[:, np.newaxis] * B

        def right_prim(t):
            t = np.atleast_1d(t)
            return np.column_stack([6.0 * np.ones_like(t), np.ones_like(t)])

        spl = BlendedPrimitivePSPSpline(
            [left_prim, edge_prim, right_prim],
            knots=[0.0, 1.0, 5.0, 6.0],
            n=3,
            delta=1.0,  # middle flat-top [2,4]
        )

        t_ft = np.linspace(2.0, 4.0, 30)
        pts = spl.evaluate(t_ft)
        chord = B - A
        rel = pts - A
        cross = rel[:, 0] * chord[1] - rel[:, 1] * chord[0]
        np.testing.assert_allclose(cross, 0.0, atol=1e-8)

        # Must move along the edge, not stall at a point.
        assert np.linalg.norm(pts[-1] - pts[0]) > 1e-3

        # Samples lie on the segment A--B.
        lam = (rel @ chord) / (chord @ chord)
        assert np.all(lam >= -1e-10)
        assert np.all(lam <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# HermitePSPSpline (Eq. 23)
# ---------------------------------------------------------------------------

class TestHermitePSPSpline:

    def _make_hermite(self, delta=0.4):
        from shape_blend_splines.curve import HermitePSPSpline
        pts = np.array([[0, 0], [1, 1], [2, 0], [3, 0.5]], dtype=float)
        vel = np.array([[1, 0.5], [0.5, -1], [1, 0.5], [0.5, 0]], dtype=float)
        return HermitePSPSpline(pts, vel, delta=delta)

    def test_position_interpolation(self):
        """P(t_i) = P_i at each node (when flat-top is non-empty)."""
        from shape_blend_splines.curve import HermitePSPSpline
        # Use wider knots to guarantee flat-tops
        pts = np.array([[0, 0], [2, 1], [4, 0]], dtype=float)
        vel = np.array([[1, 0], [0, -1], [1, 0]], dtype=float)
        knots = np.array([0.0, 2.0, 4.0, 6.0])
        delta = 0.6
        herm = HermitePSPSpline(pts, vel, knots=knots, delta=delta)
        t_nodes = herm.t_nodes
        for i, t_i in enumerate(t_nodes):
            pt = herm.evaluate(np.array([t_i]))[0]
            np.testing.assert_allclose(pt, pts[i], atol=1e-6,
                                       err_msg=f"node {i} not interpolated")

    def test_velocity_interpolation(self):
        """P'(t_i) = v_i at each node."""
        from shape_blend_splines.curve import HermitePSPSpline
        pts = np.array([[0, 0], [2, 1], [4, 0]], dtype=float)
        vel = np.array([[1, 0], [0, -1], [1, 0]], dtype=float)
        knots = np.array([0.0, 2.0, 4.0, 6.0])
        delta = 0.6
        herm = HermitePSPSpline(pts, vel, knots=knots, delta=delta)
        t_nodes = herm.t_nodes
        for i, t_i in enumerate(t_nodes):
            dpt = herm.evaluate_deriv(np.array([t_i]))[0]
            np.testing.assert_allclose(dpt, vel[i], atol=1e-4,
                                       err_msg=f"velocity at node {i} mismatch")

    def test_output_shape(self):
        herm = self._make_hermite()
        pts = herm.evaluate(np.linspace(herm.knots[0], herm.knots[-1], 100))
        assert pts.shape == (100, 2)

    def test_interpolated_indices(self):
        from shape_blend_splines.curve import HermitePSPSpline
        pts = np.array([[0, 0], [2, 1]], dtype=float)
        vel = np.array([[1, 0], [0, -1]], dtype=float)
        knots = [0.0, 3.0, 6.0]  # span = 3 > 2*delta = 1.2
        herm = HermitePSPSpline(pts, vel, knots=knots, delta=0.6)
        assert len(herm.interpolated_control_points()) == 2


# ---------------------------------------------------------------------------
# PeriodicPSPSpline
# ---------------------------------------------------------------------------

class TestPeriodicPSPSpline:

    def test_closed_curve_endpoints(self):
        """Periodic PSP curve is continuous at the period boundary."""
        from shape_blend_splines.curve import PeriodicPSPSpline
        ctrl = np.array([[0, 0], [1, 1], [2, 0], [1, -1]], dtype=float)
        knots = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        spl = PeriodicPSPSpline(ctrl, knots=knots, n=3, delta=0.3)
        t0 = spl.evaluate(np.array([0.0]))[0]
        t_end = spl.evaluate(np.array([4.0]))[0]
        np.testing.assert_allclose(t0, t_end, atol=1e-10)


# ---------------------------------------------------------------------------
# Deprecated aliases
# ---------------------------------------------------------------------------

class TestDeprecatedAliases:

    def test_shape_blend_spline_warns(self):
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from shape_blend_splines.curve import ShapeBlendSpline
            ShapeBlendSpline([circle_arc, ellipse_arc], locality=2.0)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) >= 1

    def test_shape_blend_spline_still_works(self):
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from shape_blend_splines.curve import ShapeBlendSpline
            sbs = ShapeBlendSpline([circle_arc, ellipse_arc], locality=2.0)
            pts = sbs.evaluate(np.linspace(0, 1, 100))
            assert pts.shape == (100, 2)

    def test_periodic_shape_blend_spline_warns(self):
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from shape_blend_splines.curve import PeriodicShapeBlendSpline
            PeriodicShapeBlendSpline([circle_arc, ellipse_arc])
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) >= 1

    def test_control_point_spline_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from shape_blend_splines.curve import ControlPointSpline
            ctrl = np.array([[0, 0], [1, 0.5], [2, 0]])
            ControlPointSpline(ctrl)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) >= 1

    def test_apply_knot_weights_raises(self):
        from shape_blend_splines.basis import apply_knot_weights
        with pytest.raises(NotImplementedError):
            apply_knot_weights(np.ones((3, 10)), [1, 1, 1])


# ---------------------------------------------------------------------------
# Legacy blend_weights (deprecated)
# ---------------------------------------------------------------------------

class TestLegacyBlendWeights:

    def test_blend_weights_warns(self):
        """blend_weights emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from shape_blend_splines.basis import blend_weights
            blend_weights(np.linspace(0, 1, 10), [0.25, 0.5, 0.75])
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) >= 1

    def test_blend_weights_sums_to_one(self):
        """Despite deprecation, the old blend still sums to 1."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from shape_blend_splines.basis import blend_weights
            t = np.linspace(0, 1, 100)
            W = blend_weights(t, [0.2, 0.5, 0.8], locality=2.0)
            np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Blend module
# ---------------------------------------------------------------------------

class TestBlend:

    def test_blend_two_shapes(self):
        from shape_blend_splines.blend import blend_two_shapes
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        blender = blend_two_shapes(circle_arc, ellipse_arc, blend=0.3)
        pts = blender.evaluate(np.linspace(0, 1, 100))
        assert pts.shape == (100, 2)

    def test_blend_shape_series(self):
        from shape_blend_splines.blend import blend_shape_series
        from shape_blend_splines.shapes import circle_arc, ellipse_arc, star_arc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sbs = blend_shape_series([circle_arc, ellipse_arc, star_arc], locality=2.0)
        assert len(sbs.shapes) == 3

    def test_shape_morph_returns_list(self):
        from shape_blend_splines.blend import shape_morph
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        frames = shape_morph(circle_arc, ellipse_arc, n_frames=3)
        assert len(frames) == 3
        assert frames[0].shape == (300, 2)

    def test_shape_morph_warns_when_locality_is_provided(self):
        from shape_blend_splines.blend import shape_morph
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shape_morph(circle_arc, ellipse_arc, n_frames=2, locality=3.0)
            assert any(issubclass(x.category, UserWarning) for x in w)

    def test_shape_blender_weights_sum_to_one(self):
        from shape_blend_splines.blend import ShapeBlender
        from shape_blend_splines.shapes import circle_arc, ellipse_arc, star_arc
        blender = ShapeBlender([circle_arc, ellipse_arc, star_arc])
        assert abs(blender.weights.sum() - 1.0) < 1e-12

    def test_shape_blender_evaluate(self):
        from shape_blend_splines.blend import ShapeBlender
        from shape_blend_splines.shapes import circle_arc, ellipse_arc
        blender = ShapeBlender([circle_arc, ellipse_arc])
        pts = blender.evaluate(n_points=100)
        assert pts.shape == (100, 2)

    def test_circle_to_ellipse_factory(self):
        from shape_blend_splines.blend import circle_to_ellipse
        blender = circle_to_ellipse(blend=0.5)
        pts = blender.evaluate(n_points=100)
        assert pts.shape == (100, 2)


# ---------------------------------------------------------------------------
# Shapes module
# ---------------------------------------------------------------------------

class TestShapes:

    def test_sine_wave(self):
        from shape_blend_splines.shapes import sine_wave
        t = np.linspace(0, 1, 50)
        pts = sine_wave(t)
        assert pts.shape == (50, 2)

    def test_helix_2d(self):
        from shape_blend_splines.shapes import helix_2d
        t = np.linspace(0, 1, 50)
        pts = helix_2d(t)
        assert pts.shape == (50, 2)

    def test_line_segment(self):
        from shape_blend_splines.shapes import line_segment
        t = np.array([0.0, 0.5, 1.0])
        pts = line_segment(t, p0=(0, 0), p1=(2, 2))
        np.testing.assert_allclose(pts[0], [0, 0], atol=1e-12)
        np.testing.assert_allclose(pts[-1], [2, 2], atol=1e-12)


# ---------------------------------------------------------------------------
# Integration: Fig. 11 selective interpolation scenario
# ---------------------------------------------------------------------------

class TestSelectiveInterpolation:

    def test_fig11_scenario(self):
        """
        Mimic Fig. 11: 5 control points where P_2 and P_4 (0-indexed) have
        long intervals → interpolated; others have short intervals → not.
        """
        from shape_blend_splines.curve import WeightedControlPolygonPSPSpline
        ctrl = np.array([[0, 0], [0.5, 1], [1, 0.5], [1.5, 1], [2, 0]], dtype=float)
        weights = [0.3, 0.3, 1.5, 0.3, 1.5]  # P_2, P_4 long
        delta = 0.4
        spl = WeightedControlPolygonPSPSpline(ctrl, weights=weights, n=3, delta=delta)
        interp = spl.interpolated_control_points()
        # Widths: 0.3, 0.3, 1.5, 0.3, 1.5; need >= 2*0.4=0.8
        assert 2 in interp
        assert 4 in interp
        assert 0 not in interp
        assert 1 not in interp
        assert 3 not in interp

        # Verify exact interpolation at midpoint of P_2's interval
        a2, b2 = spl.knots[2], spl.knots[3]
        t_mid2 = 0.5 * (a2 + b2)
        pt2 = spl.evaluate(np.array([t_mid2]))[0]
        np.testing.assert_allclose(pt2, ctrl[2], atol=1e-7)

        a4, b4 = spl.knots[4], spl.knots[5]
        t_mid4 = 0.5 * (a4 + b4)
        pt4 = spl.evaluate(np.array([t_mid4]))[0]
        np.testing.assert_allclose(pt4, ctrl[4], atol=1e-7)


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_psp_spline_with_psp_partition(self):
        """PSPSpline with explicit knots evaluates without error."""
        from shape_blend_splines.curve import PSPSpline
        ctrl = np.array([[0, 0], [1, 1], [2, 0]], dtype=float)
        knots = [0.0, 1.0, 2.0, 3.0]
        spl = PSPSpline(ctrl, knots=knots, n=3, delta=0.4)
        pts = spl.evaluate(np.linspace(0, 3, 100))
        assert pts.shape == (100, 2)
        assert not np.any(np.isnan(pts))

    def test_delta_sweep_same_ctrl_different_curves(self):
        """Different delta values give different curves (extra design dimension)."""
        from shape_blend_splines.curve import WeightedControlPolygonPSPSpline
        ctrl = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]], dtype=float)
        pts1 = WeightedControlPolygonPSPSpline(ctrl, n=3, delta=0.2).evaluate(
            np.linspace(0.3, 4.7, 200)
        )
        pts2 = WeightedControlPolygonPSPSpline(ctrl, n=3, delta=0.8).evaluate(
            np.linspace(0.3, 4.7, 200)
        )
        # Different delta → different curves
        assert not np.allclose(pts1, pts2)

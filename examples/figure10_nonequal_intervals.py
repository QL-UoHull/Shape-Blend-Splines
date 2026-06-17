"""
Fig. 10 demos using line-segment primitive blending (Eq. 22).

Li & Tian Fig. 10 is produced by blending *edge primitives* P_i(t), not
constant control points.  Each edge primitive is a parametric line segment on
its own interval, so a non-empty flat-top reproduces a whole straight edge.
This is the key behavior that point blending cannot produce.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape_blend_splines.basis import knots_from_weights
from shape_blend_splines.curve import BlendedPrimitivePSPSpline, PeriodicPSPSpline
from shape_blend_splines.shapes import closed_polygon_edges

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _shade_flat_tops(ax, spline, t, pts, alpha=0.4):
    """Shade flat-top (shape-preserving) curve segments."""
    for left, right in spline.shape_preserving_intervals():
        if left < right:
            mask = (t >= left) & (t <= right)
            if np.count_nonzero(mask) > 2:
                seg = pts[mask]
                tangent = np.gradient(seg, axis=0)
                normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
                norm = np.linalg.norm(normal, axis=1, keepdims=True)
                normal = normal / np.clip(norm, 1e-12, None)
                up = seg + 0.03 * normal
                down = seg - 0.03 * normal
                ribbon = np.vstack([up, down[::-1]])
                ax.fill(
                    ribbon[:, 0], ribbon[:, 1],
                    color="gold", alpha=alpha, linewidth=0
                )


def _open_polyline_edges(vertices, knots):
    """Edge primitives for an open polyline over intervals [k_i, k_{i+1}]."""
    verts = np.asarray(vertices, dtype=float)
    knot_arr = np.asarray(knots, dtype=float)
    edges = []
    for i in range(len(verts) - 1):
        p0, p1 = verts[i], verts[i + 1]
        a, b = knot_arr[i], knot_arr[i + 1]
        span = b - a

        def edge(t, p0=p0, p1=p1, a=a, span=span):
            tt = np.atleast_1d(np.asarray(t, dtype=float))
            s = np.clip((tt - a) / span, 0.0, 1.0)
            return (1.0 - s)[:, np.newaxis] * p0 + s[:, np.newaxis] * p1

        edges.append(edge)
    return edges


def demo_nonequal_intervals():
    """Top+middle row structure: square family via edge-primitive blending."""
    square = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ], dtype=float)

    configs = [
        ("Rounded square", [2.6, 2.6, 2.6, 2.6], 0.40),
        ("Rounded square (tighter)", [2.6, 2.6, 2.6, 2.6], 0.22),
        ("Teardrop / leaf A", [4.0, 1.2, 0.8, 1.2], 0.38),
        ("Teardrop / leaf B", [4.8, 1.0, 0.6, 1.0], 0.44),
        ("Ellipse-like A", [1.0, 1.0, 1.0, 1.0], 0.75),
        ("Ellipse-like B", [1.0, 1.0, 1.0, 1.0], 0.95),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0))
    for ax, (title, weights, delta) in zip(axes.flat, configs):
        knots = knots_from_weights(weights)
        edges = closed_polygon_edges(square, knots=knots)
        spline = PeriodicPSPSpline(edges, knots=knots, n=3, delta=delta)
        t = np.linspace(knots[0], knots[-1], 1600)
        pts = spline.evaluate(t)

        _shade_flat_tops(ax, spline, t, pts)
        ax.plot(pts[:, 0], pts[:, 1], color="steelblue", lw=2.3)
        poly_closed = np.vstack([square, square[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], "o:", color="gray", lw=1.0, ms=4, alpha=0.65)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n$\\delta={delta}$, weights={weights}", fontsize=9)
        ax.grid(alpha=0.16)

    fig.suptitle(
        "Fig. 10 structure (top+middle): square edge primitives blended by PSP basis\n"
        "Gold = flat-top shape-preservation (whole straight edges reproduced on non-empty flat-tops)",
        fontsize=11, y=0.98,
    )
    fig.tight_layout()
    save(fig, "figure10_nonequal_intervals.png")


def demo_square_spiral():
    """Bottom row structure: open square-spiral with delta sweep."""
    spiral = np.array([
        [-0.82, -0.72], [-0.18, -0.72], [-0.18, -0.18], [-0.70, -0.18],
        [-0.70, -0.60], [-0.30, -0.60], [-0.30, -0.30], [-0.56, -0.30],
        [-0.56, -0.47], [-0.40, -0.47], [-0.40, -0.38], [-0.48, -0.38],
    ], dtype=float)
    seg_lengths = np.linalg.norm(np.diff(spiral, axis=0), axis=1)
    weights = np.maximum(seg_lengths, 0.08)
    knots = knots_from_weights(weights)
    edges = _open_polyline_edges(spiral, knots)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    for ax, delta in zip(axes, [0.05, 0.10, 0.18]):
        spline = BlendedPrimitivePSPSpline(edges, knots=knots, n=3, delta=delta)
        t = np.linspace(knots[0], knots[-1], 2200)
        pts = spline.evaluate(t)
        _shade_flat_tops(ax, spline, t, pts, alpha=0.35)
        ax.plot(pts[:, 0], pts[:, 1], color="steelblue", lw=2.2)
        ax.plot(
            spiral[:, 0], spiral[:, 1],
            linestyle=":", color="gray", lw=1.0, marker="o", ms=2.8, alpha=0.7
        )
        ax.set_aspect("equal")
        ax.set_title(rf"$\delta={delta}$", fontsize=10)
        ax.grid(alpha=0.16)

    fig.suptitle(
        "Fig. 10 structure (bottom): nested square-spiral via open edge-primitive PSP blending",
        fontsize=11, y=0.98,
    )
    fig.tight_layout()
    save(fig, "figure10_square_spiral.png")


if __name__ == "__main__":
    print("Running Fig. 10: non-equal-interval PSP demonstrations …")
    demo_nonequal_intervals()
    demo_square_spiral()
    print("Done.")

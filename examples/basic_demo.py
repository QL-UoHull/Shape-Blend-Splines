"""
basic_demo.py — Shape Blend Splines: basic scripted demonstration

Generates and saves several demonstration plots that showcase the SBS
technique without requiring a Jupyter environment.

Run:
    python examples/basic_demo.py

Output:
    demo_blend_circle_to_star.png
    demo_blend_series.png
    demo_morph.png
    demo_weights.png
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering; change to "TkAgg" etc. for GUI
import matplotlib.pyplot as plt
from functools import partial

# ---------------------------------------------------------------------------
# Allow running from the repository root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape_blend_splines import (
    ShapeBlendSpline,
    ShapeBlender,
    blend_two_shapes,
    blend_shape_series,
    shape_morph,
)
from shape_blend_splines.shapes import (
    circle_arc,
    ellipse_arc,
    superellipse_arc,
    rectangle_arc,
    star_arc,
    line_segment,
    from_control_points,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# Demo 1: Blend a circle into a star at different locality settings
# ===========================================================================
def demo_blend_circle_to_star():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    t = np.linspace(0.0, 1.0, 500)

    for ax, alpha in zip(axes, [0.5, 1.0, 3.0]):
        sbs = blend_two_shapes(circle_arc, star_arc, blend=0.5, locality=alpha)
        pts = sbs.evaluate(t)
        c = np.atleast_2d(circle_arc(t))
        s = np.atleast_2d(star_arc(t))
        ax.plot(c[:, 0], c[:, 1], "--", color="gray", lw=1, alpha=0.5, label="Circle")
        ax.plot(s[:, 0], s[:, 1], ":", color="gray", lw=1, alpha=0.5, label="Star")
        ax.plot(pts[:, 0], pts[:, 1], color="steelblue", lw=2, label=f"SBS α={alpha}")
        ax.set_aspect("equal")
        ax.set_title(f"Locality α = {alpha}")
        ax.legend(fontsize=8)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("Demo 1 — Circle ↔ Star blend at different locality values", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "demo_blend_circle_to_star.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ===========================================================================
# Demo 2: Series blend along multiple shapes
# ===========================================================================
def demo_blend_series():
    circle  = partial(circle_arc, center=(0, 0), radius=1.0)
    ellip   = partial(ellipse_arc, center=(0, 0), a=1.5, b=0.7)
    superel = partial(superellipse_arc, center=(0, 0), a=1.2, b=1.2, exponent=4.0)
    rect    = partial(rectangle_arc, center=(0, 0), width=1.8, height=1.4)
    star    = partial(star_arc, center=(0, 0), outer_radius=1.3, inner_radius=0.5)

    shapes = [circle, ellip, superel, rect, star]
    labels = ["Circle", "Ellipse", "Superellipse\n(n=4)", "Rectangle", "Star"]
    t = np.linspace(0.0, 1.0, 600)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, alpha in zip(axes, [1.0, 4.0]):
        sbs = blend_shape_series(shapes, locality=alpha)
        pts = sbs.evaluate(t)

        colors = plt.cm.tab10.colors
        for j, (shape_fn, label) in enumerate(zip(shapes, labels)):
            sp = np.atleast_2d(shape_fn(t))
            ax.plot(sp[:, 0], sp[:, 1], "--", color=colors[j], lw=1,
                    alpha=0.4, label=label)

        ax.plot(pts[:, 0], pts[:, 1], "k-", lw=2.5, label="SBS blend")
        ax.set_aspect("equal")
        ax.set_title(f"5-shape series blend  α = {alpha}")
        ax.legend(fontsize=7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("Demo 2 — Shape Blend Spline: 5-shape series", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "demo_blend_series.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ===========================================================================
# Demo 3: Shape morphing sequence
# ===========================================================================
def demo_morph():
    n_frames = 6
    frames = shape_morph(circle_arc, star_arc, n_frames=n_frames,
                         locality=2.5, n_points=400)

    fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 3))
    betas = np.linspace(0, 1, n_frames)
    for ax, pts, beta in zip(axes, frames, betas):
        ax.plot(pts[:, 0], pts[:, 1], color="steelblue", lw=2)
        ax.set_aspect("equal")
        ax.set_title(f"β = {beta:.2f}")
        ax.axis("off")

    fig.suptitle("Demo 3 — Shape morphing: circle → star", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "demo_morph.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ===========================================================================
# Demo 4: Blend weights visualisation
# ===========================================================================
def demo_weights():
    shapes = [circle_arc, ellipse_arc, star_arc]
    t = np.linspace(0, 1, 400)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, alpha in zip(axes, [0.5, 1.5, 5.0]):
        sbs = blend_shape_series(shapes, locality=alpha)
        W = sbs.weights_at(t)
        labels = ["Circle", "Ellipse", "Star"]
        colors = ["steelblue", "darkorange", "seagreen"]
        for j in range(3):
            ax.plot(t, W[j], color=colors[j], lw=2, label=labels[j])
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("t")
        ax.set_ylabel("Weight W_j(t)")
        ax.set_title(f"Blend weights  α = {alpha}")
        ax.legend(fontsize=9)

    fig.suptitle("Demo 4 — Partition-of-unity blend weights at different α", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "demo_weights.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ===========================================================================
# Demo 5: Control-point–driven curve
# ===========================================================================
def demo_control_points():
    control_pts = np.array([
        [0.0,  0.0],
        [1.0,  1.5],
        [2.5,  0.5],
        [3.5,  2.0],
        [5.0,  0.0],
    ])

    from shape_blend_splines.curve import ControlPointSpline
    sbs = ControlPointSpline(control_pts, locality=2.0)
    t = np.linspace(0, 1, 500)
    pts = sbs.evaluate(t)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pts[:, 0], pts[:, 1], "steelblue", lw=2.5, label="Shape Blend Spline")
    ax.plot(control_pts[:, 0], control_pts[:, 1], "o--", color="tomato",
            lw=1.2, markersize=8, label="Control points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Demo 5 — Smooth curve through control points")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "demo_control_points.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("Running Shape Blend Splines demonstration …")
    demo_blend_circle_to_star()
    demo_blend_series()
    demo_morph()
    demo_weights()
    demo_control_points()
    print("\nAll demos complete.  PNG files saved to:", OUTPUT_DIR)

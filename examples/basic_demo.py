"""
basic_demo.py — package-centric Shape Blend Spline demonstrations.

Run:
    python examples/basic_demo.py

Outputs:
    demo_periodic_cycle.png
    demo_locality_sweep.png
    demo_four_point_closed_progression.png
    demo_open_sequence.png
    demo_periodic_weights.png
    demo_global_vs_local.png
    demo_control_point_modes.png
"""

import os
import sys
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape_blend_splines import (
    ControlPointSpline,
    PeriodicShapeBlendSpline,
    ShapeBlendSpline,
    ShapeBlender,
)
from shape_blend_splines.basis import CUBIC_C2_ORDER
from shape_blend_splines.shapes import (
    circle_arc,
    ellipse_arc,
    from_control_points,
    line_segment,
    rectangle_arc,
    star_arc,
    superellipse_arc,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def closed_demo_shapes():
    shapes = [
        partial(circle_arc, radius=1.00),
        partial(ellipse_arc, a=1.45, b=0.75),
        partial(superellipse_arc, a=1.15, b=1.15, exponent=4.0),
        partial(rectangle_arc, width=2.0, height=1.6),
        partial(star_arc, outer_radius=1.25, inner_radius=0.50),
    ]
    labels = ["Circle", "Ellipse", "Superellipse", "Rectangle", "Star"]
    return shapes, labels


def open_demo_shapes():
    freeform_pts = np.array([
        [-1.7, -0.3],
        [-1.0,  1.1],
        [ 0.3,  0.2],
        [ 1.7,  1.0],
    ])
    shapes = [
        partial(line_segment, p0=(-1.8, -0.55), p1=(1.8, -0.45)),
        partial(circle_arc, center=(0.0, -0.15), radius=1.3,
                theta_start=0.95 * np.pi, theta_end=0.05 * np.pi),
        partial(ellipse_arc, center=(0.2, 0.15), a=1.55, b=0.85,
                theta_start=np.pi, theta_end=2.0 * np.pi),
        partial(from_control_points, control_pts=freeform_pts),
    ]
    labels = ["Line", "Circle arc", "Ellipse arc", "Freeform"]
    return shapes, labels


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def closed_edge_cycle_shapes(corners):
    """
    Build phase-aligned edge-line shapes for a 4-point closed SBS demo.

    Each edge is re-parameterised so its midpoint is sampled when the
    corresponding periodic SBS weight reaches its centre. This produces the
    intended rounded-square → transitional → ellipse-like family.
    """
    corners = np.asarray(corners, dtype=float)
    centers = np.linspace(0.0, 1.0, len(corners), endpoint=False)
    shapes = []
    for j, center in enumerate(centers):
        p0 = tuple(corners[j])
        p1 = tuple(corners[(j + 1) % len(corners)])

        def _edge_shape(t, *, p0=p0, p1=p1, center=center):
            t = np.asarray(t, dtype=float)
            local_t = np.mod(t - center + 0.5, 1.0)
            return line_segment(local_t, p0=p0, p1=p1)

        shapes.append(_edge_shape)
    return shapes, centers


def demo_periodic_cycle():
    shapes, labels = closed_demo_shapes()
    sbs = PeriodicShapeBlendSpline(shapes, locality=3.0)
    t = np.linspace(0.0, 1.0, 900, endpoint=False)
    pts = sbs.evaluate(t)

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    colors = plt.cm.tab10.colors
    for j, (shape_fn, label) in enumerate(zip(shapes, labels)):
        sp = np.atleast_2d(shape_fn(t))
        ax.plot(sp[:, 0], sp[:, 1], "--", lw=1.2, alpha=0.35,
                color=colors[j % len(colors)], label=label)
    ax.plot(pts[:, 0], pts[:, 1], color="black", lw=2.8, label="Closed SBS")
    ax.set_aspect("equal")
    ax.set_title("Closed periodic SBS blending whole primitives")
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save(fig, "demo_periodic_cycle.png")


def demo_locality_sweep():
    shapes, _ = closed_demo_shapes()
    t = np.linspace(0.0, 1.0, 900, endpoint=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, alpha in zip(axes, [0.8, 2.0, 7.0]):
        sbs = PeriodicShapeBlendSpline(shapes, locality=alpha)
        pts = sbs.evaluate(t)
        ax.plot(pts[:, 0], pts[:, 1], color="steelblue", lw=2.6)
        ax.set_aspect("equal")
        ax.set_title(rf"$\alpha = {alpha}$")
        ax.grid(alpha=0.2)
    fig.suptitle("Locality sweep: global smoothing to strong local identity", y=1.02)
    save(fig, "demo_locality_sweep.png")


def demo_four_point_closed_progression():
    """
    4-point periodic closed-curve progression using the paper's cubic piecewise
    C^2 smooth-step basis (order=2), not a quintic smootherstep.
    """
    corners = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ])
    edges, centers = closed_edge_cycle_shapes(corners)
    t = np.linspace(0.0, 1.0, 900, endpoint=False)
    closed_outline = np.vstack([corners, corners[:1]])
    configs = [
        (1.6, "Square-like rounded shape"),
        (0.95, "Intermediate rounded shape"),
        (0.55, "Ellipse-like smooth shape"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (alpha, title) in zip(axes, configs):
        sbs = PeriodicShapeBlendSpline(
            edges,
            t_centers=centers,
            locality=alpha,
            smooth_order=CUBIC_C2_ORDER,
        )
        pts = sbs.evaluate(t)
        ax.plot(closed_outline[:, 0], closed_outline[:, 1], ":", color="gray", alpha=0.45)
        ax.plot(pts[:, 0], pts[:, 1], color="black", lw=2.6)
        ax.scatter(corners[:, 0], corners[:, 1], color="dimgray", s=26, zorder=5)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("y")
    fig.suptitle(
        "Closed SBS from 4 control points (non-rational, cubic piecewise C^2 smooth-step basis)",
        y=1.02,
    )
    save(fig, "demo_four_point_closed_progression.png")


def demo_open_sequence():
    shapes, labels = open_demo_shapes()
    sbs = ShapeBlendSpline(shapes, locality=2.4)
    t = np.linspace(0.0, 1.0, 700)
    pts = sbs.evaluate(t)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = plt.cm.Set2.colors
    for j, (shape_fn, label) in enumerate(zip(shapes, labels)):
        sp = np.atleast_2d(shape_fn(t))
        ax.plot(sp[:, 0], sp[:, 1], "--", lw=1.2, alpha=0.45,
                color=colors[j % len(colors)], label=label)
    ax.plot(pts[:, 0], pts[:, 1], color="black", lw=2.8, label="Open SBS")
    ax.set_aspect("equal")
    ax.set_title("Open SBS sequence: line, arcs, and freeform shape")
    ax.legend(fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    save(fig, "demo_open_sequence.png")


def demo_periodic_weights():
    shapes, labels = closed_demo_shapes()
    t = np.linspace(0.0, 1.0, 700, endpoint=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, alpha in zip(axes, [1.0, 3.0, 8.0]):
        sbs = PeriodicShapeBlendSpline(shapes, locality=alpha)
        W = sbs.weights_at(t)
        for j, label in enumerate(labels):
            ax.plot(t, W[j], lw=2, label=label)
        ax.set_title(rf"$\alpha = {alpha}$")
        ax.set_xlabel("t")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(r"$W_j(t)$")
    axes[0].legend(fontsize=7)
    fig.suptitle("Periodic partition-of-unity weights", y=1.03)
    save(fig, "demo_periodic_weights.png")


def demo_global_vs_local():
    shapes, labels = closed_demo_shapes()
    t = np.linspace(0.0, 1.0, 900, endpoint=False)

    global_blend = ShapeBlender(shapes, weights=[1, 1, 1, 1, 1]).evaluate(t)
    local_blend = PeriodicShapeBlendSpline(shapes, locality=5.5).evaluate(t)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, pts, title in zip(
        axes,
        [global_blend, local_blend],
        ["Global weighted baseline", "Locality-aware closed SBS"],
    ):
        ax.plot(pts[:, 0], pts[:, 1], color="black", lw=2.8)
        for j, shape_fn in enumerate(shapes):
            sp = np.atleast_2d(shape_fn(t))
            ax.plot(sp[:, 0], sp[:, 1], "--", lw=1.0, alpha=0.25,
                    label=labels[j] if title == "Global weighted baseline" else None)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Same components, different blending philosophy", y=1.02)
    save(fig, "demo_global_vs_local.png")


def demo_control_point_modes():
    control_pts = np.array([
        [0.0, 0.0],
        [1.0, 1.8],
        [3.0, 0.8],
        [4.0, 2.4],
        [5.5, 0.2],
    ])
    t_open = np.linspace(0.0, 1.0, 700)
    t_closed = np.linspace(0.0, 1.0, 700, endpoint=False)

    open_curve = ControlPointSpline(control_pts, locality=2.0)
    closed_curve = ControlPointSpline(control_pts, locality=2.0, closed=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(*open_curve.evaluate(t_open).T, color="steelblue", lw=2.5)
    axes[0].plot(control_pts[:, 0], control_pts[:, 1], "o--", color="tomato", lw=1.2)
    axes[0].set_title("Open control-point spline")
    axes[0].set_aspect("equal")

    axes[1].plot(*closed_curve.evaluate(t_closed).T, color="seagreen", lw=2.5)
    closed_pts = np.vstack([control_pts, control_pts[:1]])
    axes[1].plot(closed_pts[:, 0], closed_pts[:, 1], "o--", color="tomato", lw=1.2)
    axes[1].set_title("Closed control-point spline")
    axes[1].set_aspect("equal")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle("Control-point workflows remain available", y=1.02)
    save(fig, "demo_control_point_modes.png")


if __name__ == "__main__":
    print("Running Shape Blend Splines demonstrations …")
    demo_periodic_cycle()
    demo_locality_sweep()
    demo_four_point_closed_progression()
    demo_open_sequence()
    demo_periodic_weights()
    demo_global_vs_local()
    demo_control_point_modes()
    print("All demos complete.")

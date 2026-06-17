"""
hermite_motion.py — Hermite PSP spline: position + velocity interpolation.

Demonstrates Eq. 23 from:
    Q. Li, J. Tian, "Partial shape-preserving splines",
    Computer-Aided Design 43 (2011) 394-409.

The Hermite PSP spline uses the quadratic (n=2) PSP basis to reproduce both
position P_i and velocity v_i at each node t_i exactly, while giving smooth
transitions between nodes.

The second demo shows embedded straight-line segments (flat parts) where the
primitive P_i + (t-t_i)*v_i is reproduced exactly on the flat-top — giving
a trajectory that is straight on those intervals and smoothly curved elsewhere.

Output:
    hermite_trajectory.png
    hermite_straight_segments.png

Run:
    python examples/hermite_motion.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape_blend_splines.curve import HermitePSPSpline

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def demo_trajectory():
    """
    Particle trajectory with position and velocity interpolation.
    Verify P(t_i) = P_i and P'(t_i) = v_i.
    """
    pts = np.array([
        [0.0, 0.0],
        [1.0, 1.5],
        [2.5, 0.5],
        [4.0, 2.0],
        [5.5, 0.0],
    ], dtype=float)

    vel = np.array([
        [1.0,  0.5],   # rightward + slightly up
        [0.8,  0.0],   # rightward
        [0.6, -1.0],   # rightward + sharply down
        [0.8,  0.5],   # rightward + slightly up
        [1.0, -0.5],   # rightward + slightly down
    ], dtype=float)

    # Use wide knots so each node has a flat-top → exact interpolation
    knots = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    delta = 0.7

    herm = HermitePSPSpline(pts, vel, knots=knots, delta=delta)
    t = np.linspace(herm.knots[0], herm.knots[-1], 600)
    curve = herm.evaluate(t)
    deriv = herm.evaluate_deriv(t)

    # --- Verify interpolation numerically ---
    errors_pos = []
    errors_vel = []
    for i, t_i in enumerate(herm.t_nodes):
        p = herm.evaluate(np.array([t_i]))[0]
        v = herm.evaluate_deriv(np.array([t_i]))[0]
        errors_pos.append(np.linalg.norm(p - pts[i]))
        errors_vel.append(np.linalg.norm(v - vel[i]))
    print(f"  Position errors at nodes: {[f'{e:.2e}' for e in errors_pos]}")
    print(f"  Velocity errors at nodes: {[f'{e:.2e}' for e in errors_vel]}")

    fig, (ax_c, ax_v) = plt.subplots(1, 2, figsize=(14, 5.2))

    # --- Curve ---
    ax_c.plot(curve[:, 0], curve[:, 1], color="steelblue", lw=2.5, label="Hermite PSP")
    ax_c.scatter(pts[:, 0], pts[:, 1], s=60, color="tomato", zorder=5, label="Nodes P_i")
    for i, (px, py) in enumerate(pts):
        vx, vy = vel[i] * 0.4
        ax_c.annotate(
            "", xytext=(px, py), xy=(px + vx, py + vy),
            arrowprops=dict(arrowstyle="->", color="seagreen", lw=2.0)
        )
        ax_c.annotate(f"P{i}", (px, py),
                      textcoords="offset points", xytext=(4, 6), fontsize=9)
    # Shade flat-tops
    colors_ft = plt.cm.Pastel1.colors
    for idx, (left, right) in enumerate(herm.shape_preserving_intervals()):
        if left < right:
            mask = (t >= left) & (t <= right)
            if np.any(mask):
                seg = curve[mask]
                ax_c.fill_between(seg[:, 0], seg[:, 1] - 0.06,
                                  seg[:, 1] + 0.06,
                                  color=colors_ft[idx % len(colors_ft)],
                                  alpha=0.5, linewidth=0)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("x")
    ax_c.set_ylabel("y")
    ax_c.set_title(rf"Hermite PSP trajectory  (n=2, $\delta={delta}$)")
    ax_c.legend(fontsize=9)
    ax_c.grid(alpha=0.15)

    # --- Speed plot ---
    speed = np.linalg.norm(deriv, axis=1)
    ax_v.plot(t, speed, color="seagreen", lw=2.0)
    for t_i, (vx, vy) in zip(herm.t_nodes, vel):
        ax_v.axvline(t_i, color="tomato", lw=0.8, ls=":")
        ax_v.scatter([t_i], [np.sqrt(vx**2 + vy**2)], s=50, color="tomato", zorder=5)
    ax_v.set_xlabel("t")
    ax_v.set_ylabel("Speed |P'(t)|")
    ax_v.set_title("Speed profile (dots = prescribed |v_i|)")
    ax_v.grid(alpha=0.2)

    fig.suptitle(
        "Eq. 23: Hermite PSP spline — position + velocity interpolation\n"
        "(Li & Tian 2011)",
        fontsize=11, y=1.02
    )
    save(fig, "hermite_trajectory.png")


def demo_straight_segments():
    """
    Hermite PSP with embedded straight-line flat parts.

    Each Hermite primitive is a straight line through P_i with velocity v_i.
    When the interval is wide enough (flat-top non-empty), the curve travels
    in a straight line along that primitive, then smoothly transitions to the
    next.
    """
    # Design: a zigzag path with straight horizontal segments
    pts = np.array([
        [0.0, 0.0],
        [1.5, 0.0],   # long interval → straight right
        [3.0, 1.0],
        [4.5, 1.0],   # long interval → straight right at y=1
        [6.0, 0.0],
    ], dtype=float)

    # Horizontal velocities for the "straight" segments
    vel = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=float)

    # P1 and P3 have long intervals: they get straight-line flat parts
    knots = np.array([0.0, 2.0, 4.5, 6.5, 9.0, 11.0])
    delta = 0.8

    herm = HermitePSPSpline(pts, vel, knots=knots, delta=delta)
    t = np.linspace(herm.knots[0], herm.knots[-1], 800)
    curve = herm.evaluate(t)

    interp_idx = herm.interpolated_control_points()
    spi = herm.shape_preserving_intervals()

    fig, (ax_b, ax_c) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    # Basis functions
    B = herm.weights_at(t)
    colors = plt.cm.tab10.colors
    for i in range(len(pts)):
        ax_b.plot(t, B[i], color=colors[i % len(colors)], lw=1.8,
                  label=f"B_{i}" + (" ★" if i in interp_idx else ""))
        left, right = spi[i]
        if left < right:
            mask = (t >= left) & (t <= right)
            if np.any(mask):
                ax_b.fill_between(t[mask], 0, B[i][mask],
                                  color=colors[i % len(colors)],
                                  alpha=0.25, linewidth=0)
    ax_b.axhline(1.0, color="k", lw=0.7, ls=":")
    ax_b.set_ylabel("Basis value")
    ax_b.set_title(rf"Basis functions  n=2, $\delta={delta}$  (★ = flat-top / straight part)")
    ax_b.legend(fontsize=8, ncol=3)
    ax_b.set_ylim(-0.05, 1.15)
    ax_b.grid(alpha=0.15)

    # Curve
    ax_c.plot(curve[:, 0], curve[:, 1], color="steelblue", lw=2.5,
              label="Hermite PSP")
    ax_c.scatter(pts[:, 0], pts[:, 1], s=60, color="tomato", zorder=5,
                 label="Nodes P_i")
    # Shade straight segments (flat-tops)
    for i, (left, right) in enumerate(spi):
        if left < right:
            mask = (t >= left) & (t <= right)
            if np.any(mask):
                seg = curve[mask]
                ax_c.fill_between(seg[:, 0], seg[:, 1] - 0.04,
                                  seg[:, 1] + 0.04,
                                  color=colors[i % len(colors)], alpha=0.4, linewidth=0,
                                  label=f"Straight seg {i}" if i == interp_idx[0] else None)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("x")
    ax_c.set_ylabel("y")
    ax_c.set_title("Embedded straight segments with smooth joins")
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.15)

    fig.suptitle(
        "Hermite PSP spline with embedded straight-line flat parts\n"
        "(flat-top regions are exactly straight;  Li & Tian 2011, Eq. 23)",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    save(fig, "hermite_straight_segments.png")


if __name__ == "__main__":
    print("Running Hermite PSP spline demonstrations …")
    demo_trajectory()
    demo_straight_segments()
    print("Done.")

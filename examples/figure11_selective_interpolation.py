"""
figure11_selective_interpolation.py — Selective interpolation via flat-tops.

Reproduces Fig. 11 from:
    Q. Li, J. Tian, "Partial shape-preserving splines",
    Computer-Aided Design 43 (2011) 394-409.

A freeform curve is built from 8 control points where:
  - P0, P7 (endpoints) and P2, P5 have long intervals → exactly interpolated
  - P1, P3, P4, P6 have short intervals → only approached

Two delta values (1.0 and 1.8) are shown as Fig. 11a vs 11b:
  - Same control polygon + weights
  - Different delta → different curve family (extra design dimension)

For each, the PSP basis functions are plotted stacked above the curve.

Output:
    figure11_delta1_basis_and_curve.png
    figure11_delta18_basis_and_curve.png

Run:
    python examples/figure11_selective_interpolation.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape_blend_splines.curve import WeightedControlPolygonPSPSpline

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# Fig. 11 control polygon (8 points, freeform)
CTRL = np.array([
    [0.0,  0.0],   # P0 — long interval (interpolated)
    [0.8,  0.8],   # P1 — short (approached)
    [1.8,  0.4],   # P2 — long (interpolated)
    [2.6,  1.2],   # P3 — short (approached)
    [3.4,  0.2],   # P4 — short (approached)
    [4.4,  1.0],   # P5 — long (interpolated)
    [5.2,  0.3],   # P6 — short (approached)
    [6.0,  0.8],   # P7 — long (interpolated)
], dtype=float)

# Weights: long intervals for P0,P2,P5,P7; short for P1,P3,P4,P6
WEIGHTS = [2.0, 0.4, 2.0, 0.4, 0.4, 2.0, 0.4, 2.0]


def plot_fig11(delta, suffix):
    """Plot the Fig. 11 reconstruction for a given delta."""
    spl = WeightedControlPolygonPSPSpline(CTRL, weights=WEIGHTS, n=3, delta=delta)
    t = np.linspace(spl.knots[0], spl.knots[-1], 800)
    pts = spl.evaluate(t)

    interp_idx = spl.interpolated_control_points()
    spi = spl.shape_preserving_intervals()

    fig, (ax_b, ax_c) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    # ---- Stacked basis functions (Fig. 11 style) ----
    B = spl.weights_at(t)
    colors = cm.tab10.colors
    for i in range(len(CTRL)):
        c = colors[i % len(colors)]
        ax_b.plot(t, B[i], color=c, lw=1.8,
                  label=f"B_{i}" + (" ★" if i in interp_idx else ""))
        # Mark flat-top
        left, right = spi[i]
        if left < right:
            mask = (t >= left) & (t <= right)
            if np.any(mask):
                ax_b.fill_between(t[mask], 0, B[i][mask],
                                  color=c, alpha=0.25, linewidth=0)

    ax_b.axhline(1.0, color="k", lw=0.7, ls=":")
    ax_b.set_ylabel("Basis value")
    ax_b.set_title(
        rf"PSP basis functions  n=3, $\delta={delta}$  (★ = interpolated)"
    )
    ax_b.legend(fontsize=7, ncol=4, loc="upper right")
    ax_b.set_ylim(-0.05, 1.15)
    ax_b.set_xlim(t[0], t[-1])
    ax_b.grid(alpha=0.15)

    # ---- Curve ----
    ax_c.plot(pts[:, 0], pts[:, 1], color="steelblue", lw=2.5, label="PSP curve")

    # Shade flat-tops on curve
    for i, (left, right) in enumerate(spi):
        if left < right:
            mask = (t >= left) & (t <= right)
            if np.any(mask):
                ax_c.fill_between(
                    pts[mask, 0],
                    pts[mask, 1] - 0.03,
                    pts[mask, 1] + 0.03,
                    color=colors[i % len(colors)], alpha=0.4, linewidth=0
                )

    # Draw control polygon
    ax_c.plot(CTRL[:, 0], CTRL[:, 1], "o--", color="gray",
              lw=1.0, ms=6, alpha=0.5, label="Control polygon")

    # Annotate control points
    for i, (px, py) in enumerate(CTRL):
        marker = "★" if i in interp_idx else "·"
        ax_c.annotate(
            f"P{i}{marker}", (px, py),
            textcoords="offset points", xytext=(4, 6),
            fontsize=8, color="dimgray"
        )

    # Mark interpolated control points
    interp_pts = CTRL[interp_idx]
    ax_c.scatter(interp_pts[:, 0], interp_pts[:, 1],
                 s=80, color="tomato", zorder=5, label="Interpolated (★)")

    ax_c.set_aspect("equal")
    ax_c.set_xlabel("x")
    ax_c.set_ylabel("y")
    ax_c.set_title(
        rf"Fig. 11{'a' if delta == 1.0 else 'b'}: selective interpolation,  $\delta={delta}$"
    )
    ax_c.legend(fontsize=8)
    ax_c.grid(alpha=0.15)

    fig.suptitle(
        "Fig. 11: PSP freeform curve with selective interpolation\n"
        "(interpolated points ★ are reproduced exactly; others only approached)\n"
        "Li & Tian 2011",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    save(fig, f"figure11_delta{suffix}_basis_and_curve.png")


def demo_two_delta_comparison():
    """Side-by-side Fig. 11a vs 11b: same control points, different delta."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, delta in zip(axes, [1.0, 1.8]):
        spl = WeightedControlPolygonPSPSpline(CTRL, weights=WEIGHTS, n=3, delta=delta)
        t = np.linspace(spl.knots[0], spl.knots[-1], 800)
        pts = spl.evaluate(t)
        interp_idx = spl.interpolated_control_points()

        ax.plot(pts[:, 0], pts[:, 1], color="steelblue", lw=2.5,
                label="PSP curve")
        ax.plot(CTRL[:, 0], CTRL[:, 1], "o--", color="gray",
                lw=1.0, ms=6, alpha=0.5, label="Control polygon")
        interp_pts = CTRL[interp_idx]
        ax.scatter(interp_pts[:, 0], interp_pts[:, 1],
                   s=80, color="tomato", zorder=5, label="Interpolated ★")
        for i, (px, py) in enumerate(CTRL):
            ax.annotate(f"P{i}", (px, py),
                        textcoords="offset points", xytext=(3, 5), fontsize=8)
        ax.set_aspect("equal")
        ax.set_title(rf"$\delta = {delta}$  (interpolated: P{interp_idx})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)

    fig.suptitle(
        "Fig. 11: Same control polygon, different delta — extra design dimension\n"
        "(Li & Tian 2011)",
        fontsize=11, y=1.02
    )
    save(fig, "figure11_comparison.png")


if __name__ == "__main__":
    print("Running Fig. 11: selective interpolation demos …")
    plot_fig11(1.0, "1")
    plot_fig11(1.8, "18")
    demo_two_delta_comparison()
    print("Done.")

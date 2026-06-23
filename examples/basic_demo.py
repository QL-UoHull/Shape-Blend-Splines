"""
basic_demo.py — PSP spline demonstrations.

Reproduces key figures from:
    Q. Li, J. Tian, "Partial shape-preserving splines",
    Computer-Aided Design 43 (2011) 394-409.

Output images:
    demo_H_n.png           — smooth unit step H_1, H_2, H_3 (Fig. 3 style)
    demo_psp_basis.png     — PSP basis B^{(3)}_{[2,6],delta} for several delta (Fig. 5)
    demo_partition.png     — PSP partition of unity (Fig. 6)
    demo_bspline_special.png  — B-spline = PSP basis special case (page 398)
    demo_fig9.png          — weighted control-polygon curves (Fig. 9)

Run:
    python examples/basic_demo.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape_blend_splines.basis import (
    smooth_unit_step,
    psp_basis,
    psp_partition,
    knots_from_weights,
    bspline_basis,
)
from shape_blend_splines.curve import WeightedControlPolygonPSPSpline, PeriodicPSPSpline

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure: smooth unit steps H_1, H_2, H_3
# ---------------------------------------------------------------------------
def demo_smooth_steps():
    x = np.linspace(-4, 4, 400)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, n in zip(axes, [1, 2, 3]):
        h = smooth_unit_step(x, n)
        ax.plot(x, h, color="steelblue", lw=2.0)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.axvline(0, color="gray", lw=0.6, ls=":")
        ax.set_title(rf"$H_{n}(x)$  (n={n})", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(-4, 4)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"$H_n(x)$")
    fig.suptitle(
        r"Smooth unit steps $H_n(x)$  [Li & Tian 2011, Section 4]", y=1.02
    )
    save(fig, "demo_H_n.png")


# ---------------------------------------------------------------------------
# Figure 5: PSP basis B^{(3)}_{[2,6],delta} for several delta values
# ---------------------------------------------------------------------------
def demo_psp_basis():
    x = np.linspace(0, 8, 500)
    deltas = [0.1, 0.5, 1.0, 1.5, 1.9]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(deltas)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for delta, c in zip(deltas, colors):
        B = psp_basis(x, 2.0, 6.0, 3, delta)
        ax.plot(x, B, color=c, lw=2.0, label=rf"$\delta={delta}$")
        # Shade flat-top
        left, right = 2.0 + delta, 6.0 - delta
        if left < right:
            mask = (x >= left) & (x <= right)
            ax.fill_between(x[mask], 0, B[mask], color=c, alpha=0.15)

    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$B^{(3)}_{[2,6],\delta}(x)$")
    ax.set_title(
        r"PSP cubic basis $B^{(3)}_{[2,6],\delta}$ for various $\delta$"
        "\n(Fig. 5, Li & Tian 2011)"
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.2)
    save(fig, "demo_psp_basis.png")


# ---------------------------------------------------------------------------
# Figure 6: partition of unity over non-uniform knots
# ---------------------------------------------------------------------------
def demo_partition():
    knots = np.array([-5.0, -4.0, 0.0, 2.0, 5.0])
    x = np.linspace(-6, 6, 600)
    B = psp_partition(x, knots, 3, 0.5)
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 4.0))
    for i in range(B.shape[0]):
        ax.plot(x, B[i], color=colors[i % len(colors)], lw=2.0,
                label=rf"$B_{i}$  [{knots[i]:.0f},{knots[i+1]:.0f}]")
    ax.plot(x, B.sum(axis=0), "k--", lw=1.5, label="Sum (POU)")
    for k in knots:
        ax.axvline(k, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("x")
    ax.set_ylabel("Basis value")
    ax.set_title(
        "PSP partition of unity over non-uniform knots\n"
        "(Fig. 6, Li & Tian 2011;  delta=0.5, n=3)"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.2)
    save(fig, "demo_partition.png")


# ---------------------------------------------------------------------------
# B-spline special case (page 398)
# ---------------------------------------------------------------------------
def demo_bspline_special():
    """Show that uniform B-spline equals PSP basis with a_i=i+1.5, delta=1.5."""
    n_ctrl = 5
    degree = 3
    knots_bs = np.arange(n_ctrl + degree + 1, dtype=float)
    t = np.linspace(float(degree), float(n_ctrl), 400)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = plt.cm.tab10.colors
    for i in range(n_ctrl):
        Nip = bspline_basis(i, degree, t, knots_bs)
        delta_psp = degree / 2.0   # = 1.5
        a_i = float(i) + delta_psp
        b_i = a_i + 1.0
        Bpsp = psp_basis(t, a_i, b_i, degree, delta_psp)
        c = colors[i % len(colors)]
        ax.plot(t, Nip, color=c, lw=2.5, label=rf"B-spline $N_{{{i},3}}$")
        ax.plot(t, Bpsp, color=c, lw=1.0, ls="--")

    ax.set_xlabel("t")
    ax.set_ylabel("Basis value")
    ax.set_title(
        "B-spline special case: uniform B-spline = PSP basis\n"
        r"($a_i = i + n/2$, $\delta = n/2$;  dashed = PSP, solid = B-spline)"
        "\n(page 398, Li & Tian 2011)"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.2)
    save(fig, "demo_bspline_special.png")


# ---------------------------------------------------------------------------
# Figure 9: weighted control-polygon design
# ---------------------------------------------------------------------------
def demo_fig9():
    ctrl = np.array([
        [0.0, 0.0],
        [0.5, 1.5],
        [1.5, 2.0],
        [2.5, 1.5],
        [3.0, 0.0],
        [2.5, -1.5],
        [1.5, -2.0],
        [0.5, -1.5],
    ], dtype=float)

    # Fig. 9a: several weight sets at fixed delta
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Panel (a): varying weight sets at fixed delta=0.25 ---
    ax = axes[0]
    ax.plot(np.append(ctrl[:, 0], ctrl[0, 0]),
            np.append(ctrl[:, 1], ctrl[0, 1]),
            "o--", color="gray", lw=1.0, ms=5, alpha=0.5, label="Control polygon")

    weight_sets = [
        (np.ones(len(ctrl)), "equal", "steelblue"),
        ([2, 1, 1, 1, 2, 1, 1, 1], "top/bottom heavy", "tomato"),
        ([1, 2, 1, 2, 1, 2, 1, 2], "alternating", "seagreen"),
    ]
    delta = 0.25
    n = 3
    for w, lbl, c in weight_sets:
        knots = knots_from_weights(np.asarray(w))
        spl = PeriodicPSPSpline(ctrl, knots=knots, n=n, delta=delta)
        t = np.linspace(spl.knots[0], spl.knots[-1], 500)
        pts = spl.evaluate(t)
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.0, label=lbl)

    ax.set_aspect("equal")
    ax.set_title(rf"Fig. 9(a): different weights, $\delta={delta}$, n={n}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    # --- Panel (b): same weights, different delta (extra design dimension) ---
    ax = axes[1]
    ax.plot(np.append(ctrl[:, 0], ctrl[0, 0]),
            np.append(ctrl[:, 1], ctrl[0, 1]),
            "o--", color="gray", lw=1.0, ms=5, alpha=0.5, label="Control polygon")

    w_fixed = [2, 1, 1, 1, 2, 1, 1, 1]
    for delta_v, c in zip([0.1, 0.25, 0.5, 0.9], ["navy", "royalblue", "steelblue", "lightskyblue"]):
        knots = knots_from_weights(np.asarray(w_fixed))
        spl = PeriodicPSPSpline(ctrl, knots=knots, n=n, delta=delta_v)
        t = np.linspace(spl.knots[0], spl.knots[-1], 500)
        pts = spl.evaluate(t)
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.0, label=rf"$\delta={delta_v}$")

    ax.set_aspect("equal")
    ax.set_title(r"Fig. 9(b): same weights, varying $\delta$ (extra design dim.)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)

    fig.suptitle(
        "Fig. 9: Weighted control-polygon PSP curves  (Li & Tian 2011, Eq. 21)",
        fontsize=12, y=1.01
    )
    save(fig, "demo_fig9.png")


if __name__ == "__main__":
    print("Running PSP spline demonstrations …")
    demo_smooth_steps()
    demo_psp_basis()
    demo_partition()
    demo_bspline_special()
    demo_fig9()
    print("All demos complete.")

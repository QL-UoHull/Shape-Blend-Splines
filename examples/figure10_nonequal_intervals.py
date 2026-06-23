"""
figure10_nonequal_intervals.py — PSP splines on non-equal-spaced intervals.

Reproduces the spirit of Fig. 10 from:
    Q. Li, J. Tian, "Partial shape-preserving splines",
    Computer-Aided Design 43 (2011) 394-409.

Key insight: **long intervals create near-straight flat segments and embedded
right-angle corners** while the whole curve stays globally smooth (C^{n-1}).
This is the distinctive capability of PSP splines that B-splines cannot achieve
without sacrificing smoothness.

Output:
    figure10_nonequal_intervals.png
    figure10_square_spiral.png

Run:
    python examples/figure10_nonequal_intervals.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shape_blend_splines.curve import WeightedControlPolygonPSPSpline, PeriodicPSPSpline
from shape_blend_splines.basis import psp_partition, knots_from_weights

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def demo_nonequal_intervals():
    """
    Cubic PSP curves on non-equal spaced intervals.
    Long intervals create near-straight segments; short ones create corners.
    """
    # Control polygon — open zigzag
    ctrl = np.array([
        [0.0, 0.0],
        [1.0, 1.5],
        [2.0, 0.0],
        [3.0, 1.5],
        [4.0, 0.0],
        [5.0, 1.5],
    ], dtype=float)

    n = 3
    delta = 0.3

    # Compare equal vs non-equal weights
    weights_equal = np.ones(len(ctrl))
    weights_long_mid = [0.3, 0.3, 2.0, 2.0, 0.3, 0.3]
    weights_long_ends = [2.0, 0.3, 0.3, 0.3, 0.3, 2.0]
    weights_alternating = [2.0, 0.3, 2.0, 0.3, 2.0, 0.3]

    configs = [
        (weights_equal, "Equal weights\n(uniform PSP)"),
        (weights_long_mid, "Long middle intervals\n(near-straight in middle)"),
        (weights_long_ends, "Long end intervals\n(drawn toward endpoints)"),
        (weights_alternating, "Alternating long/short\n(zigzag emphasis)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (w, title) in zip(axes, configs):
        knots = knots_from_weights(np.asarray(w))
        spl = PeriodicPSPSpline(ctrl, knots=knots, n=n, delta=delta)
        t = np.linspace(spl.knots[0], spl.knots[-1], 600)
        pts = spl.evaluate(t)

        # Shade flat-tops
        for left, right in spl.shape_preserving_intervals():
            if left < right:
                mask = (t >= left) & (t <= right)
                if np.any(mask):
                    ax.fill_between(
                        pts[mask, 0], pts[mask, 1] - 0.05,
                        pts[mask, 1] + 0.05,
                        color="gold", alpha=0.6, linewidth=0
                    )

        ax.plot(pts[:, 0], pts[:, 1], color="steelblue", lw=2.2)
        ax.plot(np.append(ctrl[:, 0], ctrl[0, 0]),
                np.append(ctrl[:, 1], ctrl[0, 1]),
                "o--", color="tomato", lw=1.0, ms=5, alpha=0.7)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.15)

    axes[0].set_ylabel("y")
    fig.suptitle(
        "Fig. 10: Cubic PSP curves on non-equal-spaced intervals\n"
        "(gold = flat-top / shape-preservation region;  Li & Tian 2011)",
        fontsize=11, y=1.02
    )
    save(fig, "figure10_nonequal_intervals.png")


def demo_square_spiral():
    """
    Square-spiral–like closed curve using PSP splines on non-equal intervals.

    Long intervals at corners create near-right-angle turns embedded in an
    otherwise smooth closed curve.  This is impossible with a standard
    B-spline without explicit knot multiplicity (which destroys C^{n-1}).
    """
    # A roughly square control polygon
    R = 2.0
    ctrl = np.array([
        [-R, -R],   # corner 0 (SW)
        [-R/4, -R], # bottom edge midpoint
        [ R,  -R],  # corner 1 (SE)
        [ R, -R/4], # right edge midpoint
        [ R,   R],  # corner 2 (NE)
        [ R/4,  R], # top edge midpoint
        [-R,   R],  # corner 3 (NW)
        [-R,  R/4], # left edge midpoint
    ], dtype=float)

    n = 3
    delta = 0.3

    # Long weights at corners (even indices), short at midpoints
    weights_corner_emphasis = [2.0, 0.3, 2.0, 0.3, 2.0, 0.3, 2.0, 0.3]
    weights_equal = np.ones(len(ctrl))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    for ax, (w, title, lw) in zip(axes, [
        (weights_equal, "Equal weights\n(rounded curve)", "steelblue"),
        (weights_corner_emphasis, "Long corner intervals\n(square-spiral–like)", "tomato"),
        (None, None, None),  # delta sweep
    ]):
        if lw is None:
            # Delta sweep
            for delta_v, c in zip([0.2, 0.5, 0.9], ["navy", "steelblue", "lightblue"]):
                knots = knots_from_weights(weights_corner_emphasis)
                spl = PeriodicPSPSpline(
                    ctrl, knots=knots, n=n, delta=delta_v
                )
                t = np.linspace(spl.knots[0], spl.knots[-1], 600)
                pts = spl.evaluate(t)
                ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.0, label=rf"$\delta={delta_v}$")
            ax.set_title(
                "Same corner weights,\ndifferent delta (extra design dim.)"
            )
            ax.legend(fontsize=8)
        else:
            knots = knots_from_weights(np.asarray(w))
            spl = PeriodicPSPSpline(ctrl, knots=knots, n=n, delta=delta)
            t = np.linspace(spl.knots[0], spl.knots[-1], 600)
            pts = spl.evaluate(t)
            # Shade flat-tops
            for left, right in spl.shape_preserving_intervals():
                if left < right:
                    mask = (t >= left) & (t <= right)
                    if np.any(mask):
                        ax.fill_between(pts[mask, 0], pts[mask, 1] - 0.06,
                                        pts[mask, 1] + 0.06,
                                        color="gold", alpha=0.5, linewidth=0)
            ax.plot(pts[:, 0], pts[:, 1], color=lw, lw=2.2)
            ax.set_title(title)
        ax.plot(np.append(ctrl[:, 0], ctrl[0, 0]),
                np.append(ctrl[:, 1], ctrl[0, 1]),
                "o--", color="gray", lw=0.9, ms=5, alpha=0.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")
    fig.suptitle(
        "Fig. 10: Square-spiral–like PSP curves on non-equal intervals\n"
        "(gold = flat-top shape-preservation;  Li & Tian 2011, Fig. 10)",
        fontsize=11, y=1.02
    )
    save(fig, "figure10_square_spiral.png")


if __name__ == "__main__":
    print("Running Fig. 10: non-equal-interval PSP demonstrations …")
    demo_nonequal_intervals()
    demo_square_spiral()
    print("Done.")

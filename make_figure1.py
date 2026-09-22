#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figure1.py -- reproduces Figure 1 of the manuscript (synthetic
validation, Panels A and B).

NOTE: the figure is schematic; the z_j profiles and the position-sensitivity
fractions are the canonical values annotated in the manuscript, reproduced
here exactly.

Output: figure1.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(out="figure1.pdf"):
    bases = ["Haar", "db2", "db4", "db8", "Shannon"]
    z4_vals = [22.9, 39.9, 53.8, 62.6, 70.0]
    pos_sensitivity = [1.00, np.nan, 0.38, 0.24, 0.05]

    colors = ["#d6604d", "#f4a582", "#92c5de", "#4393c3", "#053061"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Synthetic validation --- Theorem 1 (manuscript Sec. 5)\n"
        "Single-band perturbation in $B_4$, $z_4^{id}=70$",
        fontsize=10, fontweight="bold",
    )

    # Panel A: z_j profiles per basis (schematic)
    J = 8
    j_levels = np.arange(1, J + 1)

    def profile(z4, spread=1.5):
        out = []
        for j in j_levels:
            if j < 4:
                out.append(z4 * np.exp(-spread * (4 - j) ** 2 / 8))
            elif j == 4:
                out.append(z4)
            else:
                out.append(z4 * np.exp(-spread * (j - 4) ** 2 / 5))
        return out

    for base, z4, col in zip(bases, z4_vals, colors):
        prof = profile(z4)
        ax1.plot(j_levels, prof, "o-", color=col, lw=1.8, ms=6, label=base)
        ax1.annotate(f"{z4:.1f}", (4, z4), textcoords="offset points",
                     xytext=(5, 2), fontsize=8, color=col)

    ax1.axvline(4, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax1.set_xlabel("Decomposition level $j$ (1 = finest)", fontsize=9)
    ax1.set_ylabel("$z_j$", fontsize=9)
    ax1.set_title("Panel A --- $z_j$ profiles per basis\n(values annotated at $j=4$)",
                  fontsize=9)
    ax1.legend(fontsize=8)
    ax1.set_xticks(j_levels)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 8.5)

    # Panel B: position sensitivity (fraction z3/z4 > 0.7)
    x = [0, 1, 2, 3]
    labs = ["Haar", "db4", "db8", "Shannon"]
    vals = [1.00, 0.38, 0.24, 0.05]
    cols2 = [colors[0], colors[2], colors[3], colors[4]]
    bars = ax2.bar(x, [v * 100 for v in vals], color=cols2, edgecolor="k",
                   lw=0.7, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5, f"{v*100:.0f}%",
                 ha="center", va="bottom", fontsize=9)
    ax2.axhline(70, color="gray", ls="--", lw=1, label="70% threshold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labs, fontsize=9)
    ax2.set_ylabel("% positions with $z_3/z_4 > 0.7$", fontsize=9)
    ax2.set_title("Panel B --- Position sensitivity\n($\\hat{v}^*$ concentration --- Corollary 3)",
                  fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

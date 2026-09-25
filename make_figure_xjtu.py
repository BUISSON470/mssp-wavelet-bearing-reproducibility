#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figure_xjtu.py -- reproduces the XJTU-SY trajectories figure
(manuscript Fig. "XJTU trajectories", Panel A: alert advance in minutes by
bearing and indicator; Panel B: fraction of remaining life at the z_max alarm).

Reads the per-bearing alert times produced by run_baselines_xjtu.py
(results_baselines_xjtu.csv). The resonant/non-resonant band class (R/N) is the
dataset metadata reported in the manuscript table and is hardcoded here.

Output: figure_xjtu.pdf
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Manuscript metadata (band class at alarm), Table "XJTU-SY alert advance".
V_CLASS = {
    "Bearing1_1": "R", "Bearing1_2": "N", "Bearing1_3": "N", "Bearing1_4": "R",
    "Bearing1_5": "N", "Bearing2_1": "R", "Bearing2_2": "N", "Bearing2_3": "R",
    "Bearing2_4": "R", "Bearing2_5": "R", "Bearing3_1": "N", "Bearing3_2": "R",
    "Bearing3_3": "R", "Bearing3_4": "N",
}


def load(csv_path):
    data = {}
    if not os.path.exists(csv_path):
        raise SystemExit(f"{csv_path} not found -- run run_baselines_xjtu.py first")
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = row["bearing_id"]
            data.setdefault(b, {})[row["method"]] = row
    return data


def main(csv_path="results_baselines_xjtu/results_baselines_xjtu.csv",
         out="figure_xjtu.pdf"):
    data = load(csv_path)
    bearings = sorted(data, key=lambda b: (int(b.split("_")[0][-1]),
                                           int(b.split("_")[1])))
    labels = [b.replace("Bearing", "") for b in bearings]

    def val(b, method, key="alert_time_min"):
        r = data[b].get(method, {})
        v = r.get(key, "") if isinstance(r, dict) else ""
        if v == "" or v is None:
            return np.nan
        return float(v)

    z = [val(b, "z") for b in bearings]
    rms = [val(b, "rms") for b in bearings]
    kurt = [val(b, "kurtosis") for b in bearings]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(bearings))
    width = 0.28

    # Panel A: alert advance (minutes), grouped bars, R/N colored background
    for i, b in enumerate(bearings):
        col = "#e7f3e3" if V_CLASS[b] == "R" else "#fbe4e4"
        ax1.axvspan(i - 0.5, i + 0.5, color=col, zorder=0)
    ax1.bar(x - width, z, width, label="$z_{\\max}$", color="#333333", zorder=3)
    ax1.bar(x, rms, width, label="RMS", color="#999999", zorder=3)
    ax1.bar(x + width, kurt, width, label="Kurtosis", color="#cccccc", zorder=3)
    # cross for kurtosis-never (2_4)
    for i, b in enumerate(bearings):
        if b == "Bearing2_4":
            ax1.plot(i + width, 0.02 * max(ax1.get_ylim()[1], 1), "kx",
                     ms=10, mew=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Alert advance (minutes)", fontsize=9)
    ax1.set_title("Panel A --- alert advance by bearing and indicator\n"
                  "(green = resonant band R, red = non-resonant N)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel B: remaining life fraction at z_max alarm
    life_frac = [val(b, "z", "life_fraction") for b in bearings]
    frac = [v for v in life_frac if not np.isnan(v)]
    ax2.bar(x, life_frac, width=0.6, color="#4393c3", edgecolor="k", lw=0.5)
    ax2.axhline(np.median(frac), color="k", ls="--", lw=1,
                label=f"median {np.median(frac):.1f}%")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Remaining life at $z_{\\max}$ alarm (%)", fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.set_title("Panel B --- remaining life fraction", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default="results_baselines_xjtu/results_baselines_xjtu.csv")
    ap.add_argument("--out", default="figure_xjtu.pdf")
    args = ap.parse_args()
    main(args.csv, args.out)

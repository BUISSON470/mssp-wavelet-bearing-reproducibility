#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_extended_table1.py -- extends Table 1 of the manuscript over
a grid of levels j and core fractions theta.

CORRECTED VERSION -- uses the EXACT spectral window construction from
make_figure1.py:
  - unilateral grid (0, pi]
  - cascaded DTFT with reduce_omega for dilated frequencies
  - trapezoidal integration over (0, pi], normalised by 1/pi
  - linear core C_theta (not log-scale)

Run:
    python make_extended_table1.py

Outputs (same directory as script):
    extended_table1.csv
    table_extended1.tex
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt


# ---------------------------------------------------------------------------
# Grid: unilateral (0, pi], exactly as in make_figure1.py.
# ---------------------------------------------------------------------------
N_GRID = 2 ** 16
omega = np.linspace(1e-6, np.pi, N_GRID)


def reduce_omega(w):
    """Fold w into [0, pi] using 2pi-periodicity and even symmetry."""
    r = np.mod(w, 2 * np.pi)
    r = np.where(r > np.pi, 2 * np.pi - r, r)
    return r


def dtft_magsq(coeffs, w):
    """|sum_n c_n exp(-i n w)|^2, with w already in [0, pi]."""
    n = np.arange(len(coeffs))
    phase = np.exp(-1j * np.outer(w, n))
    H = phase @ np.asarray(coeffs)
    return np.abs(H) ** 2


def filter_responses(basis, w):
    """Return (|h_hat(w)|^2, |g_hat(w)|^2) for the given basis, w in [0, pi]."""
    if basis.lower() == "shannon":
        h2 = np.where(w <= np.pi / 2, 2.0, 0.0)
        g2 = np.where(w > np.pi / 2, 2.0, 0.0)
        return h2, g2
    wav = pywt.Wavelet(basis)
    h = wav.dec_lo
    g = wav.dec_hi
    return dtft_magsq(h, w), dtft_magsq(g, w)


def spectral_window(basis, j, omega_grid):
    """
    Compute the normalised spectral window w_j^{(B)}(omega):

        w_j(omega) = |G_j(omega)|^2 / C_j,
        |G_j|^2   = [ prod_{m=0}^{j-2} |h(2^m omega)|^2 ] * |g(2^{j-1} omega)|^2,
        C_j       = (1/pi) integral_0^pi |G_j|^2 domega.

    Identical to window_levels() in make_figure1.py.
    """
    N = len(omega_grid)
    Gsq = np.ones(N)
    for m in range(0, j - 1):
        wr = reduce_omega(2.0 ** m * omega_grid)
        h2, _ = filter_responses(basis, wr)
        Gsq *= h2
    wr_top = reduce_omega(2.0 ** (j - 1) * omega_grid)
    _, g2 = filter_responses(basis, wr_top)
    Gsq *= g2

    C_j = np.trapezoid(Gsq, omega_grid) / np.pi
    if C_j <= 0:
        raise RuntimeError(f"Empty spectral window for basis={basis}, j={j}")
    return Gsq / C_j


def band_limits(j):
    """Return (lo, hi) = (pi/2^j, pi/2^{j-1})."""
    return np.pi / 2 ** j, np.pi / 2 ** (j - 1)


def compute_constants(basis, j, theta, omega_grid=omega):
    """
    Compute s_in, s_adj, s_far, w_minus, w_plus, m+/m- for (basis, j, theta).

    # REVIEWER NOTE -- core definition.
    C_theta is the central fraction (1 - 2*theta) of B_j, defined with
    LINEAR boundaries in frequency:
        lo_core = lo + theta * (hi - lo)
        hi_core = hi - theta * (hi - lo)
    This is the convention consistent with the published Table 1 values.
    A logarithmic core (as in the previous failed version) shifts w_minus
    and w_plus and gives m+/m- inconsistent with the manuscript.
    """
    w_j = spectral_window(basis, j, omega_grid)
    lo, hi = band_limits(j)

    # Band B_j: [pi/2^j, pi/2^{j-1}]
    m_band = (omega_grid >= lo) & (omega_grid <= hi)

    # Adjacent bands B_{j-1} union B_{j+1}
    lo_adj_fine, hi_adj_fine = band_limits(j + 1)
    lo_adj_coar, hi_adj_coar = band_limits(j - 1)
    m_adj = (
        ((omega_grid >= lo_adj_fine) & (omega_grid < lo)) |
        ((omega_grid >  hi)          & (omega_grid <= hi_adj_coar))
    )

    def integral(mask):
        seg = w_j.copy()
        seg[~mask] = 0.0
        return np.trapezoid(seg, omega_grid) / np.pi

    s_in = integral(m_band)
    s_adj = integral(m_adj)
    s_far = max(0.0, 1.0 - s_in - s_adj)

    # Linear core
    lo_core = lo + theta * (hi - lo)
    hi_core = hi - theta * (hi - lo)
    m_core = (omega_grid >= lo_core) & (omega_grid <= hi_core)

    if not np.any(m_core):
        raise RuntimeError(
            f"Empty core mask: basis={basis}, j={j}, theta={theta}, "
            f"lo_core={lo_core:.4f}, hi_core={hi_core:.4f}"
        )

    w_core = w_j[m_core]
    w_minus = float(np.min(w_core))
    w_plus = float(np.max(w_core))
    ratio = np.inf if w_minus <= 0 else w_plus / w_minus

    return {
        "basis": basis, "j": j, "theta": theta,
        "s_in": s_in, "s_adj": s_adj, "s_far": s_far,
        "w_minus": w_minus, "w_plus": w_plus,
        "m_plus_over_m_minus": ratio,
    }


# ---------------------------------------------------------------------------
# Reference values from published Table 1 (j=4, theta=0.15).
# ---------------------------------------------------------------------------
REF = {
    "haar": (0.487, 0.352, 1.60e-1, 1.14),
    "db2":  (0.625, 0.326, 4.83e-2, 1.28),
    "db4":  (0.731, 0.262, 7.53e-3, 1.40),
    "db8":  (0.809, 0.191, 1.58e-4, 1.43),
}


def _fmt_ratio(x):
    return "--" if not np.isfinite(x) else f"{x:.2f}"


def _write_latex(df, path):
    bases_tex = ["haar", "db2", "db4", "db8", "shannon"]
    labels = {"haar": "Haar", "db2": "db2", "db4": "db4",
              "db8": "db8", "shannon": "Shannon"}
    js = [2, 3, 4, 5, 6]
    thetas = [0.10, 0.15, 0.20]

    lines = [
        r"% table_extended1.tex -- generated by make_extended_table1.py",
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{siunitx}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\begin{document}",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        (r"\caption{Stability width $m_+/m_-$ of the spectral window $w_j$"
         r" on the core $C_\theta$, for each basis, level $j$, and core"
         r" fraction $\theta$. Values computed on a $2^{16}$ frequency"
         r" grid, using the same cascade formula as the main text.}"),
        r"\label{tab:extended1}",
        r"\begin{tabular}{ll" + "S[table-format=1.2]" * len(thetas) + "}",
        r"\toprule",
        ("Basis & $j$ & "
         + " & ".join(rf"$\theta={t:.2f}$" for t in thetas)
         + r" \\"),
        r"\midrule",
    ]
    for basis in bases_tex:
        first = True
        for j in js:
            cells = []
            for theta in thetas:
                sub = df[(df.basis == basis) & (df.j == j) & (df.theta == theta)]
                cells.append("--" if sub.empty
                              else _fmt_ratio(sub.iloc[0].m_plus_over_m_minus))
            label = labels[basis] if first else ""
            first = False
            lines.append(rf"{label} & $j={j}$ & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{1em}",
        (r"\begin{minipage}{\linewidth}\footnotesize"
         r"\emph{Spectral masses at $j=4$, $\theta=0.15$"
         r" (reference point of Table~1 of the manuscript):}"
         r"\end{minipage}"),
        "",
        r"\begin{tabular}{lSSS}",
        r"\toprule",
        (r"Basis & {$s_{\mathrm{in}}$} & {$s_{\mathrm{adj}}$}"
         r" & {$s_{\mathrm{far}}$} \\"),
        r"\midrule",
    ]
    for basis in bases_tex:
        sub = df[(df.basis == basis) & (df.j == 4) & (df.theta == 0.15)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.append(
            rf"{labels[basis]} & {r.s_in:.3f} & {r.s_adj:.3f} & {r.s_far:.2e} \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\end{document}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    here = Path(__file__).resolve().parent

    bases = ["haar", "db2", "db4", "db8", "shannon"]
    js = [2, 3, 4, 5, 6]
    thetas = [0.10, 0.15, 0.20]

    rows = []
    for basis in bases:
        for j in js:
            for theta in thetas:
                try:
                    rows.append(compute_constants(basis, j, theta))
                except Exception as exc:
                    print(f"[warn] {basis} j={j} theta={theta}: {exc}",
                          file=sys.stderr)

    df = pd.DataFrame(rows, columns=[
        "basis", "j", "theta",
        "s_in", "s_adj", "s_far",
        "w_minus", "w_plus", "m_plus_over_m_minus",
    ])
    csv_path = here / "extended_table1.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g")
    print(f"-> {csv_path}  ({len(df)} rows)")

    print("\nConsistency check against published Table 1 (j=4, theta=0.15):")
    ok = True
    for basis, (s_in_ref, s_adj_ref, s_far_ref, ratio_ref) in REF.items():
        sub = df[(df.basis == basis) & (df.j == 4) & (df.theta == 0.15)]
        if sub.empty:
            print(f"  MISSING  {basis}")
            ok = False
            continue
        r = sub.iloc[0]
        flag = "OK " if abs(r.m_plus_over_m_minus - ratio_ref) < 0.02 else "!! "
        if flag.strip() == "!!":
            ok = False
        print(
            f"  {flag}{basis:6s}  "
            f"s_in {r.s_in:.3f} (ref {s_in_ref:.3f})  "
            f"s_adj {r.s_adj:.3f} (ref {s_adj_ref:.3f})  "
            f"s_far {r.s_far:.2e} (ref {s_far_ref:.2e})  "
            f"m+/m- {r.m_plus_over_m_minus:.3f} (ref {ratio_ref:.3f})"
        )

    tex_path = here / "table_extended1.tex"
    _write_latex(df, tex_path)
    print(f"\n-> {tex_path}")
    print("Consistency check:", "PASS" if ok else "FAIL -- see flags above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
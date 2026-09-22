#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
window_constants.py -- reproduces Table 1 of the manuscript (window
decomposition constants of the equivalent spectral windows).

Construction (identical to fig_synth.py):
  1. G_j^(B)(omega) via the filter-bank cascade (Daubechies CQF, N=1..8).
  2. w_j^(B)(omega) = |G_j^(B)(omega)|^2 / sum(|G_j^(B)|^2).
  3. B_j   = { omega : pi/2^j <= |omega| <= pi/2^(j-1) }.
  4. C_th  = central (1-2*theta) fraction of B_j (theta removed at each edge).
  5. s_in  = energy of w on B_j;
     s_adj = energy on the two adjacent dyadic bands;
     s_far = 1 - s_in - s_adj.
  6. m_- = inf_{C_th} w,  m_+ = sup_{C_th} w,  m_+/m_- = m_+/m_-.

Output: window_constants.csv
"""

import csv
import numpy as np
from math import comb, pi

J = 4
THETA = 0.15
M = 1 << 16

_idx = np.arange(M)
_omega = 2 * pi * _idx / M
_w = _omega.copy()
_w[_w > pi] -= 2 * pi
_aw = np.abs(_w)  # folded frequency in [0, pi]


def daub_filter(N):
    if N == 1:
        return np.array([1.0, 1.0]) / np.sqrt(2)
    Pc = [comb(N - 1 + k, k) for k in range(N)]
    yroots = np.roots(Pc[::-1])
    c = np.array([1.0 + 0j])
    for _ in range(N):
        c = np.convolve(c, [1.0, 1.0])
    for y0 in yroots:
        b = 2.0 - 4.0 * y0
        d = np.sqrt(b * b - 4.0 + 0j)
        z1, z2 = (b + d) / 2.0, (b - d) / 2.0
        z = z1 if abs(z1) < 1.0 else z2
        c = np.convolve(c, [1.0, -z])
    h = np.real(c)
    return h * np.sqrt(2) / h.sum()


def hipass(h):
    L = len(h)
    return np.array([(-1) ** k * h[L - 1 - k] for k in range(L)])


def window(N, j):
    if N == 0:  # Shannon ideal band-pass
        b = (_aw >= pi / 2 ** j) & (_aw <= pi / 2 ** (j - 1))
        P = np.zeros(M)
        P[b] = 1.0
        return P / P.sum()
    h = daub_filter(N)
    g = hipass(h)
    Hf = np.fft.fft(h, M)
    Gf = np.fft.fft(g, M)
    A = Gf[(_idx * (2 ** (j - 1))) % M].astype(complex)
    for m in range(j - 1):
        A = A * Hf[(_idx * (2 ** m)) % M]
    P = np.abs(A) ** 2
    return P / P.sum()


BASES = [("Haar", 1), ("db2", 2), ("db4", 4), ("db8", 8), ("Shannon", 0)]


def compute(N, j=J, theta=THETA):
    P = window(N, j)
    lo, hi = pi / 2 ** j, pi / 2 ** (j - 1)
    wband = hi - lo
    B = (_aw >= lo) & (_aw <= hi)
    C = (_aw >= lo + theta * wband) & (_aw <= hi - theta * wband)
    adj = ((_aw >= pi / 2 ** (j + 1)) & (_aw < lo)) | \
          ((_aw > hi) & (_aw <= pi / 2 ** (j - 2)))
    s_in = float(P[B].sum())
    s_adj = float(P[adj].sum())
    s_far = float(1.0 - s_in - s_adj)
    core = P[C]
    m_minus = float(core.min())
    m_plus = float(core.max())
    return s_in, s_adj, s_far, m_minus, m_plus, m_plus / m_minus


def main(out="window_constants.csv"):
    rows = []
    for name, N in BASES:
        s_in, s_adj, s_far, m_minus, m_plus, ratio = compute(N)
        note = ("largeur relative borne Theoreme 1"
                if name != "Shannon" else "reference ideale")
        rows.append([name, s_in, s_adj, s_far, ratio, note])
        print(f"{name:8s}  s_in={s_in:.4f}  s_adj={s_adj:.4f}  "
              f"s_far={s_far:.3e}  m_+/m_-={ratio:.4f}")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base", "s_in", "s_adj", "s_far", "m+_sur_m-", "remarque"])
        for name, s_in, s_adj, s_far, ratio, note in rows:
            w.writerow([name, f"{s_in:.3f}", f"{s_adj:.3f}",
                        f"{s_far:.2e}", f"{ratio:.2f}", note])
    print(f"-> {out}")


if __name__ == "__main__":
    main()

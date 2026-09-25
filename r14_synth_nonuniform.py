#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r14_synth_nonuniform.py -- R1.4 experiments.

Verifies the adjacent-ratio bound (eq:adjacent, Theorem 2) under
(a) non-uniform single-band perturbations and (b) multiple adjacent-band
perturbations, on the synthetic reference PSD of sec:synth.

Quantities:
  z3/z4 = sum(w3 * DS)/sum(w4 * DS) * Eref4/Eref3 ,
  Eref_j = sum(w_j * S0),
  bound  = [ min(w3|C)/max(w4|C) * Eref4/Eref3 ,
             max(w3|C)/min(w4|C) * Eref4/Eref3 ]   (C = core of B4),
  point  = mean(w3|C)/mean(w4|C) * Eref4/Eref3      (uniform DS on C).

j* = 4, theta = 0.15, M = 2^16 grid.
"""

import numpy as np
from window_constants import window, _aw, M, THETA

pi = np.pi
J = 4
TH = THETA

# reference PSD (sec:synth)
S0 = 1.0 / (1.0 + (_aw / 0.05) ** 2.4) + 1e-4

# bands
lo4, hi4 = pi / 2 ** 4, pi / 2 ** 3          # B4 = [pi/16, pi/8]
wb4 = hi4 - lo4
C4 = (_aw >= lo4 + TH * wb4) & (_aw <= hi4 - TH * wb4)

lo3, hi3 = pi / 2 ** 3, pi / 2 ** 2          # B3 = [pi/8, pi/4]
wb3 = hi3 - lo3
C3 = (_aw >= lo3 + TH * wb3) & (_aw <= hi3 - TH * wb3)

center4 = (lo4 + hi4) / 2.0

BASES = [("Haar", 1), ("db2", 2), ("db4", 4), ("db8", 8)]


def ratio_z3z4(DS):
    return float((w3 * DS).sum() / (w4 * DS).sum() * Eref4 / Eref3)


print(f"{'base':6s} {'lower':>8s} {'point':>8s} {'upper':>8s} | "
      f"{'uniform':>8s} {'ramp-up':>8s} {'ramp-dn':>8s} {'halfG':>8s} | "
      f"{'multiB3B4':>9s}")
print("-" * 92)

for name, N in BASES:
    w3 = window(N, 3)
    w4 = window(N, 4)
    Eref3 = float((w3 * S0).sum())
    Eref4 = float((w4 * S0).sum())
    w3c = w3[C4]
    w4c = w4[C4]
    lower = w3c.min() / w4c.max() * Eref4 / Eref3
    upper = w3c.max() / w4c.min() * Eref4 / Eref3
    point = w3c.mean() / w4c.mean() * Eref4 / Eref3

    # single-band perturbations on C4
    DS_uniform = np.zeros(M); DS_uniform[C4] = 1.0
    t = (_aw[C4] - lo4 - TH * wb4) / (wb4 * (1 - 2 * TH))
    DS_rampup = np.zeros(M); DS_rampup[C4] = t
    DS_rampdn = np.zeros(M); DS_rampdn[C4] = 1.0 - t
    sig = 0.20 * wb4
    DS_halfg = np.zeros(M); DS_halfg[C4] = np.exp(-0.5 * ((_aw[C4] - center4) / sig) ** 2)

    # multi-band perturbation on C3 U C4 (violates Hypothesis 2)
    DS_multi = np.zeros(M); DS_multi[C3] = 1.0; DS_multi[C4] = 1.0

    r_uni = ratio_z3z4(DS_uniform)
    r_up = ratio_z3z4(DS_rampup)
    r_dn = ratio_z3z4(DS_rampdn)
    r_hg = ratio_z3z4(DS_halfg)
    r_multi = ratio_z3z4(DS_multi)

    print(f"{name:6s} {lower:8.3f} {point:8.3f} {upper:8.3f} | "
          f"{r_uni:8.3f} {r_up:8.3f} {r_dn:8.3f} {r_hg:8.3f} | "
          f"{r_multi:9.3f}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r13_theorem1_scope.py -- R1.3: Theorem 1 (stability bound) under band-edge
and multi-band perturbations, on the synthetic reference PSD.

Notation (manuscript): the normalized spectral window is
w_j(omega) = |G_j|^2 / ((1/2pi) int |G_j|^2), so that (1/2pi) int w_j = 1.
On the M-point grid this is w_j = window_constants.window(N,j) * M.

Theorem 1 bounds  z_j^(B)/zid_j  in  [w_- * gamma, w_+ * gamma]  with
  gamma = (|B_j|/2pi) * (S0_bar_{B_j} / Eref_j),
  Eref_j = <w_j, S0> = (1/M) sum w_j S0,
  w_- , w_+ = extrema of w_j over the core C_theta.

We report the ratio for a center / band-edge / multi-band perturbation.
"""

import numpy as np
from window_constants import window, _aw, M, THETA

pi = np.pi
TH = THETA

S0 = 1.0 / (1.0 + (_aw / 0.05) ** 2.4) + 1e-4

lo4, hi4 = pi / 16, pi / 8
wb4 = hi4 - lo4
B4 = (_aw >= lo4) & (_aw <= hi4)
C4 = (_aw >= lo4 + TH * wb4) & (_aw <= hi4 - TH * wb4)

lo3, hi3 = pi / 8, pi / 4
wb3 = hi3 - lo3
C3 = (_aw >= lo3 + TH * wb3) & (_aw <= hi3 - TH * wb3)

edge_center = lo4 + 0.02
sig = 0.008
DS_center = np.zeros(M); DS_center[C4] = 1.0
DS_edge = np.zeros(M); DS_edge[B4] = np.exp(-0.5 * ((_aw[B4] - edge_center) / sig) ** 2)
DS_multi = np.zeros(M); DS_multi[C3] = 1.0; DS_multi[C4] = 1.0

print(f"{'basis':6s} {'bound_lo':>9s} {'bound_hi':>9s} | {'center':>9s} {'edge':>9s} {'multi':>9s}")
print("-" * 66)

for name, N in [("Haar", 1), ("db2", 2), ("db4", 4), ("db8", 8)]:
    w4 = window(N, 4) * M            # manuscript normalization: (1/M) sum w = 1
    Eref4 = float((w4 * S0).sum() / M)
    Sbar0_B4 = float(S0[B4].mean())
    gamma = (2 * wb4 / (2 * pi)) * Sbar0_B4 / Eref4
    wminus = float(w4[C4].min())
    wplus = float(w4[C4].max())
    blo, bhi = wminus * gamma, wplus * gamma

    def ratio(DS):
        zB = float((w4 * DS).sum()) / float((w4 * S0).sum())
        zid = float(DS[B4].sum()) / float(S0[B4].sum())
        return zB / zid

    print(f"{name:6s} {blo:9.4f} {bhi:9.4f} | {ratio(DS_center):9.4f} "
          f"{ratio(DS_edge):9.4f} {ratio(DS_multi):9.4f}")

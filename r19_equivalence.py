#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r19_equivalence.py -- reproduce the R1.9 numerical-equivalence numbers.

Compares the C firmware Haar DWT (float32, in-place, detector.c/haar_dwt.c)
against the Python oracle (pywt, float64), on the frozen test_vectors.npz
segments. Reports:
  - DWT coefficient relative error (for the "+-0.5%" claim)
  - energy E_obs relative error
  - z-score absolute error (for the "+-0.001" claim)
"""

import numpy as np
import pywt

INV_SQRT2 = np.float32(1.0 / np.sqrt(2.0))

def haar_wavedec_c(x, J):
    """Exact transcription of haar_dwt.c (float32, in-place)."""
    x = x.astype(np.float32)
    N = len(x)
    s = x.copy()
    d = []
    length = N
    for j in range(1, J + 1):
        half = length >> 1
        dj = np.empty(half, dtype=np.float32)
        for i in range(half):
            lo = s[2 * i]
            hi = s[2 * i + 1]
            s[i] = (lo + hi) * INV_SQRT2
            dj[i] = (lo - hi) * INV_SQRT2
        d.append(dj)
        length = half
        if length < 1:
            break
    return d

def haar_wavedec_py(x, J):
    """Python oracle (pywt float64), fine-first [cD1, ..., cDJ]."""
    x = x.astype(np.float64)
    coeffs = pywt.wavedec(x, 'haar', level=J)
    return coeffs[1:][::-1]

def energies_c(d, N, J):
    """Exact transcription of compute_energies() (double sum, float32 out)."""
    E = np.empty(J, dtype=np.float32)
    for j in range(1, J + 1):
        n_j = N >> j
        s = 0.0
        for k in range(n_j):
            v = float(d[j - 1][k])
            s += v * v
        E[j - 1] = np.float32(s / n_j)
    return E

def energies_py(d):
    return np.array([np.mean(dj.astype(np.float64) ** 2) for dj in d])

def main():
    import argparse
    ap = argparse.ArgumentParser(description="R1.9 numerical-equivalence test (C vs Python Haar DWT)")
    ap.add_argument("--vectors", default=r"C:\Users\buisson\RHEOX-autonomous\projet_vibration\CM7\test_vectors.npz",
                    help="path to test_vectors.npz (segments of float32)")
    ap.add_argument("--J", type=int, default=9, help="DWT levels")
    args = ap.parse_args()

    data = np.load(args.vectors)
    segs = data['segments']  # (n_seg, N) float32
    N = segs.shape[1]
    J = args.J

    # E_ref (median over all segments, Python float64) -- self-consistent
    all_E_py = np.array([energies_py(haar_wavedec_py(s, J)) for s in segs])
    E_ref = np.median(all_E_py, axis=0)

    coeff_rel = []   # relative error on DWT coefficients (signed, then abs)
    energy_rel = []  # relative error on E_obs
    z_abs = []       # absolute error on z = (E_obs - E_ref)/E_ref

    for s in segs:
        dc = haar_wavedec_c(s, J)
        dp = haar_wavedec_py(s, J)
        Ec = energies_c(dc, N, J).astype(np.float64)
        Ep = energies_py(dp)

        # coefficient-level relative error (align lengths per level)
        for j in range(J):
            a = dc[j].astype(np.float64)
            b = dp[j].astype(np.float64)
            denom = np.maximum(np.abs(b), 1e-12)
            coeff_rel.append(float(np.max(np.abs(a - b) / denom)))

        # energy relative error
        energy_rel.append(float(np.max(np.abs(Ec - Ep) / np.maximum(np.abs(Ep), 1e-30))))

        # z-score absolute error (shared E_ref)
        zc = (Ec - E_ref) / E_ref
        zp = (Ep - E_ref) / E_ref
        z_abs.append(float(np.max(np.abs(zc - zp))))

    coeff_rel = np.array(coeff_rel)
    energy_rel = np.array(energy_rel)
    z_abs = np.array(z_abs)

    print("=== R1.9 numerical equivalence (C float32 vs Python float64) ===")
    print(f"segments = {segs.shape[0]}  N = {N}  J = {J}")
    print()
    print(f"DWT coefficient rel. error : max = {coeff_rel.max():.3e}   "
          f"mean = {coeff_rel.mean():.3e}")
    print(f"energy E_obs rel. error    : max = {energy_rel.max():.3e}   "
          f"mean = {energy_rel.mean():.3e}")
    print(f"z-score absolute error     : max = {z_abs.max():.3e}   "
          f"mean = {z_abs.mean():.3e}")

if __name__ == "__main__":
    main()

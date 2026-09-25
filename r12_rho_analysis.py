#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r12_rho_analysis.py -- R1.2 : empirical distribution of the switching
statistic rho = sum_j(E_obs_j) / sum_j(E_ref_j), on XJTU-SY (14 bearings,
run-to-failure) and Paderborn (healthy vs faulty, multi-condition).

Outputs the healthy-vs-faulty rho distribution, a calibration of [rho-, rho+]
from the healthy-reference dispersion, and the false-alarm / flag rates.
"""

import os
import sys
import glob
from collections import defaultdict

import numpy as np

# --- XJTU (cache from g2_xjtu_horizon) ---
XJTU_CACHE = r"C:\Users\buisson\resultats_G2\cache"
REF_FRACTION = 0.10
REF_MAX = 30


def robust_band(x, k=3.0):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return med, mad, med - k * 1.4826 * mad, med + k * 1.4826 * mad


def xjtu_rho():
    ref, test = [], []
    for f in sorted(glob.glob(os.path.join(XJTU_CACHE, "*_features.npz"))):
        name = os.path.basename(f).replace("_features.npz", "")
        if name == "Bearing3_5":
            continue
        E = np.load(f)["E_all"]
        n = E.shape[0]
        n_ref = min(max(int(round(REF_FRACTION * n)), 5), REF_MAX)
        E_ref = np.median(E[:n_ref], axis=0)
        rho = E.sum(axis=1) / E_ref.sum()
        ref.extend(rho[:n_ref].tolist())
        test.extend(rho[n_ref * 2:].tolist())
    return np.array(ref), np.array(test)


def main():
    r_ref, r_test = xjtu_rho()
    print("=== XJTU-SY (14 bearings, run-to-failure) ===")
    print(f"healthy ref rho: n={len(r_ref)} median={np.median(r_ref):.4f} "
          f"[{r_ref.min():.4f}, {r_ref.max():.4f}]")
    print(f"faulty test rho: n={len(r_test)} median={np.median(r_test):.4f} "
          f"[{r_test.min():.4f}, {r_test.max():.4f}]")
    med, mad, rho_m, rho_p = robust_band(r_ref)
    fa = np.mean((r_ref < rho_m) | (r_ref > rho_p))
    flag = np.mean((r_test < rho_m) | (r_test > rho_p))
    print(f"calibration [rho-, rho+] = [{rho_m:.4f}, {rho_p:.4f}] "
          f"(median {med:.4f} +/- 3*1.4826*MAD {mad:.4f})")
    print(f"false-alarm (healthy outside): {fa*100:.2f}%")
    print(f"flag rate (faulty outside):     {flag*100:.2f}%")

    # --- Paderborn (healthy K0xx vs faulty KA/KI, per-condition) ---
    sys.path.insert(0, r"C:\Users\buisson")
    from evaluation_paderborn_v2 import energies_cached, collect, J_USED
    PB_ROOT = r"C:\Users\buisson\Paderborn_data"
    PB_CACHE = r"C:\Users\buisson\resultats_PB\cache"
    by_code = collect(PB_ROOT)
    HEALTHY = {"K001", "K002", "K003", "K004", "K005", "K006"}

    def med_sum(path):
        E = energies_cached(path, PB_CACHE)
        return float(np.median(E[:, :J_USED], axis=0).sum())

    # per-condition E_ref from K001+K002
    ref_by_cond = defaultdict(list)
    for code in ("K001", "K002"):
        for p, cond, meas in by_code.get(code, []):
            try:
                ref_by_cond[cond].append(med_sum(p))
            except Exception:
                pass

    print("\n=== Paderborn (healthy reference total energy, per condition) ===")
    for cond in sorted(ref_by_cond):
        v = np.array(ref_by_cond[cond])
        print(f"  {cond:12s} median={np.median(v):.4f} [{v.min():.4f}, {v.max():.4f}]")

    print("\n=== Paderborn (healthy rho, per condition) ===")
    for cond in sorted(ref_by_cond):
        E_ref_c = np.median(ref_by_cond[cond])
        rhos = []
        for code in HEALTHY:
            for p, c, meas in by_code.get(code, []):
                if c == cond:
                    try:
                        rhos.append(med_sum(p) / E_ref_c)
                    except Exception:
                        pass
        rhos = np.array(rhos)
        print(f"  {cond:12s} n={len(rhos)} median={np.median(rhos):.4f} "
              f"[{rhos.min():.4f}, {rhos.max():.4f}]")


if __name__ == "__main__":
    main()

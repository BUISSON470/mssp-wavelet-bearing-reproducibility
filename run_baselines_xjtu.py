#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baselines_xjtu.py -- RMS / kurtosis / FGEK on XJTU-SY, + sign test.

Reuses VERBATIM the feature/threshold/alarm/split code of g2_xjtu_horizon.py
so the RMS and kurtosis baselines are bit-identical to the published
p=0.002 / p=0.73 figures. Only FGEK and the sign test are new.

The threshold function robust_threshold() is PASTED below so the 1.4826
scaling is visible, as requested. It is applied on the VALIDATION window
(REF/VAL), never on TEST.

Output:
  results_baselines_xjtu.csv  [bearing_id, method, alert_time_min, life_fraction, threshold_valid]
  sign_test_xjtu.csv          detector-vs-baseline sign tests (binomtest, greater)
"""

import os
import csv
import argparse
from glob import glob

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import compute_fgek

# ============================================================================
# CONFIG -- verbatim from g2_xjtu_horizon.py
# ============================================================================
FS          = 25600          # Hz -- XJTU-SY
SEG_LEN     = 4096           # points per segment
J_LEVELS    = 12             # full Haar levels (for E_med storage)
J_USED      = 8              # M1 : levels 1..J_USED used for z_max
CHANNEL     = 0              # 0 = horizontal
REF_FRACTION = 0.10          # fraction of life for the reference window
REF_MAX_SNAPS = 30           # cap (minutes) on the reference window
VAL_EQUALS_REF = True        # validation window = same size as REF
N_SIGMA     = 3.0            # M2 : robust threshold factor
PERSIST_K   = 3              # alarm if K consecutive snapshots > threshold
SQRT2 = np.sqrt(2.0)

# REVIEWER NOTE (FGEK frequency coverage at fs=25.6 kHz): dyadic octave bands
# [fs/2^(k+1), fs/2^k] for k=3..8 span ~50 Hz .. 3.2 kHz, covering the XJTU-SY
# fault fundamentals (~100-500 Hz) and the low resonance band. Fixed,
# non-adaptive grid (no per-snapshot search); k-range stated in Methods.
XJTU_FGEK_K_RANGE = range(3, 9)

# REVIEWER NOTE (excluded bearing, manuscript Sec. 6.5): Bearing3_5 is dropped
# from the sign tests because its initial reference window is insufficient
# ("Quatorze des quinze roulements... un roulement ecarte pour reference
# initiale insuffisante"). It must NOT enter any sign test.
EXCLUDED_BEARINGS = {"Bearing3_5"}


# ============================================================================
# VERBATIM functions from g2_xjtu_horizon.py
# ============================================================================
def haar_energies(x, J):
    a = np.asarray(x, dtype=np.float64)
    E = np.empty(J, dtype=np.float64)
    for j in range(1, J + 1):
        n = len(a) // 2
        if n == 0:
            E[j - 1:] = np.nan
            break
        pairs = a[: 2 * n].reshape(n, 2)
        d = (pairs[:, 0] - pairs[:, 1]) / SQRT2
        a = (pairs[:, 0] + pairs[:, 1]) / SQRT2
        E[j - 1] = float(np.mean(d * d))
    return E


def list_snapshots(bearing_dir):
    files = glob(os.path.join(bearing_dir, "*.csv"))
    files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return files


def load_snapshot_fast(path, channel=CHANNEL):
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(channel,))
    return data


def compute_snapshot_features(sig):
    nseg = len(sig) // SEG_LEN
    Es = []
    for k in range(nseg):
        seg = sig[k * SEG_LEN:(k + 1) * SEG_LEN]
        seg = seg - np.mean(seg)
        E = haar_energies(seg, J_LEVELS)
        Es.append(E)
    Es = np.array(Es)
    E_med = np.median(Es, axis=0)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    kurt = float(scipy_kurtosis(sig, fisher=False))
    return E_med, rms, kurt


def cache_features_for_bearing(bearing_dir, cache_dir, force_recalc=False):
    name = os.path.basename(bearing_dir)
    cache_file = os.path.join(cache_dir, f"{name}_features.npz")
    if not force_recalc and os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["E_all"], data["rms_all"], data["kurt_all"]
    files = list_snapshots(bearing_dir)
    E_all, rms_all, kurt_all = [], [], []
    for f in files:
        sig = load_snapshot_fast(f, channel=CHANNEL)
        E, rms, kurt = compute_snapshot_features(sig)
        E_all.append(E)
        rms_all.append(rms)
        kurt_all.append(kurt)
    E_all = np.array(E_all)
    rms_all = np.array(rms_all)
    kurt_all = np.array(kurt_all)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_file, E_all=E_all, rms_all=rms_all, kurt_all=kurt_all)
    return E_all, rms_all, kurt_all


def compute_z_and_vstar(E_row, E_ref):
    E_used = E_row[:J_USED]
    E_ref_used = E_ref[:J_USED]
    z = (E_used - E_ref_used) / np.maximum(E_ref_used, 1e-12)
    return float(np.max(z)), int(np.argmax(z)) + 1


# Threshold function PASTED as requested -- the 1.4826 scaling is visible here.
def robust_threshold(series):
    """thr = median + N_SIGMA * (1.4826 * MAD)."""
    med = np.median(series)
    mad = np.median(np.abs(series - med))
    sigma_robust = 1.4826 * mad
    thr = med + N_SIGMA * sigma_robust
    return thr, med, sigma_robust


def first_persistent_alarm(series, threshold, k=PERSIST_K):
    count = 0
    for i, v in enumerate(series):
        count = count + 1 if v > threshold else 0
        if count >= k:
            return i - k + 1
    return None


def has_persistent_alarm(series, threshold, k=PERSIST_K):
    return first_persistent_alarm(series, threshold, k) is not None


# ============================================================================
# NEW: FGEK per snapshot (max over segments, same convention as SK)
# ============================================================================
def compute_snapshot_fgek(sig, k_range=XJTU_FGEK_K_RANGE):
    nseg = len(sig) // SEG_LEN
    if nseg == 0:
        return 0.0
    vals = []
    for k in range(nseg):
        seg = sig[k * SEG_LEN:(k + 1) * SEG_LEN]
        seg = seg - np.mean(seg)
        vals.append(compute_fgek(seg, FS, k_range=k_range))
    return float(np.max(vals))


def cache_fgek_for_bearing(bearing_dir, cache_dir, force_recalc=False):
    name = os.path.basename(bearing_dir)
    cache_file = os.path.join(cache_dir, f"{name}_fgek.npz")
    if not force_recalc and os.path.exists(cache_file):
        return np.load(cache_file)["fgek_all"]
    files = list_snapshots(bearing_dir)
    fgek_all = [compute_snapshot_fgek(load_snapshot_fast(f, channel=CHANNEL))
                for f in files]
    fgek_all = np.array(fgek_all)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_file, fgek_all=fgek_all)
    return fgek_all


# ============================================================================
# Sign test -- detector_geq convention (matches summarize): d >= b counts as a
# win (ties folded into wins), baseline-never-alarms = win, detector-never-
# alarms = excluded. One-sided binomial, "greater".
# ============================================================================
def sign_test(wins, losses, ties=0):
    n_pairs = wins + losses
    if n_pairs <= 0:
        return float("nan")
    from scipy.stats import binomtest
    return float(binomtest(wins, n_pairs, 0.5, alternative="greater").pvalue)


def wins_losses_ties(detector_leads, baseline_leads):
    wins = losses = 0
    for d, b in zip(detector_leads, baseline_leads):
        if d is None:
            continue                       # detector never alarmed: excluded
        if b is None or d >= b:
            wins += 1                       # baseline never alarmed, or d>=b
        else:
            losses += 1
    return wins, losses, 0


def wilcoxon_p(detector_leads, baseline_leads):
    """Paired Wilcoxon signed-rank, one-sided 'greater'.

    The CSV stores "alert_time_min" as the ADVANCE (lead = n_total - alarm
    index), so a win for z = MORE advance = LARGER value. The paired
    difference is therefore d - b (positive when z wins), consistent with the
    binomtest 'greater' direction. Pairs with a missing value on either side
    are excluded pairwise."""
    from scipy.stats import wilcoxon
    diff = [d - b for d, b in zip(detector_leads, baseline_leads)
            if d is not None and b is not None]
    if len(diff) == 0:
        return float("nan")
    try:
        return float(wilcoxon(diff, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


# ============================================================================
# Per-bearing processing
# ============================================================================
def process_bearing_baselines(bearing_dir, name, cache_dir):
    E_all, rms_all, kurt_all = cache_features_for_bearing(bearing_dir, cache_dir)
    fgek_all = cache_fgek_for_bearing(bearing_dir, cache_dir)
    n_total = E_all.shape[0]
    if n_total < 20:
        return None

    n_ref = min(max(int(round(REF_FRACTION * n_total)), 5), REF_MAX_SNAPS)
    n_val = n_ref if VAL_EQUALS_REF else n_ref
    sl_val = slice(n_ref, n_ref + n_val)
    sl_test = slice(n_ref + n_val, n_total)
    off = n_ref + n_val

    E_ref = np.median(E_all[:n_ref], axis=0)
    z_series = np.array([compute_z_and_vstar(E_all[i], E_ref)[0]
                         for i in range(n_total)])

    # Threshold learned on VAL (REF/VAL), NEVER on TEST -- per protocol.
    thr_z = robust_threshold(z_series[sl_val])[0]
    thr_rms = robust_threshold(rms_all[sl_val])[0]
    thr_kurt = robust_threshold(kurt_all[sl_val])[0]
    thr_fgek = robust_threshold(fgek_all[sl_val])[0]

    def alarm_idx(series, thr):
        i = first_persistent_alarm(series[sl_test], thr, PERSIST_K)
        return None if i is None else off + i

    def valid(series, thr):
        return not has_persistent_alarm(series[sl_val], thr, PERSIST_K)

    series = {
        "z":       (z_series, thr_z),
        "rms":     (rms_all, thr_rms),
        "kurtosis": (kurt_all, thr_kurt),
        "fgek":    (fgek_all, thr_fgek),
    }
    out = {"bearing": name, "n_total": n_total}
    for m, (s, thr) in series.items():
        i = alarm_idx(s, thr)
        lead = None if i is None else (n_total - i)
        pct = None if i is None else 100.0 * (n_total - i) / n_total
        out[m] = {"alert_time_min": lead, "life_fraction": pct,
                  "threshold_valid": bool(valid(s, thr))}
    return out


# ============================================================================
# Main
# ============================================================================
def run(root, out_dir, bearing=None):
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cache")

    import re as _re
    bearing_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if any(f.lower().endswith(".csv") for f in filenames) and \
           any(_re.match(r"^\d+\.csv$", f, _re.IGNORECASE) for f in filenames):
            bearing_dirs.append(dirpath)
    bearing_dirs = sorted(set(bearing_dirs))
    bearing_dirs = [d for d in bearing_dirs
                    if os.path.basename(d) not in EXCLUDED_BEARINGS]
    if bearing:
        bearing_dirs = [d for d in bearing_dirs if os.path.basename(d) == bearing]
    if not bearing_dirs:
        raise SystemExit("Aucun dossier Bearing* (CSV numerotes) trouve sous --root")

    results = []
    for d in bearing_dirs:
        name = os.path.basename(d)
        try:
            r = process_bearing_baselines(d, name, cache_dir)
            if r:
                results.append(r)
                print(f"  {name}: {r['n_total']} snapshots")
        except Exception as e:
            print(f"  [ERREUR] {name}: {e}")

    # CSV
    csv_path = os.path.join(out_dir, "results_baselines_xjtu.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bearing_id", "method", "alert_time_min",
                    "life_fraction", "threshold_valid"])
        for r in results:
            for m in ("z", "rms", "kurtosis", "fgek"):
                w.writerow([r["bearing"], m, r[m]["alert_time_min"],
                            r[m]["life_fraction"], r[m]["threshold_valid"]])
    print(f"CSV: {csv_path}")

    # Sign tests (detector vs each baseline)
    det_leads = [r["z"]["alert_time_min"] for r in results]
    sign_path = os.path.join(out_dir, "sign_test_xjtu.csv")
    with open(sign_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["baseline", "wins", "losses", "ties", "p_value", "p_wilcoxon"])
        for b in ("rms", "kurtosis", "fgek"):
            base_leads = [r[b]["alert_time_min"] for r in results]
            wins, losses, ties = wins_losses_ties(det_leads, base_leads)
            p = sign_test(wins, losses, ties)
            pw = wilcoxon_p(det_leads, base_leads)
            n_pairs = sum(1 for d, bb in zip(det_leads, base_leads)
                          if d is not None and bb is not None)
            w.writerow([b, wins, losses, ties, p, pw])
            print(f"  detector vs {b:9s}: {wins}W/{losses}L/{ties}T  "
                  f"p_binom={p:.4f}  p_wilcoxon={pw:.4f}  (n_pairs={n_pairs})")
    print(f"Sign tests: {sign_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\buisson\XJTU-SY_Bearing_Datasets")
    ap.add_argument("--out", default="results_baselines_xjtu")
    ap.add_argument("--bearing", default=None)
    args = ap.parse_args()
    run(args.root, args.out, args.bearing)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baselines_cwru.py -- RMS / kurtosis / FGEK / kurtogram (frozen + adaptive)
on CWRU, under the manuscript protocol.

Reuses the EXACT fault-set IDs and healthy 50/50 split of
Maintenance_Predictive_v10c_2.py so that the new baselines are comparable to
the published DWT z-score results.

Output:
  results_baselines_cwru.csv   [config, severity, load, method, AUC, CI_low, CI_high]
  kurtogram_bands_cwru.json    frozen band per load + adaptive band per segment
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import (
    compute_rms,
    compute_temporal_kurt,
    compute_fgek,
    kurtogram_search,
    apply_fixed_band_kurtosis,
    bootstrap_auc_ci,
)

# ---------------------------------------------------------------------------
# Config -- verbatim from Maintenance_Predictive_v10c_2.py (identical splits)
# ---------------------------------------------------------------------------
SEG_LEN = 4096
FS = 12000
DATA_ROOT = r"C:\Users\buisson\CWRU"

# REVIEWER NOTE (kurtogram tree depth): max_level=6 gives a finest 1/3-tree
# bandwidth of fs/(3*2^7) = 31.25 Hz at 12 kHz. A 4096-sample segment lasts
# 0.341 s, so a 31 Hz band holds ~10 carrier cycles -- the practical lower
# bound for a stable envelope-kurtosis estimate. Both variants (frozen and
# adaptive) use the SAME max_level so the two remain comparable.
KURTOGRAM_MAX_LEVEL = 6

METHODS = ["rms", "kurtosis", "fgek", "kurtogram_frozen", "kurtogram_adaptive"]


def _fs(fault_type, label_long, severity, id0, id1):
    return {
        "fault_type": fault_type,
        "label_long": label_long,
        "severity": severity,
        "ids": {"0hp": [id0], "1hp": [id1]},
    }


# REVIEWER NOTE (CWRU availability at Drive-End 1797rpm/12kHz): for the 0.014"
# diameter, the CWRU dataset records ONLY OR014@6h; OR014@3h and OR014@12h are
# NOT recorded at any load. These two combinations are therefore absent from
# Table 4 and are intentionally OMITTED here (13 configs x 2 loads = 26),
# matching the published config set exactly. File 223 belongs to Ball021 (1hp)
# and to nothing else.
FAULT_SETS = {
    "IR007":     _fs("BPFI", "Inner Race 0.007\"",      "0.007\"", 105, 106),
    "Ball007":   _fs("BSF",  "Ball 0.007\"",            "0.007\"", 118, 119),
    "OR007_6h":  _fs("BPFO", "Outer Race 0.007\" @6h",  "0.007\"", 130, 131),
    "OR007_3h":  _fs("BPFO", "Outer Race 0.007\" @3h",  "0.007\"", 144, 145),
    "OR007_12h": _fs("BPFO", "Outer Race 0.007\" @12h", "0.007\"", 156, 158),
    "IR014":     _fs("BPFI", "Inner Race 0.014\"",      "0.014\"", 169, 170),
    "Ball014":   _fs("BSF",  "Ball 0.014\"",            "0.014\"", 185, 186),
    "OR014_6h":  _fs("BPFO", "Outer Race 0.014\" @6h",  "0.014\"", 197, 198),
    "IR021":     _fs("BPFI", "Inner Race 0.021\"",      "0.021\"", 209, 210),
    "Ball021":   _fs("BSF",  "Ball 0.021\"",            "0.021\"", 222, 223),
    "OR021_6h":  _fs("BPFO", "Outer Race 0.021\" @6h",  "0.021\"", 234, 235),
    "OR021_3h":  _fs("BPFO", "Outer Race 0.021\" @3h",  "0.021\"", 246, 247),
    "OR021_12h": _fs("BPFO", "Outer Race 0.021\" @12h", "0.021\"", 258, 259),
}

NORMAL_IDS = {"0hp": 97, "1hp": 98}


# ---------------------------------------------------------------------------
# Data loading -- verbatim from Maintenance_Predictive_v10c_2.py
# ---------------------------------------------------------------------------
def find_mat(data_root, fid, charge):
    root = Path(data_root)
    for folder in [charge, "0hp"]:
        p = root / folder / f"{fid}.mat"
        if p.exists():
            return p
    return None


def load_signal(filepath):
    mat = loadmat(filepath)
    de_key = next((k for k in mat if "_DE_time" in k), None)
    if de_key is None:
        raise ValueError(f"Pas de cle DE_time dans {filepath}")
    return mat[de_key].squeeze().astype(float)


def segments_from_file(filepath, seg_len=SEG_LEN):
    sig = load_signal(filepath)
    n = len(sig) // seg_len
    return sig[:n * seg_len].reshape(n, seg_len)


# ---------------------------------------------------------------------------
# Per-method score computation
# ---------------------------------------------------------------------------
def segment_scores(segments, method, frozen_band=None, fs=FS,
                   max_level=KURTOGRAM_MAX_LEVEL):
    out = []
    if method == "rms":
        out = [compute_rms(s) for s in segments]
    elif method == "kurtosis":
        out = [compute_temporal_kurt(s, fisher=True) for s in segments]
    elif method == "fgek":
        out = [compute_fgek(s, fs) for s in segments]
    elif method == "kurtogram_frozen":
        low, high = frozen_band
        out = [apply_fixed_band_kurtosis(s, fs, low, high) for s in segments]
    elif method == "kurtogram_adaptive":
        out = [kurtogram_search(s, fs, max_level)[2] for s in segments]
    return np.asarray(out, dtype=float)


def auc_from_scores(pos, neg, n_boot=2000):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan, np.nan
    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    y_score = np.concatenate([pos, neg])
    return bootstrap_auc_ci(y_true, y_score, n_boot=n_boot)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_cwru(data_root, out_dir, max_level=KURTOGRAM_MAX_LEVEL, quick=False):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    band_log = {}

    for charge in ["0hp", "1hp"]:
        nid = NORMAL_IDS[charge]
        p_norm = find_mat(data_root, nid, charge)
        if p_norm is None:
            print(f"[SKIP] normal {nid} introuvable pour {charge}")
            continue
        segs_norm = segments_from_file(p_norm)
        split_idx = len(segs_norm) // 2
        segs_ref = segs_norm[:split_idx]
        segs_neg = segs_norm[split_idx:]
        print(f"[{charge}] normal {nid}: {len(segs_norm)} segs -> "
              f"REF:{len(segs_ref)} NEG:{len(segs_neg)}")

        # Variant A: frozen band, searched ONCE on the pooled reference half
        pooled_ref = segs_ref.reshape(-1)
        frozen_low, frozen_high, frozen_kurt = kurtogram_search(
            pooled_ref, FS, max_level)
        band_log[f"frozen_{charge}"] = {
            "band_low": frozen_low, "band_high": frozen_high,
            "kurtosis": float(frozen_kurt),
            "n_ref_segments": int(len(segs_ref)),
        }
        print(f"  frozen band [{charge}] = [{frozen_low:.1f}, {frozen_high:.1f}] Hz")

        neg_scores = {m: segment_scores(segs_neg, m, (frozen_low, frozen_high),
                                        max_level=max_level)
                      for m in METHODS}

        configs = list(FAULT_SETS.items())
        if quick:
            configs = configs[:1]

        for name, cfg in configs:
            fid = cfg["ids"][charge][0]
            p = find_mat(data_root, fid, charge)
            if p is None:
                continue
            segs = segments_from_file(p)
            if len(segs) == 0:
                continue

            pos_scores = {m: segment_scores(segs, m, (frozen_low, frozen_high),
                                            max_level=max_level)
                          for m in METHODS}

            # log adaptive bands for inspection
            if not quick:
                adaptive_bands = []
                for s in segs:
                    lo, hi, _ = kurtogram_search(s, FS, max_level)
                    adaptive_bands.append((round(lo, 1), round(hi, 1)))
                band_log[f"{name}_{charge}"] = {
                    "n_segments": int(len(segs)),
                    "adaptive_bands": adaptive_bands,
                }

            for m in METHODS:
                auc, lo, hi = auc_from_scores(pos_scores[m], neg_scores[m])
                rows.append([name, cfg["severity"], charge, m, auc, lo, hi])
            print(f"  {name} [{charge}] done")

    # CSV
    csv_path = out_path / "results_baselines_cwru.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "severity", "load", "method", "AUC",
                    "CI_low", "CI_high"])
        w.writerows(rows)
    print(f"CSV: {csv_path} ({len(rows)} lignes)")

    # Band log
    json_path = out_path / "kurtogram_bands_cwru.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(band_log, f, indent=2)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--out_dir", default="results_baselines_cwru")
    ap.add_argument("--max_level", type=int, default=KURTOGRAM_MAX_LEVEL)
    ap.add_argument("--quick", action="store_true",
                    help="1 config par charge (smoke test)")
    args = ap.parse_args()
    run_cwru(args.data_root, args.out_dir, args.max_level, args.quick)

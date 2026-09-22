#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baselines_huang.py -- RMS / kurtosis / FGEK on Huang-Baddour.

Reuses VERBATIM the file-naming pattern, SHA-256 dedup, trial split and
loading of evaluation_huang_v20_fixed.py, so the new baselines share the
exact same files as the published DWT/spectral-kurtosis results.

RMS / kurtosis / FGEK have NO wavelet-basis dependence: they are computed
ONCE per dataset (not once per basis). The output table caption must state
this explicitly.

Output: results_baselines_huang.csv  [method, auc_IR, auc_OR, fpr, tpr_IR]
"""

import os
import sys
import re
import csv
import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import compute_rms, compute_temporal_kurt, compute_fgek

# Verbatim from evaluation_huang_v20_fixed.py
FS = 200_000
SEG_LEN = 4096
MAD_FACTOR = 3.0
NAME_PATTERN = re.compile(r"^(H|I|O)-([A-D])-([123])\.mat$", re.IGNORECASE)
DATA_ROOT = r"C:\Users\buisson\CWRU\huang"

# REVIEWER NOTE (FGEK frequency coverage at fs=200 kHz): the dyadic octave
# bands [fs/2^(k+1), fs/2^k] for k=2..7 span ~0.78 kHz .. 50 kHz, covering the
# bearing resonance band of the Huang-Baddour spindles. The exact k-range is a
# hyperparameter of the FIXED (non-adaptive) grid and must be stated in the
# Methods; it is NOT tuned per recording, so no information leaks.
HUANG_FGEK_K_RANGE = range(2, 8)


def load_signal(path, matkey=None):
    path = str(path)
    md = loadmat(path)
    cands = {k: v for k, v in md.items() if not k.startswith("__")}
    if matkey and matkey in cands:
        return np.asarray(cands[matkey]).squeeze().astype(np.float64)
    best, blen = None, -1
    for k, v in cands.items():
        v = np.asarray(v).squeeze()
        if v.ndim == 1 and v.size > blen:
            best, blen = v, v.size
    if best is None:
        raise ValueError(f"{path}: aucun signal trouve")
    return best.astype(np.float64)


def segments_of(path, matkey=None):
    sig = load_signal(path, matkey)
    nseg = len(sig) // SEG_LEN
    if nseg == 0:
        return np.empty((0, SEG_LEN))
    return sig[:nseg * SEG_LEN].reshape(nseg, SEG_LEN)


def build_splits(root):
    data_dir = Path(root)
    files = sorted(data_dir.glob("*.mat"))
    seen = {}
    unique_files = []
    for f in files:
        with open(f, "rb") as fp:
            h = hashlib.sha256(fp.read()).hexdigest()
        if h not in seen:
            seen[h] = f
            unique_files.append(f)
    files = unique_files

    ref, test_neg, test_pos = [], [], []
    for f in files:
        m = NAME_PATTERN.match(f.name)
        if not m:
            continue
        typ, _, essai = m.group(1).upper(), m.group(2), m.group(3)
        if typ == "H":
            if essai in ("1", "2"):
                ref.append(f)
            elif essai == "3":
                test_neg.append(f)
        else:
            test_pos.append(f)
    print(f"[diag] REF={len(ref)} TEST_neg={len(test_neg)} TEST_pos={len(test_pos)}")
    return ref, test_neg, test_pos


def median_mad_threshold(scores):
    if len(scores) == 0:
        return 10.0
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    if mad < 1e-12:
        mad = 1e-12
    return med + MAD_FACTOR * mad


def safe_auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    try:
        return roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg))
    except ValueError:
        return np.nan


def file_score(filepath, method, matkey=None):
    """File-level score = max over non-overlapping segments (same convention
    as the existing spectral-kurtosis baseline: max over segments)."""
    segs = segments_of(filepath, matkey)
    if len(segs) == 0:
        return None
    if method == "rms":
        vals = [compute_rms(s) for s in segs]
    elif method == "kurtosis":
        vals = [compute_temporal_kurt(s, fisher=True) for s in segs]
    elif method == "fgek":
        vals = [compute_fgek(s, FS, k_range=HUANG_FGEK_K_RANGE) for s in segs]
    else:
        raise ValueError(method)
    return float(np.max(vals))


def run_huang(root, out_csv):
    ref, test_neg, test_pos = build_splits(root)
    methods = ["rms", "kurtosis", "fgek"]

    rows = []
    for method in methods:
        ref_scores = [file_score(p, method) for p in ref]
        ref_scores = [s for s in ref_scores if s is not None]
        neg = [file_score(p, method) for p in test_neg]
        neg = [s for s in neg if s is not None]
        pos_I = [file_score(p, method) for p in test_pos
                 if p.name.upper().startswith("I")]
        pos_O = [file_score(p, method) for p in test_pos
                 if p.name.upper().startswith("O")]
        pos_I = [s for s in pos_I if s is not None]
        pos_O = [s for s in pos_O if s is not None]

        auc_IR = safe_auc(pos_I, neg)
        auc_OR = safe_auc(pos_O, neg)
        # REVIEWER NOTE (threshold placement, FIXED): the threshold is learned
        # on the 8 REFERENCE files (trials 1-2), per the manuscript protocol,
        # NOT on the 4 trial-3 negatives. A threshold tuned on the points it
        # judges would flag ~1/4 negatives by construction under asymmetry.
        thr = median_mad_threshold(ref_scores)
        fpr = np.mean([s > thr for s in neg]) if neg else np.nan
        tpr_IR = np.mean([s > thr for s in pos_I]) if pos_I else np.nan
        rows.append([method, auc_IR, auc_OR, fpr, tpr_IR])
        print(f"  {method:10s} AUC_IR={auc_IR:.4f} AUC_OR={auc_OR:.4f} "
              f"thr_ref={thr:.4f} fpr={fpr:.3f} tpr_IR={tpr_IR:.3f}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "auc_IR", "auc_OR", "fpr", "tpr_IR"])
        w.writerows(rows)
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--out", default="results_baselines_huang.csv")
    args = ap.parse_args()
    run_huang(args.root, args.out)

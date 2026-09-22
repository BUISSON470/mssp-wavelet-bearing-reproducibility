#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baselines.py -- RMS / temporal kurtosis / FGEK / kurtogram for the MSSP
revision (Reviewer R1.5 response).

Shared scoring primitives. Each function is a pure, standalone scalar/array
computation; the run scripts (Steps 2-4) drive dataset-specific splits.

Conventions (must stay identical across all run scripts):
  * segment length N = 4096, non-overlapping
  * robust threshold = median + 3 * MAD  (MAD = median(|x - median(x)|))
  * alarm = sliding median(window=21) then persistence k=3 consecutive crossings
  * bootstrap CI: resample units (unit_ids=None -> segment level), seed=42
  * AUC: sklearn.metrics.roc_auc_score
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kurtosis
from sklearn.metrics import roc_auc_score

SEED = 42


# ---------------------------------------------------------------------------
# Baseline 1: RMS
# ---------------------------------------------------------------------------
def compute_rms(x):
    """Root-mean-square of a single segment (float)."""
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


# ---------------------------------------------------------------------------
# Baseline 2: temporal kurtosis
# ---------------------------------------------------------------------------
def compute_temporal_kurt(x, fisher=True):
    """
    Temporal (whole-segment) kurtosis.

    # REVIEWER NOTE (kurtosis definition): the prompt specifies fisher=True
    # (excess kurtosis, Gaussian -> 0). The EXISTING XJTU manuscript code
    # (g2_xjtu_horizon.py, line ~120) uses scipy_kurtosis(sig, fisher=False)
    # (non-excess, Gaussian -> 3). The argmax/ranking is unaffected by the
    # fisher flag, but the absolute threshold (median + 3*MAD) is. This MUST
    # be reconciled before the XJTU table is regenerated, otherwise the new
    # temporal-kurtosis baseline will not be comparable to the published
    # p=0.73 kurtosis figure. Kept as a parameter so the run scripts can
    # match either convention explicitly.
    """
    x = np.asarray(x, dtype=float)
    return float(kurtosis(x, fisher=fisher))


# ---------------------------------------------------------------------------
# Baseline 3: FGEK -- fixed-grid envelope kurtosis (NOT adaptive)
# ---------------------------------------------------------------------------
def _bandpass_envelope_kurtosis(x, fs, low, high, order=4, fisher=True):
    """Butterworth bandpass -> Hilbert envelope -> kurtosis of envelope."""
    nyq = fs / 2.0
    low = float(np.clip(low / nyq, 1e-6, 1.0 - 1e-6))
    high = float(np.clip(high / nyq, low + 1e-6, 1.0 - 1e-6))
    if high <= low:
        return np.nan
    sos = butter(order, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, x)
    envelope = np.abs(hilbert(filtered))
    if envelope.std() < 1e-12:
        return 0.0
    return float(kurtosis(envelope, fisher=fisher))


def compute_fgek(x, fs, k_range=range(5, 11), order=4, fisher=True):
    """
    Fixed-grid envelope kurtosis: for dyadic octave bands [fs/2^(k+1), fs/2^k],
    k in k_range, compute the envelope kurtosis and return the MAX over k.

    # REVIEWER NOTE (FGEK band edges + grid): the band is the frequency
    # OCTAVE [fs/2^(k+1), fs/2^k], not a time window of 2^k samples -- the
    # prompt is ambiguous on this point ("dyadic windows 2^k" vs "bandpass on
    # [fs/2^(k+1), fs/2^k]"); this implementation follows the explicit
    # bandpass definition. For fs=12 kHz, k=5..10 spans ~5.9 Hz .. 375 Hz,
    # which contains the CWRU fault fundamentals (70-162 Hz at k=6,7) but
    # NOT their higher harmonics. The k_range is a hyperparameter and should
    # be justified in the Methods section (fixed, non-adaptive grid -- no
    # per-signal search, hence no information leak).
    """
    x = np.asarray(x, dtype=float)
    vals = []
    for k in k_range:
        low = fs / (2 ** (k + 1))
        high = fs / (2 ** k)
        vals.append(_bandpass_envelope_kurtosis(x, fs, low, high,
                                                order=order, fisher=fisher))
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals)) if vals.size else 0.0


# ---------------------------------------------------------------------------
# Baseline 4: fast kurtogram (Antoni 2007), 1/3-binary tree
# ---------------------------------------------------------------------------
def kurtogram_bands(fs, max_level):
    """
    Enumerate the 1/3-binary-tree bands of the fast kurtogram (Antoni 2007).
    Returns a list of (low, high) tuples in Hz.

    Binary level k:  2^k        bands of width fs / 2^(k+1)
    1/3 level k+1/3: 3 * 2^k    bands of width fs / (3 * 2^(k+1))

    # REVIEWER NOTE (tree structure): Antoni's 1/3-binary tree interleaves a
    # binary tree with a "1/3" refinement (three bands per binary band). The
    # exact edge convention here is left-edge-inclusive / right-edge-exclusive,
    # [low, high), with level-0 = [0, fs/2]. Bands touching DC (low == 0) are
    # kept but are flagged downstream: their envelope kurtosis is dominated by
    # the carrierless low-frequency content and can be unstable.
    """
    nyq = fs / 2.0
    bands = []
    for k in range(max_level + 1):
        n_bin = 2 ** k
        w_bin = nyq / n_bin          # fs / 2^(k+1)
        for i in range(n_bin):
            bands.append((i * w_bin, (i + 1) * w_bin))
        if k < max_level:
            n_third = 3 * (2 ** k)
            w_third = nyq / n_third  # fs / (3 * 2^(k+1))
            for i in range(n_third):
                bands.append((i * w_third, (i + 1) * w_third))
    return bands


def apply_fixed_band_kurtosis(x, fs, band_low, band_high, order=4, fisher=True):
    """
    Scalar for a FIXED band: bandpass [band_low, band_high] -> Hilbert
    envelope -> kurtosis. Structurally identical to a single z_j band value.
    """
    return _bandpass_envelope_kurtosis(x, fs, band_low, band_high,
                                       order=order, fisher=fisher)


def kurtogram_search(x, fs, max_level, order=4, fisher=True):
    """
    Run the kurtogram band search on ONE signal (segment or pooled reference)
    and return (band_low, band_high, kurtosis_value) of the maximizing band.

    This is the shared primitive for BOTH kurtogram variants:
      - Variant A (frozen):     call ONCE on the pooled reference signal.
      - Variant B (adaptive):   call independently on EVERY segment.
    """
    best = (0.0, 0.0, -np.inf)
    for (low, high) in kurtogram_bands(fs, max_level):
        kv = apply_fixed_band_kurtosis(x, fs, low, high,
                                       order=order, fisher=fisher)
        if not np.isfinite(kv):
            continue
        if kv > best[2]:
            best = (low, high, kv)
    return best


# ---------------------------------------------------------------------------
# Alarm pipeline (sliding median + persistence), MSSP-shadow semantics
# ---------------------------------------------------------------------------
def alarm_pipeline(scores_array, threshold, window=21, k=3):
    """
    Convert a per-segment score series into a boolean alarm series.

    For each index i: m_i = median of the last `window` scores (warm-up uses a
    partial window), exceed_i = m_i > threshold, alarm_i = k consecutive
    exceed. Returns a boolean array of the same length.

    # REVIEWER NOTE (warm-up): the first window-1 scores use a partial window
    # (fewer than 21 values) rather than being forced silent. This differs from
    # the firmware MSSP shadow which emits no decision until 21 values are
    # collected. The manuscript's run-to-failure alarm time is dominated by the
    # post-warm-up region, so the impact is negligible, but it is noted for
    # exact reproducibility.
    """
    scores = np.asarray(scores_array, dtype=float)
    n = scores.shape[0]
    alarm = np.zeros(n, dtype=bool)
    if n == 0:
        return alarm
    exceed = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - window + 1)
        exceed[i] = np.median(scores[lo:i + 1]) > threshold
    cnt = 0
    for i in range(n):
        cnt = cnt + 1 if exceed[i] else 0
        if cnt >= k:
            alarm[i] = True
    return alarm


# ---------------------------------------------------------------------------
# Bootstrap AUC confidence interval
# ---------------------------------------------------------------------------
def bootstrap_auc_ci(y_true, y_score, n_boot=2000, unit_ids=None, alpha=0.05,
                     seed=SEED):
    """
    Point AUC + percentile bootstrap CI.

    unit_ids=None : resample individual scores (segment-level bootstrap).
    unit_ids given: cluster/block bootstrap -- resample whole units (e.g. a
                    file) with replacement, preserving within-unit dependence.

    # REVIEWER NOTE (bootstrap unit): the EXISTING CWRU script
    # (Maintenance_Predictive_v10c_2.py) resamples pos/neg scores separately
    # (segment-level) with n_iter capped at 500, seed=42, RandomState. This
    # implementation defaults to n_boot=2000, uses default_rng(seed=42) and a
    # joint resample; the CI widths will differ slightly from the older run.
    # If bit-identical CI to the published tables is required, the run scripts
    # must reproduce the older RandomState/500-iter convention.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if unit_ids is None:
        unit_ids = np.arange(len(y_true))
    unit_ids = np.asarray(unit_ids)

    auc_point = roc_auc_score(y_true, y_score)

    units = np.unique(unit_ids)
    idx_by_unit = {u: np.where(unit_ids == u)[0] for u in units}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        chosen = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([idx_by_unit[u] for u in chosen])
        yt, ys = y_true[idx], y_score[idx]
        if np.unique(yt).size < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.asarray(aucs, dtype=float)
    lo = float(np.percentile(aucs, 100 * alpha / 2.0))
    hi = float(np.percentile(aucs, 100 * (1.0 - alpha / 2.0)))
    return float(auc_point), lo, hi


if __name__ == "__main__":
    # Minimal smoke test -- no dataset required.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    print("rms            =", round(compute_rms(x), 4))
    print("temporal kurt  =", round(compute_temporal_kurt(x), 4))
    print("fgek           =", round(compute_fgek(x, 12000), 4))
    low, high, kv = kurtogram_search(x, 12000, max_level=4)
    print(f"kurtogram band = [{low:.1f}, {high:.1f}] Hz  kurt={kv:.3f}")
    thr = np.median(np.abs(x)) + 3 * np.median(np.abs(np.abs(x) - np.median(np.abs(x))))
    al = alarm_pipeline(np.abs(x), thr, window=21, k=3)
    print("alarm flags    =", int(al.sum()))
    y = (np.abs(x) > 2.0).astype(int)
    auc, lo, hi = bootstrap_auc_ci(y, np.abs(x), n_boot=100)
    print(f"auc (smoke)    = {auc:.3f} [{lo:.3f}, {hi:.3f}]")

# mssp-wavelet-bearing-reproducibility

Reproducibility package for the manuscript *"Relative wavelet energy scores
for bearing fault detection: a stability quantification under basis change"*
(MSSP26-4340, under revision).

This repository reproduces the offline, dataset-level numerical results of the
manuscript: the baseline comparison requested in the revision (RMS, temporal
kurtosis, fixed-grid envelope kurtosis, and the Antoni fast kurtogram in
frozen/adaptive variants), the run-to-failure sign tests, and the LaTeX tables.

## Scope

This package covers only the offline analysis reported in the paper. It does
**not** include:

- any embedded firmware or microcontroller code;
- any real-time detection or threshold-adaptation pipeline;
- the Möbius statistic `S_d1a` implementation (an exploratory variant outside
  the theoretical framework, defined explicitly in the manuscript);
- raw data files (only SHA-256 fingerprints are provided in
  `splits_sha256.json`).

## Datasets (download separately)

- CWRU (1797 rpm, Drive End, 12 kHz): https://engineering.case.edu/bearingdatacenter/
- Huang–Baddour (200 kHz): https://data.mendeley.com/datasets/v43hmbwxpm/1
- XJTU-SY (25.6 kHz): https://biaowang.tech/xjtu-sy-bearing-datasets/
  (DOI: 10.1109/TR.2018.2882682)
- Paderborn KAt: https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/

Point each script at your local copy via its `--data_root` / `--root` flag or
the `DATA_ROOT` constant at the top of the file.

## Requirements

Python 3.10+. Install dependencies:

```
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`,
`PyWavelets`.

## Reproducing each result

| Output | Command | Produces |
|---|---|---|
| Table 1 (window constants) | `python window_constants.py` | `window_constants.csv` |
| Table 1 extended, R1.3 (grid over $j,\theta$) | `python table_extended_table1.py` | `extended_table1.csv`, `table_extended1.tex` |
| Switching protocol, R1.2 ($\rho$ distribution + $[\rho_-,\rho_+]$) | `python r12_rho_analysis.py` | prints the XJTU-SY / Paderborn $\rho$ calibration |
| Adjacent-ratio scope, R1.4 (non-uniform + multi-band) | `python r14_synth_nonuniform.py` | prints $z_3/z_4$ vs the bound |
| Table 8 (CWRU baselines) | `python run_baselines_cwru.py --data_root <CWRU>` | `results_baselines_cwru.csv`, `kurtogram_bands_cwru.json` |
| Table 9 (Huang–Baddour) | `python run_baselines_huang.py --root <huang>` | `results_baselines_huang.csv` |
| Table 10 + sign tests (XJTU-SY) | `python run_baselines_xjtu.py --root <xjtu>` | `results_baselines_xjtu.csv`, `sign_test_xjtu.csv` |
| LaTeX tables (booktabs) | `python make_table_R15.py` | `table_R15_cwru.tex`, `table_R15_huang.tex`, `table_R15_xjtu.tex` |
| Figure 1 (synthetic validation) | `python make_figure1.py` | `figure1.pdf` |
| Figure "XJTU trajectories" | `python make_figure_xjtu.py` | `figure_xjtu.pdf` |

The revision additions map to reviewer comments as follows: R1.2 (switching
protocol) → `r12_rho_analysis.py`; R1.3 (extended Table 1 over the
$j,\theta$ grid) → `table_extended_table1.py`; R1.4 (non-uniform and
multi-band adjacent-ratio experiments) → `r14_synth_nonuniform.py`;
R1.5 (baseline comparison) → `run_baselines_*.py` + `make_table_R15.py`.

Notes:

- `run_baselines_cwru.py` runs the full kurtogram at `max_level=6` by default
  (~6–7 minutes). `--quick` limits to one configuration for a smoke test.
- `run_baselines_xjtu.py` excludes `Bearing3_5` (insufficient initial reference,
  manuscript Sec. 6.5) and applies the one-sided binomial sign test
  (`greater`, ties counted as wins) plus a paired Wilcoxon signed-rank test.
- All randomness uses `seed=42`.
- `window_constants.py` recomputes the Table 1 constants (Daubechies CQF +
  equivalent spectral window, 2^16 grid, `j=4`, `theta=0.15`). `s_in` and
  `m_+/m_-` match the manuscript exactly; `s_adj`/`s_far` match to ~1e-4. The
  original implementation was not preserved, so the table has been recomputed
  here.

## Data integrity and splits

`splits_sha256.json` records SHA-256 fingerprints of every data file used,
organized by dataset, so the results can be reproduced on the exact same files.
The train/validation/test split for each dataset is defined in the
corresponding script as the reference-window / validation-window / test-window
constants at the top of the file (e.g. `REF_FRACTION`, `REF_MAX_SNAPS`,
`VAL_EQUALS_REF` in `run_baselines_xjtu.py`; the half-healthy reference split in
`run_baselines_cwru.py`; the trial-1/2 reference, trial-3 negatives in
`run_baselines_huang.py`). Together the pinned file hashes and the split
constants specify, for each result, exactly which files and which portions were
used.

## Citation

```bibtex
@article{buisson2026relative,
  author  = {Buisson, Christophe},
  title   = {Relative wavelet energy scores for bearing fault detection:
             a stability quantification under basis change},
  journal = {Mechanical Systems and Signal Processing},
  year    = {2026},
  note    = {under review (MSSP26-4340)},
  volume  = {},   % to be completed upon acceptance
  pages   = {},   % to be completed upon acceptance
  doi     = {},   % to be completed upon acceptance
}
```

## License

MIT.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_table_R15.py -- LaTeX booktabs tables for the R1.5 baselines.

Reads the result CSVs (either next to this script or inside the
matching results_baselines_<dataset>/ sub-folder) and emits:
  table_R15_cwru.tex   (26 configs x 5 methods, kurtogram frozen/adaptive split)
  table_R15_huang.tex  (3 methods, run-once -- no basis)
  table_R15_xjtu.tex   (14 bearings x 4 methods + sign-test summary)

p-values and thresholds are NOT rounded below 3 significant digits.
"""

import csv
import os
from collections import OrderedDict

METHOD_LABEL = {
    "rms": "RMS",
    "kurtosis": "Kurtosis",
    "fgek": "FGEK",
    "kurtogram_frozen": r"Kurtogramme (gel\'e)",
    "kurtogram_adaptive": r"Kurtogramme (adaptatif)",
}


def find_csv(base, name, subdir):
    """Look for `name` next to the script, then inside base/subdir/."""
    candidates = [
        os.path.join(base, name),
        os.path.join(base, subdir, name),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Could not find {} in any of:\n  {}".format(name, "\n  ".join(candidates))
    )


def load(path):
    print("  loading", path)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fr(x):
    """French decimal comma."""
    return str(x).replace(".", ",")


def tex_escape(s):
    return (s.replace("\\", "\\textbackslash{}")
             .replace("_", "\\_")
             .replace("&", "\\&")
             .replace("%", "\\%")
             .replace("#", "\\#")
             .replace("$", "\\$")
             .replace("{", "\\{")
             .replace("}", "\\}")
             .replace("~", "\\textasciitilde{}")
             .replace("^", "\\textasciicircum{}"))


def auc_cell(auc, lo, hi):
    return ("{} [{}, {}]".format(
        fr("{:.3f}".format(float(auc))),
        fr("{:.3f}".format(float(lo))),
        fr("{:.3f}".format(float(hi))),
    ))


def clean_sev(s):
    return s.replace('"', "''").replace(".", ",")


def make_cwru(cwru_csv, out_tex):
    rows = load(cwru_csv)
    methods = ["rms", "kurtosis", "fgek", "kurtogram_frozen", "kurtogram_adaptive"]

    # pivot config -> method -> (auc, lo, hi)
    configs = OrderedDict()
    for r in rows:
        key = (r["config"], clean_sev(r["severity"]), r["load"])
        configs.setdefault(key, {})[r["method"]] = (r["AUC"], r["CI_low"], r["CI_high"])

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[tb]\n\\centering\n")
        f.write("\\caption{R1.5 baselines -- CWRU (1797 rpm, 12 kHz), AUC [95\\% CI].\n")
        f.write("The two kurtogram columns are DIFFERENT claims: \\emph{gel\\'e} fixes the\n"
                "band on the reference half only (information budget = z); \\emph{adaptatif}\n"
                "re-searches the band on every test segment (literature ceiling).}\n")
        f.write("\\label{tab:R15_cwru}\n")
        f.write("\\begin{tabular}{lll" + "c" * len(methods) + "}\n")
        f.write("\\toprule\n")
        f.write("Config & S\\'ev. & Load & RMS & Kurtosis & FGEK & "
                "\\multicolumn{2}{c}{Kurtogramme}\\\\\n")
        f.write("\\cmidrule(lr){7-8}\n")
        f.write("& & & & & & gel\\'e & adaptatif\\\\\n")
        f.write("\\midrule\n")
        for (cfg, sev, ld), m in configs.items():
            cells = []
            for mm in methods:
                if mm in m:
                    cells.append(auc_cell(*m[mm]))
                else:
                    cells.append("--")
            f.write("{} & {} & {} & ".format(tex_escape(cfg), sev, ld)
                    + " & ".join(cells) + "\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\\end{table}\n")
    print("  -> {}".format(out_tex))


def make_huang(huang_csv, out_tex):
    rows = load(huang_csv)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[tb]\n\\centering\n")
        f.write("\\caption{R1.5 baselines -- Huang-Baddour (200 kHz). "
                "Computed ONCE per dataset (no wavelet-basis dependence). "
                "AUC$_\\mathrm{IR}$ / AUC$_\\mathrm{OR}$, file-level "
                "false-alarm rate (FPR) and true-positive rate (IR).}\n")
        f.write("\\label{tab:R15_huang}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Method & AUC$_\\mathrm{IR}$ & AUC$_\\mathrm{OR}$ & FPR & TPR$_\\mathrm{IR}$\\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            auc_ir = fr("{:.3f}".format(float(r["auc_IR"])))
            auc_or = fr("{:.3f}".format(float(r["auc_OR"])))
            fpr    = fr("{:.3f}".format(float(r["fpr"])))
            tpr_ir = fr("{:.3f}".format(float(r["tpr_IR"])))
            f.write("{} & {} & {} & {} & {}\\\\\n".format(
                METHOD_LABEL[r["method"]], auc_ir, auc_or, fpr, tpr_ir))
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("  -> {}".format(out_tex))


def make_xjtu(xjtu_csv, sign_csv, out_tex):
    rows = load(xjtu_csv)
    sign = load(sign_csv)

    # pivot bearing -> method -> (alert, life_frac)
    bearings = OrderedDict()
    for r in rows:
        bearings.setdefault(r["bearing_id"], {})[r["method"]] = (
            r["alert_time_min"], r["life_fraction"])

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[tb]\n\\centering\n")
        f.write("\\caption{R1.5 baselines -- XJTU-SY. Alert time (min) / "
                "life fraction (\\%). z = detector (scaled MAD threshold).}\n")
        f.write("\\label{tab:R15_xjtu}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Bearing & z & RMS & Kurtosis & FGEK\\\\\n\\midrule\n")
        for b, m in bearings.items():
            def cell(mm):
                if mm not in m:
                    return "--"
                a, pct = m[mm]
                if a == "" or a is None:
                    return "n.d."
                return "{} ({:.0f}\\%)".format(a, float(pct))
            f.write("{} & {} & {} & {} & {}\\\\\n".format(
                tex_escape(b), cell("z"), cell("rms"),
                cell("kurtosis"), cell("fgek")))
        f.write("\\bottomrule\n\\end{tabular}\n\n")
        # sign-test summary
        f.write("\\smallskip\nSign tests (detector vs baseline, one-sided binomial, "
                "\\texttt{binomtest(...,'greater')}):\\\\\n")
        for s in sign:
            p = fr("{:.3g}".format(float(s["p_value"])))
            f.write("vs {}: {}W/{}L/{}T, p={}.\\\\\n".format(
                METHOD_LABEL[s["baseline"]],
                s["wins"], s["losses"], s["ties"], p))
        f.write("\\end{table}\n")
    print("  -> {}".format(out_tex))


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    make_cwru(
        find_csv(base, "results_baselines_cwru.csv", "results_baselines_cwru"),
        os.path.join(base, "table_R15_cwru.tex"))

    make_huang(
        find_csv(base, "results_baselines_huang.csv", "results_baselines_huang"),
        os.path.join(base, "table_R15_huang.tex"))

    make_xjtu(
        find_csv(base, "results_baselines_xjtu.csv", "results_baselines_xjtu"),
        find_csv(base, "sign_test_xjtu.csv", "results_baselines_xjtu"),
        os.path.join(base, "table_R15_xjtu.tex"))


if __name__ == "__main__":
    main()
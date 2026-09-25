#!/usr/bin/env python3
"""Print summary tables for the CSVs written by audit/null_calibration.py."""

import os

import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "results")
FEATURES = ["h1_count", "h1_entropy", "h1_total_persistence",
            "h1_mean_lifetime", "h1_max_lifetime", "max_betti1"]
FLOOR = 0.0021  # 1/501 rounded up; the smallest p the sign-flip test can return


def false_positive_table(df, title):
    print(f"\n{title}  (replicates = {len(df)})")
    print(f"  {'feature':22s} {'P(p<0.05)':>9s} {'P(p=0.002)':>10s} {'median|d|':>9s} {'max|d|':>7s}")
    for f in FEATURES:
        p, d = df[f"p_{f}"], df[f"d_{f}"].abs()
        print(f"  {f:22s} {(p < 0.05).mean():9.2f} {(p < FLOOR).mean():10.2f} "
              f"{d.median():9.2f} {d.max():7.2f}")
    sig = df[[f"p_{f}" for f in FEATURES]] < 0.05
    print(f"  replicates with >=1 feature p<0.05: {sig.any(axis=1).mean():.2f}; "
          f"with all 6: {sig.all(axis=1).mean():.2f}")


def permutation_table(df, taxa_key, title):
    sub = df[df["taxa"] == taxa_key]
    obs = sub[sub["rep"] == -1].iloc[0]
    null = sub[sub["rep"] >= 0]
    print(f"\n{title}  (subject-level permutations = {len(null)})")
    print(f"  {'feature':22s} {'obs delta':>10s} {'obs d':>7s} {'sign-flip p':>11s} {'subject-perm p':>14s}")
    for f in FEATURES:
        o = obs[f"delta_{f}"]
        p = (np.sum(np.abs(null[f"delta_{f}"]) >= abs(o)) + 1) / (len(null) + 1)
        print(f"  {f:22s} {o:10.4f} {obs[f'd_{f}']:7.2f} {obs[f'p_{f}']:11.3f} {p:14.3f}")


def seed_table(df, key, title):
    print(f"\n{title}")
    print(f"  {'feature':22s} {'d min':>7s} {'d max':>7s} {'#p<0.05':>8s} {'#d>0':>5s}  of {len(df)}")
    for f in FEATURES:
        d, p = df[f"d_{f}"], df[f"p_{f}"]
        print(f"  {f:22s} {d.min():7.2f} {d.max():7.2f} {(p < 0.05).sum():8d} {(d > 0).sum():5d}")


def main():
    a = pd.read_csv(os.path.join(RES, "A_synthetic_null_agp_config.csv"))
    false_positive_table(a, "A. Synthetic null, AGP configuration (165 vs 3,084, n=100)")

    b = pd.read_csv(os.path.join(RES, "B_healthy_split_negative_control.csv"))
    false_positive_table(b, "B. Two random halves of the 26 healthy IBDMDB subjects (n=60)")

    cd = pd.read_csv(os.path.join(RES, "CD_subject_permutation.csv"))
    permutation_table(cd, "paper_taxa",
                      "C. IBDMDB IBD vs non-IBD, diagnosis permuted across subjects (paper's taxa)")
    permutation_table(cd, "detect_taxa",
                      "D. Same, taxa chosen by detection prevalence")

    e = pd.read_csv(os.path.join(RES, "E_hbi_seeds.csv"))
    seed_table(e, "seed", "E. Active vs remission CD (HBI), 10 RNG seeds; paper reports 4/6 significant")

    f = pd.read_csv(os.path.join(RES, "F_one_per_subject_draws.csv"))
    seed_table(f, "seed", "F. One timepoint per subject, 10 different random draws")

    g = pd.read_csv(os.path.join(RES, "G_features_vs_mean_abs_r.csv"))
    print(f"\nG. Variance in each H1 feature explained by mean |Spearman r| "
          f"(quadratic fit, {len(g)} random subsamples of 60)")
    x = g["mean_abs_r"].values
    for feat in FEATURES:
        y = g[feat].values
        fit = np.polyval(np.polyfit(x, y, 2), x)
        r2 = 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  {feat:22s} R^2 = {r2:.2f}")


if __name__ == "__main__":
    main()

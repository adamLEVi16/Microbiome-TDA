#!/usr/bin/env python3
"""Negative-control and calibration checks for the paper's paired-bootstrap test.

Every analysis in the paper uses `paired_resample_test` (src/analysis/bootstrap.py):
draw n samples from pool A and n from pool B, compute H1 features on each
subsample's co-occurrence network, repeat 200 times, then sign-flip the 200
deltas to get a p-value and compute Cohen's d on the 200 bootstrap values.

These checks ask whether that procedure gives valid answers when the truth is
known. Requires the IBDMDB files from scripts/download_ibdmdb.sh.

  A  Synthetic null, AGP configuration: pools of 165 and 3,084 drawn from the
     SAME distribution, n=100 per iteration. Any "significant" result is a
     false positive.
  B  Real-data negative control: the 26 IBDMDB non-IBD subjects split at random
     into two halves of 13 healthy people, run with the paper's IBDMDB settings.
  C  The test the paper should have run: IBD vs non-IBD in IBDMDB with the
     diagnosis label permuted across subjects (80 vs 26), paper's taxon set.
  D  Same as C with taxa chosen by real detection prevalence (see audit notes
     on select_global_taxa).
  E  HBI comparison repeated across 10 RNG seeds.
  F  One-sample-per-subject check repeated across 10 timepoint draws.
  G  How much of each H1 feature is explained by the mean |Spearman r| of the
     network, i.e. a one-number summary with no topology in it.

Usage:  python audit/null_calibration.py            (about 10 minutes on 4 cores)
Writes: audit/results/*.csv
"""

import logging
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.analysis.bootstrap import (  # noqa: E402
    FEATURES, paired_resample_test, select_global_taxa, tda_features,
)
from src.data.ibdmdb_loader import ibdmdb_group_ids, load_ibdmdb  # noqa: E402
from src.data.preprocess import clr_transform, filter_low_abundance  # noqa: E402

logging.disable(logging.CRITICAL)

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

N_ITER = 200
N_PERM_SIGNFLIP = 500


# ── Data (loaded once per worker process) ─────────────────────────────────────

_DATA = {}


def _data():
    if not _DATA:
        abundance, meta = load_ibdmdb(os.path.join(ROOT, "data", "raw", "ibdmdb"))
        filtered = filter_low_abundance(abundance, min_prevalence=0.05, min_reads=0)
        clr = clr_transform(filtered)
        meta = meta.loc[clr.index]
        paper_taxa = select_global_taxa(clr, 80)
        detect_taxa = (filtered > 0).mean(axis=0).nlargest(80).index.tolist()
        _DATA.update(clr=clr, meta=meta, paper_taxa=paper_taxa,
                     detect_taxa=detect_taxa)
    return _DATA


def _signflip_summary(res):
    return {f"d_{r.feature}": r.cohens_d for r in res.itertuples()} | \
           {f"p_{r.feature}": r.permutation_p for r in res.itertuples()} | \
           {f"delta_{r.feature}": r.mean_delta for r in res.itertuples()}


# ── A: synthetic null, AGP configuration ──────────────────────────────────────

def task_a(rep):
    d = _data()
    taxa = d["paper_taxa"]
    corr = np.corrcoef(d["clr"][taxa].values, rowvar=False)
    rng = np.random.default_rng(10_000 + rep)
    z = rng.multivariate_normal(np.zeros(len(taxa)), corr, size=165 + 3084)
    df = pd.DataFrame(z, columns=taxa)
    ids = np.arange(len(df))
    res, _ = paired_resample_test(
        df, ids[:165], ids[165:], taxa, n_iter=N_ITER, subsample_size=100,
        n_perm=N_PERM_SIGNFLIP, rng=np.random.default_rng(rep))
    return {"rep": rep} | _signflip_summary(res)


# ── B: real negative control, random halves of the non-IBD subjects ───────────

def task_b(rep):
    d = _data()
    meta = d["meta"]
    healthy = meta[meta["diagnosis"] == "nonIBD"]
    subjects = np.array(sorted(healthy["Participant ID"].unique()))
    rng = np.random.default_rng(20_000 + rep)
    half = set(rng.permutation(subjects)[: len(subjects) // 2])
    in_half = healthy["Participant ID"].isin(half)
    res, _ = paired_resample_test(
        d["clr"], healthy.index[in_half], healthy.index[~in_half], d["paper_taxa"],
        n_iter=N_ITER, subsample_size=60, n_perm=N_PERM_SIGNFLIP,
        rng=np.random.default_rng(rep))
    return {"rep": rep, "n_a": int(in_half.sum()), "n_b": int((~in_half).sum())} \
        | _signflip_summary(res)


# ── C/D: subject-level label permutation for IBD vs non-IBD ───────────────────

def task_cd(args):
    rep, taxa_key = args
    d = _data()
    meta = d["meta"]
    meta = meta[meta["diagnosis"].isin(["CD", "UC", "nonIBD"])]
    subj_diag = meta.groupby("Participant ID")["diagnosis"].first()
    is_ibd = subj_diag.isin(["CD", "UC"]).values
    if rep >= 0:  # rep < 0 means the observed (unpermuted) labels
        is_ibd = np.random.default_rng(30_000 + rep).permutation(is_ibd)
    ibd_subjects = set(subj_diag.index[is_ibd])
    in_a = meta["Participant ID"].isin(ibd_subjects)
    res, _ = paired_resample_test(
        d["clr"], meta.index[in_a], meta.index[~in_a], d[taxa_key],
        n_iter=N_ITER, subsample_size=60, n_perm=N_PERM_SIGNFLIP,
        rng=np.random.default_rng(42))
    return {"rep": rep, "taxa": taxa_key} | _signflip_summary(res)


# ── E: HBI across seeds ───────────────────────────────────────────────────────

def task_e(seed):
    d = _data()
    ids_a, ids_b, _, _ = ibdmdb_group_ids(d["meta"], "high_vs_low_hbi")
    res, _ = paired_resample_test(
        d["clr"], ids_a, ids_b, d["paper_taxa"], n_iter=N_ITER, subsample_size=60,
        n_perm=N_PERM_SIGNFLIP, rng=np.random.default_rng(seed))
    return {"seed": seed} | _signflip_summary(res)


# ── F: one-sample-per-subject across timepoint draws ──────────────────────────

def task_f(seed):
    d = _data()
    meta = d["meta"]
    rng = np.random.default_rng(seed)
    picks = {}
    for label, diags in [("a", ["CD", "UC"]), ("b", ["nonIBD"])]:
        sub = meta[meta["diagnosis"].isin(diags)]
        picks[label] = [rng.choice(g.index) for _, g in sub.groupby("Participant ID")]
    n = min(60, len(picks["a"]), len(picks["b"]))
    res, _ = paired_resample_test(
        d["clr"], picks["a"], picks["b"], d["paper_taxa"], n_iter=N_ITER,
        subsample_size=n, n_perm=N_PERM_SIGNFLIP, rng=np.random.default_rng(seed + 1))
    return {"seed": seed} | _signflip_summary(res)


# ── G: H1 features vs mean |r| ────────────────────────────────────────────────

def task_g(rep):
    d = _data()
    rng = np.random.default_rng(40_000 + rep)
    clr, taxa = d["clr"], d["paper_taxa"]
    idx = rng.choice(clr.index, size=60, replace=False)
    sub = clr.loc[idx, taxa].reset_index(drop=True)
    r, _ = spearmanr(sub.values)
    off = ~np.eye(len(taxa), dtype=bool)
    feats = tda_features(sub, taxa)
    return {"rep": rep, "mean_abs_r": float(np.abs(r[off]).mean())} | feats


def main():
    with Pool(4) as pool:
        print("A: synthetic null (AGP configuration) ...", flush=True)
        a = pd.DataFrame(pool.map(task_a, range(40)))
        a.to_csv(os.path.join(OUT_DIR, "A_synthetic_null_agp_config.csv"), index=False)

        print("B: random halves of healthy IBDMDB subjects ...", flush=True)
        b = pd.DataFrame(pool.map(task_b, range(30)))
        b.to_csv(os.path.join(OUT_DIR, "B_healthy_split_negative_control.csv"), index=False)

        print("C/D: subject-level label permutation ...", flush=True)
        jobs = [(rep, key) for key in ("paper_taxa", "detect_taxa") for rep in range(-1, 200)]
        cd = pd.DataFrame(pool.map(task_cd, jobs))
        cd.to_csv(os.path.join(OUT_DIR, "CD_subject_permutation.csv"), index=False)

        print("E: HBI across seeds ...", flush=True)
        e = pd.DataFrame(pool.map(task_e, range(42, 52)))
        e.to_csv(os.path.join(OUT_DIR, "E_hbi_seeds.csv"), index=False)

        print("F: one-per-subject across timepoint draws ...", flush=True)
        f = pd.DataFrame(pool.map(task_f, range(100, 110)))
        f.to_csv(os.path.join(OUT_DIR, "F_one_per_subject_draws.csv"), index=False)

        print("G: H1 features vs mean |r| ...", flush=True)
        g = pd.DataFrame(pool.map(task_g, range(300)))
        g.to_csv(os.path.join(OUT_DIR, "G_features_vs_mean_abs_r.csv"), index=False)

    print("done; run audit/summarize.py")


if __name__ == "__main__":
    main()

# Audit of "Persistent Homology Reveals Topological Simplification in IBD-Associated Microbial Co-occurrence Networks"

Audit date: 2026-09-25. Scope: the paper (`paper/`), all analysis code (`src/`, `scripts/`, `simulations/`), and the committed result files (`results/`). IBDMDB was downloaded with the repo's own script and the pipeline was re-run. The American Gut Project data (14 GB) was not downloaded, so AGP-specific claims were checked against the code, the committed CSVs, and a simulation of the AGP design.

## Bottom line

The paper's conclusions are not supported, and the paper should not be submitted or cited in its current form. If it is already posted anywhere (medRxiv, F1000, Zenodo), withdraw it or post a correction.

1. **The significance test cannot distinguish a real difference from noise.** When both groups are drawn from the same population, the test reports p < 0.05 in 53–90% of runs (a valid test gives 5%) and often reports all six features as significant. Every topology group comparison in the paper (AGP, IBDMDB, biomarkers, treatment response) uses this test.
2. **The "top-80 most prevalent taxa" are the first 80 columns of the data file.** The prevalence score is exactly 0.5 for every taxon, so the ranking is a tie and pandas returns columns in file order. In IBDMDB this drops all Proteobacteria, *F. prausnitzii*, *E. coli*, *R. gnavus*, *Roseburia* and *Akkermansia*, which are the taxa the paper's biological explanations rely on.
3. **Cohen's d is computed on bootstrap replicates, not on people**, so it is not an effect size. Two random groups of healthy IBDMDB subjects give a median |d| of 1.25 for loop count, larger than the paper's IBD-vs-non-IBD value for the same feature (0.96). Changing the subsample size from 60 to 100 roughly doubles the paper's d on the same data.
4. **The IBDMDB replication fails.** The paper admits this in Results ("not a robust cross-cohort replication") but claims replication in the Abstract, Introduction, Discussion and Conclusion. A subject-level permutation test (the correct test) is reported below.
5. **Many numbers in the paper contradict the repo's own result files.** The clearest case: the inductive-benchmark permutation p-value is 0.185 in `results/inductive_benchmark.csv`, but the paper reports "p < 0.001".
6. **The AI Tools Declaration is inaccurate.** Git history shows every analysis script and every paper section was written and committed by Claude.

Ratings: **idea 4/10; paper as it stands 1/10.** Reasons are in Section 11.

---

## 1. The paired-bootstrap sign-flip test is invalid

### What the code does

`paired_resample_test` (`src/analysis/bootstrap.py:153`) draws n samples from group A and n from group B, builds one correlation network per group, computes six H1 features, and records the difference. It repeats this 200 times, then:

- **p-value:** randomly flips the signs of the 200 differences, 500 times (`bootstrap.py:225-228`).
- **Cohen's d:** divides the difference in means by the spread of the 200 bootstrap values (`bootstrap.py:237-241`).

### Why that is wrong

The 200 iterations resample the same two fixed pools of people, so they are not 200 independent observations. Any fixed difference between the two pools shows up in every iteration, including differences that exist only because of which people happened to be in each pool. The sign-flip test then treats that fixed difference as 200 replicated findings. Adding iterations shrinks the p-value without adding any data. The correct null ("group label doesn't matter") requires shuffling people between the groups and rebuilding the networks. The loop-attribution scripts do this; the main analyses do not.

The same problem applies to Cohen's d. The spread of bootstrap values depends on the subsample size and the pool overlap (100 drawn from 165 IBD samples each time), not on how much people differ. The repo shows this directly: the same IBDMDB IBD-vs-non-IBD comparison gives total-persistence d = −1.46 with 60 samples per group (`results/ibdmdb_bootstrap.csv`) and d = −2.83 with 100 per group (`results/h2_exploration.csv`). The paper quotes both ranges (|d| = 0.52–1.46 and 1.33–2.83 for H1) in the same subsection without saying they come from different subsample sizes. The paper compares this "d" with a per-person Shannon d of 0.51 and concludes Shannon detects IBD "at roughly one-quarter to one-third the effect size of the topological features" (Results §3.2; also Discussion §4.1). That comparison is invalid.

### Evidence (`audit/null_calibration.py`)

**A. Synthetic null, AGP configuration.** 40 runs. Each draws 165 "IBD" and 3,084 "healthy" samples from one multivariate distribution with IBDMDB's real correlation structure, then runs the paper's code unchanged (n = 100, 200 iterations, 500 sign flips). There is no true difference.

| feature | P(p < 0.05) | P(p = 0.002, the floor) | median \|d\| | max \|d\| |
|---|---|---|---|---|
| h1_count | 0.75 | 0.55 | 0.57 | 1.55 |
| h1_entropy | 0.75 | 0.65 | 0.54 | 1.54 |
| h1_total_persistence | 0.70 | 0.57 | 0.46 | 1.55 |
| h1_mean_lifetime | 0.57 | 0.45 | 0.25 | 0.80 |
| h1_max_lifetime | 0.53 | 0.17 | 0.21 | 0.93 |
| max_betti1 | 0.85 | 0.70 | 0.47 | 1.34 |

98% of null runs have at least one "significant" feature, and 28% have all six. "6/6 features significant" is the paper's headline result, and it happens under pure noise more than a quarter of the time.

**B. Real-data negative control.** 30 runs. The 26 healthy IBDMDB subjects are split at random into two groups of 13, then run with the paper's IBDMDB settings (n = 60).

| feature | P(p < 0.05) | P(p = 0.002) | median \|d\| | max \|d\| |
|---|---|---|---|---|
| h1_count | 0.90 | 0.83 | 1.25 | 3.12 |
| h1_entropy | 0.83 | 0.80 | 1.09 | 3.08 |
| h1_total_persistence | 0.80 | 0.73 | 1.05 | 3.04 |
| h1_mean_lifetime | 0.73 | 0.47 | 0.30 | 1.58 |
| h1_max_lifetime | 0.73 | 0.53 | 0.37 | 0.93 |
| max_betti1 | 0.83 | 0.70 | 0.72 | 2.54 |

43% of splits of healthy people produce "all six features significant". The paper's IBDMDB IBD-vs-non-IBD effects (|d| = 0.52–1.46) sit inside this null range.

**What this means for AGP.** The AGP IBD effects (|d| = 1.23–2.28) are larger than most values in run A, and two of them (total persistence 2.27, max Betti-1 2.07) exceed all 40 null runs. So a real difference between the AGP IBD and non-IBD pools is not ruled out. But the paper never tested it correctly, and Section 3 explains why a real difference would not mean what the paper says. The diet (|d| ≤ 0.65) and antibiotic (|d| ≤ 0.39) results are inside the null range and carry no evidence. Every p = 0.002 and every "FDR-significant" star in the paper should be disregarded.

**Further evidence of instability.** Several results changed completely when only the random-number stream changed:

- The HBI comparison (active vs remission Crohn's) went from 4/6 significant at p = 0.002 (`results/old_v1/ibdmdb_bootstrap.csv`) to 0/6 with p up to 0.93 (`results/ibdmdb_bootstrap.csv`) after commit `1df7aed` changed per-comparison seeding. The paper still reports 4/6 (Results §3.6, Table 6).
- Matched antibiotics at N = 80 is 4/6 significant in `results/agp_bootstrap_v2.csv` and 0/6 in `results/taxa_sensitivity.csv`. These are the same analysis. The paper quotes each in different sentences.

## 2. Taxon selection is by file order, not prevalence

`select_global_taxa` (`src/analysis/bootstrap.py:61-86`):

```python
prevalence = (clr_df > clr_df.median()).mean(axis=0)
top = prevalence.nlargest(n).index.tolist()
```

For any column without ties, exactly half the values exceed the median, so every taxon scores the same. After CLR with a pseudocount there are no ties (zero counts map to sample-specific values), and `nlargest` returns the first `n` columns. The same code is copied in `scripts/run_classification_benchmark_v2.py:113`, `scripts/run_loop_attribution.py:110` and `scripts/run_precision_recall_analysis.py:91`.

Verified on IBDMDB: all 148 taxa score exactly 0.500, and the selection equals the first 80 columns. MetaPhlAn lists species in taxonomic order, so the "network" is:

| phylum | selected (80) | available (148) |
|---|---|---|
| Bacteroidetes | 43 | 43 |
| Firmicutes | 24 | 75 |
| Actinobacteria | 11 | 11 |
| Euryarchaeota | 2 | 2 |
| Proteobacteria | 0 | 15 |
| Verrucomicrobia | 0 | 1 |

Excluded despite being in the data: *F. prausnitzii*, *E. coli*, *R. gnavus*, *R. intestinalis*, *R. hominis*, *A. muciniphila*, *K. pneumoniae*, *V. parvula*. Only 43 of the 80 selected taxa are among the true top 80 by detection prevalence.

Consequences:

- Results §3.6 explains the high-calprotectin network as a cluster "plausibly dominated by Proteobacteria and *Enterobacteriaceae*". The IBDMDB network contains no Proteobacteria. The Discussion's *Faecalibacterium*/*Roseburia* butyrate-loop explanation cannot apply to IBDMDB either, since neither genus is included.
- "*Bacteroides* dominates IBDMDB attribution" follows from Bacteroidetes being 54% of the nodes.
- The "taxa-set sensitivity" analysis (N = 50/80/120) compares nested prefixes of the file. It is not a test of prevalence thresholds.
- The AGP selection includes *Gardnerella* (vaginal), *Neisseria* and *Aggregatibacter* (oral), and *Erwinia*/*Trabulsiella* (plant/environmental), per `results/loop_attribution_differential.csv`. That fits file order, not "the 80 most prevalent gut taxa".
- The Methods section (§2.3) describes a selection the code does not perform.

## 3. What the H1 features measure

Correlation networks built from pure noise (80 independent taxa, n = 100) have far more "topological complexity" than the real data:

| feature | pure noise | AGP healthy (paper) | AGP IBD (paper) |
|---|---|---|---|
| h1_count | 222.1 | 23.7 | 16.9 |
| h1_total_persistence | 8.65 | 0.80 | 0.40 |
| h1_mean_lifetime | 0.039 | 0.034 | 0.024 |
| h1_max_lifetime | 0.115 | 0.119 | 0.080 |
| max_betti1 | 149.1 | 7.1 | 3.4 |

So in this pipeline, "more loops" means "less correlation structure". A network with no biology in it scores highest on every "complexity" feature. Fewer loops in IBD therefore does not show that interaction cycles were lost. It shows that the IBD correlation matrix is further from random in some respect, and which respect is unclear. In IBDMDB, Crohn's networks have a higher mean |Spearman r| than non-IBD (0.398 vs 0.372), but pooled IBD and non-IBD are the same (0.371 vs 0.372), so the pooled difference is not simply stronger correlation.

The real loops are also no longer-lived than noise loops (max lifetime 0.12 vs 0.115). The sampling error of a Spearman correlation at n = 100 is about 0.10, several times larger than typical loop lifetimes (0.02–0.04).

The Discussion (§4.1) says "Persistent loops in the H1 sense require at least three taxa forming a correlation triangle." In a Vietoris–Rips complex, three mutually correlated taxa form a filled triangle, which removes a loop rather than creating one. An H1 class needs at least four taxa in a cycle whose opposite pairs are *not* correlated. The distance 1 − |r| also treats negative correlations (exclusion) the same as positive ones, so the "cross-feeding circuit" interpretation does not follow from the maths.

## 4. IBDMDB: replication and biomarker claims

**Pseudoreplication.** IBDMDB has 12.6 samples per subject on average (106 subjects, 1,338 samples). The main IBDMDB analysis treats samples as independent. The paper's own one-sample-per-subject check reverses the direction for 5 of 6 features, with both directions at p = 0.002. The Results section says this is "not a robust cross-cohort replication". The Abstract ("topologically simpler … across two independent cohorts"), Introduction contribution 4, Discussion §4.5 ("qualitatively successful") and Conclusion item 1 ("reproduced in IBDMDB") still claim replication. It is also the same pipeline on different data, not "a fully independent pipeline".

**Correct test (checks C/D).** The paper's IBDMDB IBD-vs-non-IBD result is reproduced exactly (d = −0.955, −0.730, −1.458, −0.963, −1.229, −0.518). The diagnosis label is then permuted across the 106 subjects, keeping each person's samples together, and the network comparison is recomputed each time: *(results pending; this run was still in progress at the time of this commit)*

**One sample per subject, repeated (check F).** The paper's 1-per-subject result used a single random draw of timepoints. Repeating it with 10 draws: *(results pending)*

**Calprotectin, dose-response and trajectories.** These analyses contradict each other and the text:

- The paper describes the quartile analysis as "Q4 ≥250 µg/g vs Q1 <50 µg/g". The actual `pd.qcut` cutoffs are 15.5, 28.8 and 206.9 µg/g, so Q1 and Q2 are both in the normal range.
- The paper claims a "monotonic dose-response relationship" (§3.10, Conclusion 7). Mean H1 count by quartile is 39.3, 29.9, 40.3, 35.3, and max Betti-1 is 15.5, 9.5, 12.2, 11.3 (`results/treatment_response.csv`). Neither is monotonic. Q4-vs-Q2 has the opposite sign to Q4-vs-Q1.
- The binary calprotectin test gives fewer loops and longer lifetimes (total persistence d = +1.20, max Betti-1 d = +0.64). The quartile "extreme" comparison gives total persistence d = −0.97 and max Betti-1 d = −1.79. The text says the second "confirms" the first.
- Stable-high vs stable-low calprotectin gives more loops (d = +1.75) and shorter lifetimes (d = −2.23), the reverse of the binary result. The text says it "mirrors the calprotectin binary finding". More loops under chronic inflammation also contradicts the paper's central hypothesis, yet the Conclusion lists it as support.
- The Abstract and §3.6 call the calprotectin comparison "within-IBD" and "unaffected by this sampling bias". The code (`src/data/ibdmdb_loader.py:132-135`) does not filter by diagnosis: 79 of the 189 low-calprotectin samples are non-IBD controls.
- The immunosuppressant comparison is described as "within-subject", with 38 subjects and n = 266 vs 504 samples. The code pools samples into two unpaired groups. Re-running the selection gives 30 subjects, and `results/treatment_response.csv` shows n = 164 vs 235.

## 5. Classification benchmark

- **Wrong p-value.** The inductive benchmark's permutation test (`simulations/run_inductive_benchmark.py:264-331`) tests whether topology improves on Shannon. `results/inductive_benchmark.csv` stores the result in columns named for AUCs: observed improvement 0.0276, null SD 0.0293, **p = 0.185**. The paper (§3.9) reports "The mean permuted AUC was 0.028 (i.e., far below the 0.50 chance level … reflecting the class imbalance), establishing a one-sided p < 0.001". The number was misread, AUC does not depend on class imbalance, and the actual result is not significant. The same claim is repeated in the Discussion, Conclusion and Introduction contribution 8.
- **Wrong dataset.** Table 8 is captioned "AGP", but the script loads IBDMDB (`run_inductive_benchmark.py:343`). The paper's "70/30 split, 25 random seeds" is actually 5-fold CV × 5 seeds. The Conclusion's "Δ = 0.15 gap to Aitchison under inductive evaluation" subtracts an IBDMDB number from an AGP number.
- **Same person in train and test.** IBDMDB cross-validation splits samples rather than subjects, so each person's other timepoints are in the training fold. The k-NN "per-sample topology" (`src/tda/sample_features.py`) is built from a sample's 40 nearest neighbours. On average 28% of those neighbours are the same person's other timepoints (nearly all of them), and for 91% of samples the single nearest neighbour is the same person.
- **"Per-sample topology" excludes the sample.** It is computed from the sample's 60 nearest neighbours in CLR space, excluding the sample itself (`sample_features.py:88`). It encodes where a sample sits in Aitchison space, which explains why it adds nothing over Aitchison-PCoA (0.728 vs 0.734).
- Even taken at face value, topology alone is the weakest feature set tested (AGP LR AUC 0.668 vs Aitchison 0.734; precision 12% at 95% specificity).

## 6. Loop attribution

AGP: 5 of 80 taxa have raw p < 0.05, about the 4 expected by chance. **None survive Benjamini–Hochberg correction** (minimum adjusted p = 0.16). Only one of the five has a bootstrap CI that excludes zero; *Blautia*'s CI is [−1.03, 2.40]. IBDMDB: 5 of 80 raw, **0 after FDR** (minimum adjusted p = 0.08). The paper calls these "dominant loop anchors" with "cross-dataset replication". The genus overlap between the two top-20 lists follows from *Bacteroides* being over-represented in both selected taxon sets.

## 7. Other contradictions between the text and the result files

- Supplementary Table S1: "effect sizes shift by less than 0.1d" after matching. `agp_bootstrap_v2.csv` shows shifts up to 0.61 (mean lifetime −1.65 → −1.04; H1 count −1.34 → −1.80).
- Diet, full groups: the text says 5/6 significant; the table says 4/6; `agp_bootstrap_v2.csv` says 6/6.
- Antibiotics: "All matched antibiotic effect sizes fall below this threshold (|d| = 0.01–0.25)". The main CSV shows 0.003–0.31; the sensitivity CSV shows 0.03–0.22 at N = 80.
- "Pre-specified FDR policy … stated before any result is examined" (Introduction; Methods §2.8). Results were committed on 2026-02-24 (`d208b44`, "10/18 tests FDR-significant"). The FDR family was redefined on 2026-03-21 (`1df7aed`, "apply BH correction per subset … not pooled"). Nothing was pre-registered.
- "Wilcoxon p-values … two independent lines of evidence." Both tests run on the same 200 pseudo-replicated differences, so they are not independent evidence.
- Methods says bars with persistence < 1e-8 are dropped; `tda_features` does not do this. Methods says numpy 2.4; `requirements.txt` pins `numpy<2.0`.

## 8. Data handling not addressed

- AGP samples are not deduplicated by participant (`host_subject_id`); no code touches it. The claim that "each subject contributes exactly one sample" is unverified.
- AGP "bloom" sequences (room-temperature shipping artefacts, Amir et al. 2017, mSystems) are not removed, although the Discussion interprets Proteobacteria.
- No rarefaction or depth control. CLR with a 0.5 pseudocount on raw counts gives zero entries values that depend on each sample's depth and richness. Since IBD samples have lower richness, this is a plausible source of group differences in correlation structure.
- "Healthy" in AGP means "answered 'I do not have IBD'"; IBD status is self-reported.
- Plain Spearman on CLR is not a standard network-inference method for compositional data. SparCC, SPIEC-EASI and FlashWeave exist because it produces spurious edges.

## 9. Other material in the repo

- **H2 "voids" and network metrics.** Both use the same invalid test. The Discussion says standard network metrics are threshold-dependent while "all six TDA features are significant … without requiring any threshold choice". The script's own output (`results/network_metrics_output.txt`) ends with "Standard network metrics show comparable or larger effect sizes": modularity is at p = 0.002 at all three thresholds, and 14 of 18 metric–threshold combinations are significant. The network-metrics docstring says AGP, but the code loads IBDMDB.
- **Clinical simulations (`simulations/`).** These score real IBDMDB samples and print hospital-screening dollar "savings" (`simulate_screening.py:286-288`), triage dashboards and "alert thresholds". Every train/test split is by sample (`simulate_triage.py:267`, `simulate_patient_scoring.py:242`, `simulate_screening.py:87`), so every test patient also appears in training. In `results/screening_simulation.csv`, topology alone has about 0.56 sensitivity and 0.55 specificity. None of this belongs in a public repo attached to a medical paper.
- **GitHub Pages visualisation (`docs/index.html`).** It says "This is the d = 2.38 effect you measured, made visible" (line 219). The 31 edges are typed by hand in `scripts/generate_network_viz.py:130-177`, and the inflammation slider hides edges by a fixed rule. It illustrates the conclusion rather than showing data.
- **Cover letter (`cover_letter.docx`, to F1000Research).** It claims "IBDMDB validation", a "pre-registration-style" design, and a benchmark against "Shannon diversity, UniFrac, and Aitchison-PCoA with nested cross-validation". No UniFrac is computed anywhere, and nothing was pre-registered.
- **`_future/cirs/`** encodes Shoemaker "CIRS" biomarker panels. CIRS is not a recognised diagnosis. It is excluded from the package, but it should not sit in a repo you want taken seriously.

## 10. Research integrity: AI Tools Declaration

`paper/main.tex` states that Claude was used only for "LaTeX formatting, editing of figure captions for consistency, and repository organisation", that "all scientific analyses, interpretations, statistical decisions, and written conclusions are solely the author's own work", and that the AI "was not used to generate or interpret results, design methods, or draft the main body of the manuscript".

Git history: 63 of 71 commits are authored by Claude, including every file in `scripts/`, `src/analysis/`, `simulations/` and every `paper/sections/*.tex` file. The human-authored commits are the initial LICENSE and PR merges. You described the code as something you "never understood". Submitting a manuscript with this declaration misrepresents how it was produced. Journals and preprint servers require accurate disclosure of AI use, and an inaccurate declaration is a publication-ethics problem whether or not the science is correct. Fix it before this goes anywhere, even as a preprint.

## 11. Was the idea good?

**Idea: 4/10.** Asking whether disease changes the interaction structure of the gut microbiome, beyond which taxa are present, is legitimate, and persistent homology is a real tool for multi-scale structure. The weaknesses are in the design itself:

- A co-occurrence network describes a group, not a person. Each comparison is one network against one network, so any valid inference must permute people and rebuild networks. That limits power, and the features cannot diagnose individuals without workarounds like the k-NN trick, which reduces to ordinary beta diversity.
- H1 of a correlation-distance complex has no established biological meaning, and it is dominated by estimation noise and overall correlation strength (Section 3).
- IBD already has well-characterised microbiome signatures. A new representation needs to beat Aitchison distance; here it doesn't.
- TDA on microbiome data is not new (e.g., tmap, Liao et al. 2019), so the novelty is modest.

**Execution: 1/10.** The central test does not work, the taxon set is wrong, the replication failed, and the text contradicts its own result files in more than a dozen places. The code is well organised and tested for crashes, but tests that only check the code runs could not catch any of this.

## 12. What to do next

1. Do not submit. If anything is posted, withdraw it or post a correction that fixes the AI declaration.
2. If you want to continue, the minimum sound version is:
   - Select taxa by detection prevalence: `(counts > 0).mean()`.
   - Test with a label permutation over people, rebuilding the networks each time. For IBDMDB, permute subjects, or use one sample per subject.
   - Report effect sizes on a per-person scale, or report the permutation distribution itself.
   - Show that topology adds information beyond mean |r| and beyond Aitchison distance before interpreting it.
   - Deduplicate AGP by participant, remove bloom sequences, and control for sequencing depth.
   - Drop the clinical and treatment claims.
3. Expect this to produce a null or weak result. An honest write-up of "we tested this and it doesn't separate IBD beyond standard metrics, here's why" is legitimate. The current paper is not.
4. Learn enough statistics to judge the code yourself: permutation tests, pseudoreplication, and the difference between a bootstrap standard error and a population standard deviation. Without that, you can't tell a correct analysis from a broken one, whoever wrote it.

## Reproducing this audit

```bash
pip install -r requirements.txt && pip install -e .
bash scripts/download_ibdmdb.sh
python audit/null_calibration.py   # about 10–15 minutes on 4 cores
python audit/summarize.py
```

Outputs are in `audit/results/`.

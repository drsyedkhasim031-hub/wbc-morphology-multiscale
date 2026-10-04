# Evaluation and statistical analysis

All probabilities and predictions use one fixed class ordering saved in the run configuration. Metric values are fractions; percentage-point differences multiply a fraction difference by 100. Undefined per-class precision/recall/F1 is zero over the complete task label list. Balanced accuracy is the mean of these class recalls.

## Validation and calibration

Select a checkpoint by highest validation macro-F1, retaining the earliest tie. LODO selection averages source-domain macro-F1 equally. Fit one temperature from 1,001 logarithmically spaced candidates in [0.05,20] by source-validation mean negative log-likelihood, with smaller-temperature ties. Temperature scaling does not change the argmax class. Each AML fold has its own fitted temperature.

Test output includes logits, probabilities, cell ID, group, domain, seed and checkpoint hash. For AML, concatenate exactly one out-of-fold prediction per cell within each seed. `wbc metrics` rejects repeated cell IDs within a seed. Mean and sample standard deviation are across seeds, not folds. Seed variation is not sampling uncertainty.

ECE uses 15 equal-width bins, left-open/right-closed, with zero in the first bin. Brier score sums squared errors over classes, then averages over cells. Risk–coverage uses maximum probability, admits tied confidence groups together and integrates by the trapezoid rule from (0,0). Coverage at empirical risk ≤1% is retrospective and is not prospective risk control.

## Paired comparisons

`wbc statistics` first aligns identical cells, labels, groups, domains and seeds. Bootstrap draws resample sorted group identifiers with replacement using PCG64 seed 1729, preserving every cell in a sampled group. The same sampled groups are used for both models and every training seed; metric differences are averaged across seeds. The default is 2,000 draws with linearly interpolated 2.5/97.5 percentiles. Positive differences favor model A. With `--equal-domains`, each domain is resampled separately and domain metrics are equally weighted; groups spanning domains are rejected.

Intervals are conditional on fitted models and the available grouping. When only duplicate groups exist, do not label these patient-level intervals. The code does not interpret overlapping cross-validation training sets as independent experimental replicates.

`--permutation` additionally runs a two-sided paired cluster sign-randomization test on pooled accuracy contributions with 10,000 draws and plus-one Monte Carlo correction. This assumes exchangeability at the supplied independent-group level. `statistics.mcnemar_exact` is restricted to one seed and singleton groups; it rejects clustered observations. `statistics.holm` adjusts a user-declared family of p-values; post hoc choice of the family does not create a prespecified test. The paper's rounded tables are never converted to p-values.

## Difficult pairs

Use `wbc pairs`. Keep original multiclass decisions for cases belonging to either reference class. Predictions outside the pair remain errors. AUROC uses the second class's original, unrenormalized probability. No binary retraining or probability restriction occurs. Compare balanced errors only when the baseline balanced error is nonzero; relative reduction is `(baseline_error - model_error) / baseline_error`.

## Independent compartment validation

Prepare an NPZ containing `predictions`, `references` (N×28×28 integer arrays: nucleus0, cytoplasm1, background2), `classes` (fixed-length Unicode array), optionally `validities` (N×28×28) and `extents` (N×2, width/height in crop coordinates). Do not use object arrays. Align rows by an independently stored cell-ID manifest before exporting.

```bash
wbc segment-metrics --arrays data/heldout_compartments.npz --out runs/compartment_metrics.json
```

`segmentation.reference_grid` converts original-resolution reference nucleus/whole-cell masks by exact area-majority voting, ties nucleus→cytoplasm→background. Nuclei outside the reference cell are rejected. Predicted foreground uses the central eight-connected component. Dice/IoU/boundary F1 include empty failures; two empty masks score one, only one empty scores zero. Boundary matching uses four-neighbor boundaries and one-grid-pixel Euclidean tolerance. Outputs include per-cell, per-class and support-weighted means. N:C divides nuclear area by cytoplasm excluding the nucleus; report valid measurement coverage, conditional signed bias/MAE and correlation. Whole-cell-only annotations (e.g. basophils) cannot be passed as complete compartment labels; evaluate those separately with `segmentation.overlap` on whole-cell masks. No bank renaming using evaluation annotations is allowed.

## Robustness

Prepare `roots.json` mapping dataset names to image roots. Corrupt acquired RGB before crop extraction. Save each corruption once and evaluate every comparator on exactly that manifest and image set.

```bash
wbc corrupt --manifest data/splits/raabin.csv --roots data/roots.json --kind stain --severity 1 --variant 0 --preprocessing runs/reference/preprocessing.json --out data/corrupt_stain1_v0
wbc evaluate --checkpoint runs/full/calibrated.pt --manifest data/corrupt_stain1_v0/manifest.csv --roots data/corrupt_stain1_v0/roots.json --out runs/full_stain1_v0
```

Repeat stain/gamma for variants 0 and 1, severities 1–3; blur/JPEG only need variant0. Stain rotates both columns of a single fitting-only reference basis by ±5/10/15 degrees about the unit grey axis and retains the original decomposition residual for corruption generation. Gamma pairs are 0.8/1.2, 0.6/1.4, 0.4/1.6; blur σ=0.5/1/1.5 with ceil(3σ) radius and reflection; JPEG quality90/70/50, 4:4:4. The preprocessing basis for all comparators must be the same. No target-fitted corruption basis is allowed.

```bash
python scripts/summarize_robustness.py --clean runs/full/test_predictions.csv --catalog runs/corruption_catalog.csv --classes raabin --out runs/robustness
```

The catalog columns are `kind,severity,variant,predictions`; the last column is a prediction-file path. Summaries average paired variants equally, use each model's own clean baseline and report corrupted-minus-clean percentage points. Synthetic corruptions do not represent all laboratory shifts.

## Timing

`wbc profile` defaults to 50 warm-ups and 200 synchronized timed repetitions; run both `--batch-size 1` and `--batch-size 24`. Timing begins with in-memory original RGB and includes crop transforms/validity, transfer, stain decomposition, all backbone passes, CPU geometry and classification. Disk reads and temperature formatting are excluded. Report measured throughput, not reciprocal single-cell latency as if it were batch throughput. CUDA peak allocated memory is unavailable on CPU; parameter counts and FLOPs are distinct.

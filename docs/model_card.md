# Model card

**Model:** stain-associated morphology-aware multiscale WBC classifier.

**Artifact status:** new source implementation plus tests. No fully trained full-cohort ConvNeXt-T checkpoint is released. The numerical coefficients under `results/executed/bloodmnist_cpu/` belong to a separate classical baseline. The reduced encoder run is an integration check, not a trained research checkpoint intended for use.

**Intended use:** reproducibility research on cropped single-cell images under declared native or common-five protocols. Inputs are RGB cell images. Outputs are task-label probabilities and optional provisional compartment assignments. Auxiliary maturation applies only to eligible AML lineage labels.

**Evidence:** executed CPU baseline/integration artifacts, full architecture shape tests, numerical and leakage tests. Original full-cohort, cross-dataset, annotation, robustness and clinical validation are not completed by publishing source code.

**Known limitations:** labels and domain composition differ across datasets; rare AML classes have very small counts; grouping may be limited to duplicates; common-five excludes many abnormal/immature categories; crop padding is not acquired context; unsupervised bank identities do not validate anatomy; missing morphology values become standardized zeros; source-calibrated probabilities may be miscalibrated in a new domain.

**Outside scope:** patient diagnosis, whole-slide cell detection, treatment recommendations, autonomous clinical decisions, and claims of clinical safety or diagnostic validity.

**Reproducibility:** model/preprocessing configuration, exact split manifest, source versions, seed, environment, checkpoint hash and predictions are required for each reported result. Independent anatomical evaluation and new-patient/laboratory evaluation are separate work.

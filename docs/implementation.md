# Specification and implementation decisions

This repository is a new implementation of the detailed proposed Methods in the supplied manuscript. It is not a recovery of code that generated the supplied results. Section numbering in the document is inconsistent (some text calls section 3.x section 4.x); this document uses the visible 3.x headings.

| Manuscript section | Files | Implemented behavior |
|---|---|---|
| 3.1 | `labels.py`, `downloads.py`, `data.py` | Native 5/6/15 classes, original AML codes preserved, common-five mapping, counts/exclusions audited |
| 3.2 | `data.py` | Connected patient/slide/exact/adjudicated-duplicate groups, PCG64 1729, greedy split, fixed Test-A, AML OOF, LODO curation exclusion |
| 3.4 | `crops.py`, `stain.py`, `preprocessing.py` | 0.8/1.0/1.2 crop factors, half-up extents, floor origin, white padding, bilinear without antialias, valid-footprint tracking, fixed I0=255, alternating NNLS |
| 3.5–3.6 | `model.py` | Shared four-stage ConvNeXt-T, stage-2 384→64 projection, stage-4 global vector, 3×12×64 prototypes, max-per-bank cosine softmax with temperature 0.1 |
| 3.7 | `morphology.py` | Eight-neighbor central component, disjoint compartments, invalidity rules, eight dimensionless descriptors, coordinate-scaled perimeter/covariance, frozen initial-model descriptor statistics |
| 3.8–3.9 | `model.py` | 512+128→640→384, CLS and learned scale vectors, two post-norm six-head blocks, FFN1536, class and restricted maturation heads |
| 3.10 | `losses.py` | Effective-number class weights, smoothed CE multiplied by true-class weight, pairwise cosine consistency, positive-cosine prototype penalty, eligible Huber δ=0.1 |
| 3.11 | `engine.py` | Float32, deterministic operations, AdamW, parameter-specific decay, update-wise warm-up/cosine, earliest validation tie, 80 complete epochs |
| 3.12 | `metrics.py`, `statistics.py`, `segmentation.py`, `corruptions.py`, `profile.py` | Fixed-label metrics, paired group bootstrap, calibration, tied-confidence risk curves, mask metrics, corruption generation, full-path timing |

## Resolved inconsistencies and explicit extensions

1. **Detailed prose takes precedence over unresolved table cells.** The implementation uses two reconstructed stain-associated RGB streams, no original RGB third stream and no residual stream. Stage 2 and stage 4 have the detailed assignments in 3.5. Table 5 still contains older unresolved descriptions, and the figures remain unchanged reference assets.
2. **Expanded crops use white padding.** Registered parent-slide context is not supplied; parent-image registration and coordinate transforms are not implemented. Padding is invalid acquired content and never additional tissue.
3. **White reference is fixed at 255.** Dataset-specific values mentioned for Supplementary Table S2 are not adopted. No reference-matching stain normalization is performed.
4. **Train-only basis fit is exact but can be slow.** All valid unaugmented canonical fitting pixels contribute to the OD-norm cutoff. A temporary disk-backed norm vector bounds RAM; three data passes select a uniform sample of at most 100,000 surviving pixels. Sampling seed is always 1729. An exact two-variable NNLS active-set solution is used for coefficient updates. SciPy NNLS updates basis rows. Full pooled cohorts may need several GB of scratch disk.
5. **Geometry has no gradients.** It is computed on CPU from hard masks; only soft pooling supplies classification gradients to prototypes. Initial random banks can produce zero valid geometry observations, in which case the prescribed mean zero/scale one fallback is recorded. This does not establish anatomical identities.
6. **Comparator branch choices were not fully specified in the manuscript.** Adapters use ResNet layer2, MobileNetV3 feature6, EfficientNet-B0 feature3, EfficientNetV2-S feature3, or Swin feature3 as the local stage, and their final feature block for global features. Local maps interpolate to input-size/8. Official torchvision default pretrained weights are used for these comparators. These are declared new implementation choices.
7. **Ablations are explicit.** `stain=false` duplicates the RGB crop into the two stream slots. `morphology=false` zeros regional and geometry inputs while retaining projection dimensions; it does not claim a parameter-matched alternative architecture. `use_descriptors=false` retains compartment pooling but removes explicit descriptors. `rgb_only=true` bypasses stain, compartment and Transformer computation and uses a canonical global RGB head; dormant parameters remain registered, so its allocated parameter total is not a minimal standalone baseline count. Objective ablations change weights without pretending to remove architecture. Mask-grid sensitivity uses bilinear resizing of the projected feature map; this extension is not an authenticated historical protocol.
8. **Maturation eligibility is deliberately restricted.** AML MMZ/NGB/NGS map to 0/0.5/1. Only NGB/NGS survive common-five mapping. Other datasets receive no maturation target. Metadata can explicitly disable ambiguous targets. No universal biological ordering is claimed.
9. **Supplementary materials were not supplied separately.** S1–S4 are mentioned in the paper but complete supplemental source tables/raw observations are unavailable. No raw predictions, seed scores, timing observations, N:C distributions or p-values are reverse-engineered from rounded summaries.
10. **Profiling reports measured quantities.** Hardware/software, all crop/stain passes, crop/decomposition/geometry/transfer timing and measured batch throughput are recorded. FLOPs are left unavailable rather than copied from the paper's inconsistent totals. Batch 1 and batch 24 must be profiled separately on actual hardware.
11. **Statistical extensions are labeled.** Paired group bootstrap follows the manuscript. Cluster sign permutation, exact McNemar with strict applicability checks, and Holm are optional additions. They cannot authenticate historical results.
12. **Fresh runs only.** The trainer rejects an existing `best.pt` in its output directory. Exact mid-epoch resume, distributed training and mixed precision are not implemented; the stated recipe uses ordinary float32 training without gradient accumulation.

## Reproducibility boundaries

Partitions must be constructed once and held fixed for seeds 1729–1733. Native AML needs five fresh fold fits per seed. Do not reuse fitted preprocessing between folds or targets. LODO excludes target data from all fitting, normalization, model selection and calibration; exact/adjudicated overlap screening is a disclosed curation access. Patient/slide independence is established only when verified grouping metadata exist. Duplicate hashes are a narrower control.

All outputs should retain their configuration, manifest, dataset version, software/hardware, checkpoint hash, fitted preprocessing and case-level predictions. Deterministic algorithms raise rather than silently fall back to nondeterministic operations. Exact bitwise equality across PyTorch versions, operating systems or hardware is not promised.

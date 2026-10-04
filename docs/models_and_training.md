# Model extensions and training guide

Version 0.2 adds standalone RGB models and training utilities. These are software extensions; they do not establish additional manuscript results. The original proposed architecture remains the default, and existing checkpoints still load.

## Choose a model

`wbc models` prints the catalog. Set `model.architecture: rgb` to use these classifiers. They contain only the selected encoder and classifier, so parameter counts exclude dormant morphology or attention branches.

| Backbone | Representation width | Initialization |
|---|---:|---|
| `compact_cnn` | 128 | Random; three residual stages with GroupNorm |
| `resnet18` | 512 | Random or ImageNet-1K V1 |
| `resnet34` | 512 | Random or ImageNet-1K V1 |
| `resnet50` | 2048 | Random or ImageNet-1K V1 |
| `densenet121` | 1024 | Random or ImageNet-1K V1 |
| `mobilenet_v3_large` | 1280 | Random or ImageNet-1K V1 |
| `efficientnet_b0` | 1280 | Random or ImageNet-1K V1 |
| `efficientnet_v2_s` | 1280 | Random or ImageNet-1K V1 |
| `convnext_tiny` | 768 | Random or ImageNet-1K V1 |
| `swin_t` | 768 | Random or ImageNet-1K V1 |
| `vit_b_16` | 768 | Random or ImageNet-1K V1; exactly 224-pixel input |

All except the compact CNN use [Torchvision's model implementations and explicit weight enums](https://docs.pytorch.org/vision/stable/models.html). `pretrained: true` downloads the selected weights on first use. `weights_version: IMAGENET1K_V1` avoids a moving `DEFAULT` alias. Set `TORCH_HOME` to choose a download/cache directory. Checkpoint loading sets pretrained initialization to false, so inference and resume do not download weights.

The common WBC crop/augmentation pipeline uses RGB values in [0,1] followed by ImageNet mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`. It does **not** use each model's ImageNet evaluation resize recipe; do not compare these WBC runs to ImageNet benchmark numbers. A single factor `[1.0]` is the default. With multiple factors the classifier averages scale features before its final head. MobileNet retains its penultimate classifier projection and dropout; ConvNeXt retains its final normalization.

`freeze_backbone: true` performs head-only training. The encoder's parameters, BatchNorm statistics, and dropout behavior remain frozen. Use a pretrained encoder for this mode. For full fine-tuning set it to false. RGB models use only the class-balanced smoothed classification loss; the proposed method's auxiliary losses do not apply.

## Train one model

First prepare the original datasets and leakage-audited manifests as described in [datasets.md](datasets.md). Edit paths and select your backbone in `configs/rgb_base.yaml`, or generate independent configs using the planner below.

```bash
wbc models
wbc train --config configs/rgb_pbc.yaml --out runs/pbc_resnet18
wbc train --config configs/rgb_raabin.yaml --out runs/raabin_resnet18
wbc train --config configs/rgb_aml_fold1.yaml --out runs/aml_fold1_resnet18
```

AML requires all five outer folds for OOF evaluation; repeat using the corresponding fold manifests. Models share the task's fixed class order and train/validation/test partitions. The engine fits data-dependent preprocessing only on training data, selects checkpoints by validation macro-F1 (equal-domain macro-F1 for LODO), calibrates only on validation, and evaluates test after training.

Optional `training.clip_grad_norm: 1.0` clips the total gradient norm and rejects nonfinite gradients. It is disabled by default because enabling it changes the original recipe. `history.csv` now records each loss component and the minimum/maximum optimizer learning rate. `provenance.json` records total and trainable parameters.

## Resume an interrupted run

```bash
# Optional controlled pause; config still specifies the complete 80-epoch schedule.
wbc train --config configs/rgb_pbc.yaml --out runs/pbc_resnet18 --stop-after-epoch 10

# Resume from the last completely saved epoch.
wbc train --config configs/rgb_pbc.yaml --out runs/pbc_resnet18 --resume runs/pbc_resnet18/resume.pt
```

Every completed epoch atomically writes `resume.pt`: model, optimizer, learning-rate step, history, best checkpoint, preprocessing, Python/NumPy/PyTorch random states, and both DataLoader generators. A crash during an epoch restarts that epoch from the preceding checkpoint. A crash while writing a new checkpoint leaves the old checkpoint intact. There is no mid-batch recovery.

Keep the configuration, original total epoch count, manifest, input images, software environment and device unchanged. The code rejects changed configuration/manifest, PyTorch version, device type/index or CUDA device count. It cannot detect every environment or hardware difference; exact numeric reproducibility is verified for CPU in the included regression test and is not promised across platforms. Resume uses a self-contained copy of the best checkpoint, so it can restore into a new empty directory. Changing batch size, model, epochs, or learning rates requires a fresh run; it is not an exact resume.

A controlled pause performs validation but does not fit temperature or access test images. `run_state.json` records paused/completed status. `best.pt`, `last.pt` and `calibrated.pt` remain inference-compatible. The larger `resume.pt` includes optimizer and best-model state and is ignored by Git, like other checkpoints.

## Plan model/seed comparisons

```bash
python scripts/train_baselines.py --config configs/pbc.yaml --backbones resnet18 resnet50 densenet121 mobilenet_v3_large efficientnet_b0 efficientnet_v2_s convnext_tiny swin_t vit_b_16 --out runs/pbc_rgb
# Add --execute to train; the default writes inspectable YAML configurations only.
python scripts/train_baselines.py --config configs/pbc.yaml --backbones compact_cnn resnet18 --from-scratch --seeds 1729 --out runs/scratch
python scripts/train_baselines.py --config configs/pbc.yaml --backbones resnet18 --freeze-backbone --out runs/head_only

wbc summarize --runs runs/model_a runs/model_b --out runs/comparison.csv
```

The planner defaults to five seeds (1729–1733), retains the base task's schedule/splits and replaces the entire proposed-model configuration with an RGB configuration. Full-scale runs require substantial compute. The summary CSV reports both validation and test metrics, class order, parameters and checkpoint/manifest hashes without ranking models. Compare only compatible class spaces and protocols; select models using validation data.

## Fixed probability ensembles

```bash
wbc ensemble --runs runs/model_a runs/model_b --split validation --weights 1 1 --out runs/ensemble_val
wbc ensemble --runs runs/model_a runs/model_b --split test --weights 1 1 --out runs/ensemble_test
```

Choose members/weights in advance or on validation only; the command does not tune them. It checks class order from run provenance and exact IDs, seeds, labels, groups, domains and partition/fold metadata. Different row order is accepted. Averaging uses calibrated probabilities already stored in each run; it does not recalibrate the ensemble. This is a corresponding-seed model ensemble, not averaging unrelated seed IDs. No artificial logits or single-member checkpoint identity are attached to the result. Source prediction hashes and normalized weights are stored separately. The resulting predictions can be passed to the existing paired statistics tools.

## Batch inference and feature export

```bash
wbc infer --checkpoint runs/pbc_resnet18/calibrated.pt --images path/to/cells --batch-size 16 --features --out runs/inference
wbc evaluate --checkpoint runs/pbc_resnet18/calibrated.pt --manifest data/splits/pbc.csv --split test --out runs/retest
```

`infer` accepts files or recursively scanned directories. It records image/checkpoint hashes, class probabilities, raw logits, class names and temperature provenance. `--features` writes numeric `features.npz` with IDs and per-scale tokens; load using `numpy.load(..., allow_pickle=False)`. RGB tokens are encoder representations; proposed-model tokens precede the scale transformer. Features from different models have different meanings/dimensions. Inference without labels produces no performance claims. Existing `scripts/infer.py` additionally exports provisional compartment overlays for the proposed model.

## Verification

Run `pytest -q`. Tests cover every catalog model's output shape and finite logits, gradients, frozen BatchNorm, proposed checkpoint compatibility, pause/resume equality, changed-manifest rejection, ensemble alignment, feature export and inference/evaluation agreement. Synthetic images used in tests are software fixtures, not research results. See [verification.json](verification.json) for the executed checks.

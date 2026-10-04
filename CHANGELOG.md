# Changelog

## 0.2.0

- Added 11 standalone RGB classifiers, including a compact residual CNN, ResNet, DenseNet, MobileNet, EfficientNet, ConvNeXt, Swin and ViT.
- Added pretrained/full fine-tuning and frozen-encoder training, with task configurations and a model/seed planner.
- Added atomic epoch checkpoints and resume with optimizer, random states, loader generators and the original schedule.
- Added optional gradient clipping, per-component training logs and parameter counts.
- Added aligned probability ensembles, comparison CSVs, batched image/directory inference and numeric feature export.
- Added integration and regression tests; retained legacy proposed-model checkpoint compatibility.

These are implementation extensions. The manuscript's original full-cohort scores remain unverified.

## 0.1.0

- Initial proposed-model reference implementation, dataset protocols, training/evaluation, calibration, statistics, robustness, figures and executed CPU demonstrations.

# Executed RGB integration checks

These are **software integration checks, not benchmark experiments**. Both models trained from random initialization for two epochs on 72 real BloodMNIST images, selected checkpoints on 24 validation images, fitted temperature on validation, and evaluated 24 test images. Original 28×28 images were enlarged to 64×64. There are six WBC classes, four test cells per class, and one seed (1729). This does not establish patient-independent performance.

| Model | Parameters | Selected epoch | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Compact residual CNN | 298,566 | 1 | 33.33% | 22.02% |
| ResNet-18 | 11,179,590 | 2 | 29.17% | 20.56% |
| Prespecified 1:1 probability ensemble | — | Members above | 25.00% | 14.13% |

The ensemble is included to exercise the implementation; its performance is lower here. No model/weight selection was performed on test results. These short runs are not evidence about the architectures' achievable accuracy.

Each model directory contains histories, exact split manifests, case-level validation/test probabilities, class metrics, calibration, parameter counts, environment/configuration provenance, checkpoint hashes, confusion matrices, reliability plots, risk–coverage plots and training curves. Local machine paths in public provenance are replaced with `<demo-work>`; executed hyperparameters and hashes are unchanged. Trained checkpoint files and source images stay outside Git.

Reproduce from the repository root:

```bash
wbc download --demo --out data/bloodmnist.npz
python scripts/demo_rgb.py --data data/bloodmnist.npz --work runs/rgb_demo --results runs/rgb_demo_results
```

The selection uses the same deterministic, duplicate-screened 120-cell subset as the proposed debug-model demonstration. `scope.json` records source checksum and scope; `comparison.csv` contains unrounded metrics. Full original-resolution and full-cohort experiments remain outstanding.

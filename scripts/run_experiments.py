"""Generate an explicit experiment plan or execute it sequentially.

Examples: python scripts/run_experiments.py --task pbc --out runs/plan
          python scripts/run_experiments.py --task aml --execute --out runs/aml
Full matrix training requires substantial GPU time. Default only writes configs.
"""

import argparse
import copy
from pathlib import Path
import yaml
from wbc.engine import read_config, train

ABLATIONS = {
    "full": {},
    "no_stain": {"model": {"stain": False}},
    "no_compartment_morphology": {"model": {"morphology": False}},
    "canonical_only": {"model": {"factors": [1.0], "scale_encoding": False}},
    "no_scale_encoding": {"model": {"scale_encoding": False}},
    "no_maturation_head": {"model": {"maturation": False}, "training": {"maturation_weight": 0.0}},
    "rgb_backbone": {"model": {"rgb_only": True, "stain": False, "morphology": False, "factors": [1.0]}},
    "classification_only": {
        "training": {"scale_weight": 0.0, "prototype_weight": 0.0, "maturation_weight": 0.0}
    },
    "no_scale_loss": {"training": {"scale_weight": 0.0}},
    "no_prototype_loss": {"training": {"prototype_weight": 0.0}},
    "maturation_weight_050": {"training": {"maturation_weight": 0.5}},
    "two_crops": {"model": {"factors": [0.8, 1.0]}},
    "four_crops": {"model": {"factors": [0.7, 0.85, 1.0, 1.2]}},
    "eight_prototypes": {"model": {"prototypes": 8}},
    "sixteen_prototypes": {"model": {"prototypes": 16}},
    "no_descriptors": {"model": {"use_descriptors": False}},
    "mask_grid_14": {"model": {"mask_grid": 14}},
    "mask_grid_56": {"model": {"mask_grid": 56}},
    "four_prototypes_grid14": {"model": {"prototypes": 4, "mask_grid": 14}},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", choices=["raabin", "pbc", "aml", "lodo_raabin", "lodo_pbc", "lodo_aml"], required=True
    )
    p.add_argument("--variants", nargs="+", default=["full"], choices=ABLATIONS)
    p.add_argument(
        "--backbones",
        nargs="+",
        default=["convnext_tiny"],
        choices=[
            "convnext_tiny",
            "swin_t",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_v2_s",
            "mobilenet_v3_large",
        ],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1729, 1734)))
    p.add_argument("--out", required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    base = read_config("configs/aml_fold1.yaml" if args.task == "aml" else f"configs/{args.task}.yaml")
    for fold in range(1, 6) if args.task == "aml" else [None]:
        for backbone in args.backbones:
            for variant in args.variants:
                for seed in args.seeds:
                    cfg = copy.deepcopy(base)
                    cfg["seed"], cfg["model"]["backbone"] = seed, backbone
                    if fold is not None:
                        cfg["manifest"] = f"data/splits/aml_fold{fold}.csv"
                    for key, values in ABLATIONS[variant].items():
                        cfg[key].update(values)
                    run = Path(args.out) / f"{args.task}_{backbone}_{variant}_fold{fold or 0}_seed{seed}"
                    run.mkdir(parents=True, exist_ok=True)
                    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
                    print(run)
                    if args.execute:
                        train(cfg, run)


if __name__ == "__main__":
    main()

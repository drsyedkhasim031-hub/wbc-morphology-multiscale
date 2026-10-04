"""Generate standalone RGB model/seed runs on an existing task split; optionally train."""

import argparse

from wbc.engine import read_config, train
from wbc.experiments import plan_baselines
from wbc.models import MODEL_CATALOG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Existing task config defining labels, data and schedule"
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=list(MODEL_CATALOG),
        default=["resnet18", "densenet121", "efficientnet_b0"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1729, 1730, 1731, 1732, 1733])
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plans = plan_baselines(
        read_config(args.config),
        args.backbones,
        args.seeds,
        args.out,
        not args.from_scratch,
        args.freeze_backbone,
    )
    print(f"Wrote {len(plans)} model/seed configurations to {args.out}", flush=True)
    if args.execute:
        for plan in plans:
            train(plan["config"], plan["out"])


if __name__ == "__main__":
    main()

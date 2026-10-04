"""Command-line entry points; run wbc --help for the complete interface."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .utils import write_json


def get_classes(text):
    from .labels import NATIVE, COMMON

    return COMMON if text == "common5" else NATIVE[text] if text in NATIVE else text.split(",")


def main():
    parser = argparse.ArgumentParser(description="Stain-associated morphology-aware WBC research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List standalone RGB model architectures and explicit weight variants")
    p = commands.add_parser("sources", help="Print official dataset acquisition instructions")
    p = commands.add_parser(
        "download", help="Download a URL with optional checksum, or the verified CPU demo dataset"
    )
    p.add_argument("--url")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--sha256")
    p.add_argument("--out", required=True)
    p = commands.add_parser(
        "manifest", help="Index decoded images, exact hashes, labels and grouping evidence"
    )
    p.add_argument("--root", required=True)
    p.add_argument("--dataset", choices=["raabin", "pbc", "aml"], required=True)
    p.add_argument("--metadata")
    p.add_argument("--out", required=True)
    p = commands.add_parser(
        "duplicates", help="Produce candidate pairs for visual adjudication; no automatic pHash merging"
    )
    p.add_argument("--manifest", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser(
        "split", help="Generate fixed group-aware native, AML outer-fold or LODO partitions"
    )
    p.add_argument("--manifest", nargs="+", required=True)
    p.add_argument("--protocol", choices=["raabin", "pbc", "aml", "lodo"], required=True)
    p.add_argument("--target", choices=["raabin", "pbc", "aml"])
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--links", help="CSV with id_a,id_b of adjudicated confirmed duplicate links only")
    p.add_argument("--out", required=True)
    p = commands.add_parser(
        "train", help="Fit preprocessing, train, select on validation, calibrate and test"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", help="Epoch-boundary resume.pt; keep the original config and schedule")
    p.add_argument("--stop-after-epoch", type=int, help="Pause after this completed epoch without testing")
    p = commands.add_parser("infer", help="Batch inference on images/directories; optional numeric features")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--features", action="store_true")
    p.add_argument("--out", required=True)
    p = commands.add_parser("summarize", help="Export a comparison CSV from completed run directories")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("ensemble", help="Prespecified average of aligned probabilities from >=2 runs")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--weights", nargs="+", type=float)
    p.add_argument("--out", required=True)
    p = commands.add_parser("evaluate", help="Run a frozen checkpoint on an explicit partition")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--roots", help="JSON file mapping dataset names to local image roots")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    p = commands.add_parser("metrics", help="Summarize predictions per seed; pool AML OOF cells within seed")
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("statistics", help="Paired group bootstrap plus optional group permutation")
    p.add_argument("--a", nargs="+", required=True)
    p.add_argument("--b", nargs="+", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--equal-domains", action="store_true")
    p.add_argument("--permutation", action="store_true")
    p.add_argument("--out", required=True)
    p = commands.add_parser("figures", help="Confusion, reliability and risk–coverage from one seed")
    p.add_argument("--predictions", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--title", default="Executed evaluation")
    p.add_argument("--out", required=True)
    p = commands.add_parser("pairs", help="Difficult pairs retaining original multiclass decisions")
    p.add_argument("--predictions", required=True)
    p.add_argument("--classes", default="aml")
    p.add_argument("--pairs", default="MYO:MOB,MYB:MMZ,PMO:MYB,PMO:MYO,LYA:LYT,NGB:NGS")
    p.add_argument("--out", required=True)
    p = commands.add_parser("segment-metrics", help="Evaluate held-out canonical reference masks")
    p.add_argument(
        "--arrays",
        required=True,
        help="NPZ with predictions, references, classes; optional validities, extents",
    )
    p.add_argument("--out", required=True)
    p = commands.add_parser("corrupt", help="Materialize fixed corruptions before cropping")
    p.add_argument("--manifest", required=True)
    p.add_argument("--roots", required=True)
    p.add_argument("--kind", choices=["stain", "gamma", "blur", "jpeg"], required=True)
    p.add_argument("--severity", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--variant", type=int, default=0)
    p.add_argument("--preprocessing", help="Fitting-only preprocessing.json for stain corruptions")
    p.add_argument("--out", required=True)
    p = commands.add_parser("profile", help="Time all crop/stain passes and preprocessing from in-memory RGB")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repetitions", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "models":
        from .models import MODEL_CATALOG

        print(json.dumps(MODEL_CATALOG, indent=2))
    elif args.command == "sources":
        from .downloads import SOURCES

        print(json.dumps(SOURCES, indent=2))
    elif args.command == "download":
        from .downloads import SOURCES, download_file

        if args.demo:
            info = SOURCES["bloodmnist_demo"]
            download_file(info["url"], args.out, info["md5"], "md5")
        elif args.url:
            download_file(args.url, args.out, args.sha256)
        else:
            parser.error("Use --demo or --url")
    elif args.command == "manifest":
        from .data import build_manifest

        build_manifest(args.root, args.dataset, args.out, args.metadata)
    elif args.command in ("duplicates", "split"):
        from .data import duplicate_candidates, split_manifest

        frame = pd.concat([pd.read_csv(p, dtype={"phash": str}) for p in args.manifest], ignore_index=True)
        if args.command == "duplicates":
            result = duplicate_candidates(frame)
        else:
            links = pd.read_csv(args.links) if args.links else None
            result = split_manifest(frame, args.protocol, args.target, args.fold, links=links)
            write_json(
                args.out + ".audit.json",
                {
                    "counts": result.groupby(["dataset", "split", "label"]).size().to_dict(),
                    "protocol": args.protocol,
                    "target": args.target,
                    "fold": args.fold,
                    "split_seed": 1729,
                    "target_curation_access": args.protocol == "lodo",
                    "note": "LODO overlap screening examines target hashes only for curation; target images never fit preprocessing or model.",
                },
            )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
    elif args.command == "train":
        from .engine import read_config, train

        config = read_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        train(config, args.out, resume=args.resume, stop_after_epoch=args.stop_after_epoch)
    elif args.command == "infer":
        from .inference import infer_images

        infer_images(args.checkpoint, args.images, args.out, args.batch_size, args.device, args.features)
    elif args.command == "summarize":
        from .experiments import summarize_runs

        summarize_runs(args.runs, args.out)
    elif args.command == "ensemble":
        from .ensemble import ensemble_runs

        ensemble_runs(args.runs, args.split, args.out, args.weights)
    elif args.command == "evaluate":
        from .engine import evaluate_checkpoint

        roots = json.loads(Path(args.roots).read_text()) if args.roots else None
        evaluate_checkpoint(args.checkpoint, args.manifest, args.split, args.out, roots, args.device)
    elif args.command == "metrics":
        from .metrics import evaluate_frame

        frame = pd.concat([pd.read_csv(p) for p in args.predictions], ignore_index=True)
        write_json(args.out, evaluate_frame(frame, get_classes(args.classes)))
    elif args.command == "statistics":
        from .statistics import paired_bootstrap, cluster_randomization

        a = pd.concat([pd.read_csv(p) for p in args.a], ignore_index=True)
        b = pd.concat([pd.read_csv(p) for p in args.b], ignore_index=True)
        classes = get_classes(args.classes)
        result = paired_bootstrap(a, b, classes, args.draws, equal_domains=args.equal_domains)
        if args.permutation:
            result["permutation"] = cluster_randomization(a, b, classes)
        write_json(args.out, result)
    elif args.command == "figures":
        from .plots import plot_predictions

        plot_predictions(pd.read_csv(args.predictions), get_classes(args.classes), args.out, args.title)
    elif args.command == "pairs":
        from .metrics import difficult_pair

        frame = pd.read_csv(args.predictions)
        classes = get_classes(args.classes)
        pcols = [f"prob_{i}" for i in range(len(classes))]
        result = {}
        for seed, part in frame.groupby("seed"):
            result[str(seed)] = {
                pair: difficult_pair(part.y_true, part[pcols], *[classes.index(c) for c in pair.split(":")])
                for pair in args.pairs.split(",")
            }
        write_json(args.out, result)
    elif args.command == "segment-metrics":
        from .segmentation import evaluate_masks

        with np.load(args.arrays, allow_pickle=False) as arrays:
            result = evaluate_masks(**{key: arrays[key] for key in arrays.files})
        write_json(args.out, result)
    elif args.command == "corrupt":
        from .corruptions import corrupt
        from .utils import sha256

        frame = pd.read_csv(args.manifest)
        roots = json.loads(Path(args.roots).read_text())
        basis = (
            np.array(json.loads(Path(args.preprocessing).read_text())["basis"])
            if args.preprocessing
            else None
        )
        frame = frame[frame.split == "test"].copy()
        for idx, row in frame.iterrows():
            root = Path(roots[row.dataset]).resolve()
            source = (root / row.path).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Path outside root")
            # Hash-based filenames avoid collisions and traversal from foreign manifests.
            import hashlib

            relative = row.dataset + "/" + hashlib.sha256(row.id.encode()).hexdigest() + ".png"
            destination = Path(args.out) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                corrupt(image.convert("RGB"), args.kind, args.severity, args.variant, basis).save(destination)
            frame.loc[idx, "path"] = relative
            frame.loc[idx, "corrupted_file_sha256"] = sha256(destination)
        frame.to_csv(Path(args.out) / "manifest.csv", index=False)
        write_json(
            Path(args.out) / "roots.json", {d: str(Path(args.out).resolve()) for d in frame.dataset.unique()}
        )
        write_json(
            Path(args.out) / "corruption.json",
            {
                "kind": args.kind,
                "severity": args.severity,
                "variant": args.variant,
                "basis": basis,
                "preprocessing_sha256": sha256(args.preprocessing) if args.preprocessing else None,
            },
        )
    elif args.command == "profile":
        from .profile import profile

        images = []
        for path in args.images:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        write_json(
            args.out,
            profile(args.checkpoint, images, args.batch_size, args.warmup, args.repetitions, args.device),
        )


if __name__ == "__main__":
    main()

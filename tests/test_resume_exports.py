from copy import deepcopy
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from wbc.engine import train, load_model
from wbc.ensemble import average_probabilities, ensemble_runs
from wbc.experiments import plan_baselines, summarize_runs
from wbc.inference import infer_images


@pytest.fixture
def tiny_config(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    rng = np.random.default_rng(18)
    rows = []
    for split, count in [("train", 8), ("val", 4), ("test", 4)]:
        for index in range(count):
            identifier = f"{split}_{index}"
            path = identifier + ".png"
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(root / path)
            rows.append(
                {
                    "id": identifier,
                    "group": identifier,
                    "dataset": "demo",
                    "path": path,
                    "split": split,
                    "label": ["a", "b"][index % 2],
                    "maturation": float("nan"),
                }
            )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return {
        "manifest": str(manifest),
        "roots": {"demo": str(root)},
        "classes": ["a", "b"],
        "seed": 1729,
        "threads": 2,
        "device": "cpu",
        "image_size": 32,
        "model": {"architecture": "rgb", "backbone": "compact_cnn", "factors": [1.0], "dropout": 0.2},
        "training": {"epochs": 2, "warmup_epochs": 0, "batch_size": 4, "workers": 0, "clip_grad_norm": 1.0},
    }


def test_epoch_resume_matches_continuous_training_and_exports(tiny_config, tmp_path):
    full, paused = tmp_path / "full", tmp_path / "paused"
    train(tiny_config, full)
    train(tiny_config, paused, stop_after_epoch=1)
    assert not (paused / "test_predictions.csv").exists()
    assert not (paused / "calibrated.pt").exists()
    assert json.loads((paused / "run_state.json").read_text())["status"] == "paused"
    train(tiny_config, paused, resume=paused / "resume.pt")
    a = torch.load(full / "last.pt", weights_only=True)
    b = torch.load(paused / "last.pt", weights_only=True)
    for key, tensor in a["state_dict"].items():
        torch.testing.assert_close(tensor, b["state_dict"][key], rtol=0, atol=0)
    first, second = [
        pd.read_csv(directory / "history.csv").drop(columns="elapsed_seconds") for directory in [full, paused]
    ]
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    for partition in ["validation", "test"]:
        first, second = [
            pd.read_csv(directory / f"{partition}_predictions.csv").drop(
                columns="checkpoint_sha256", errors="ignore"
            )
            for directory in [full, paused]
        ]
        pd.testing.assert_frame_equal(first, second, check_exact=True)
    model, _, temperature = load_model(paused / "calibrated.pt")
    assert not model.training and temperature > 0
    with pytest.raises(FileExistsError):
        train(tiny_config, full)

    image = tmp_path / "images" / "test_0.png"
    exported = infer_images(full / "calibrated.pt", [image], tmp_path / "inference", features=True)
    evaluation = pd.read_csv(full / "test_predictions.csv").query("id == 'test_0'")
    np.testing.assert_allclose(exported[["prob_0", "prob_1"]], evaluation[["prob_0", "prob_1"]], atol=1e-6)
    with np.load(tmp_path / "inference" / "features.npz", allow_pickle=False) as arrays:
        assert arrays["tokens"].shape == (1, 1, 128)
        assert arrays["ids"].tolist() == exported.id.tolist()
    comparison = summarize_runs([full, paused], tmp_path / "comparison.csv")
    assert len(comparison) == 4
    result = ensemble_runs([full, paused], "test", tmp_path / "ensemble")
    np.testing.assert_allclose(
        result[["prob_0", "prob_1"]], pd.read_csv(full / "test_predictions.csv")[["prob_0", "prob_1"]]
    )


def test_resume_rejects_changed_config_and_manifest(tiny_config, tmp_path):
    run = tmp_path / "run"
    train(tiny_config, run, stop_after_epoch=1)
    changed = deepcopy(tiny_config)
    changed["training"]["epochs"] = 3
    with pytest.raises(ValueError, match="configuration"):
        train(changed, run, resume=run / "resume.pt")
    with open(tiny_config["manifest"], "a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Manifest changed"):
        train(tiny_config, run, resume=run / "resume.pt")


def test_baseline_plan_replaces_proposed_options_and_preserves_data(tiny_config, tmp_path):
    base = deepcopy(tiny_config)
    base["model"] = {"backbone": "convnext_tiny", "stain": True, "prototypes": 12}
    plans = plan_baselines(base, ["compact_cnn", "resnet18"], [1729, 1730], tmp_path / "plans")
    assert len(plans) == 4
    assert base["model"]["stain"] is True
    for plan in plans:
        assert "stain" not in plan["config"]["model"]
        assert plan["config"]["manifest"] == base["manifest"]
        assert plan["config_path"].exists()
    with pytest.raises(ValueError, match="frozen randomly"):
        plan_baselines(base, ["compact_cnn"], [1], tmp_path / "bad", freeze_backbone=True)


def test_probability_ensemble_alignment_and_fixed_weights():
    a = pd.DataFrame(
        {
            "id": ["x", "y"],
            "seed": [1, 1],
            "group": ["g1", "g2"],
            "dataset": ["d", "d"],
            "y_true": [0, 1],
            "prediction": [0, 1],
            "prob_0": [0.8, 0.4],
            "prob_1": [0.2, 0.6],
        }
    )
    b = a.copy()
    b[["prob_0", "prob_1"]] = [[0.6, 0.4], [0.2, 0.8]]
    result = average_probabilities([a, b.iloc[::-1]], ["a", "b"], [3, 1])
    np.testing.assert_allclose(result.prob_0, [0.75, 0.35])
    b.loc[0, "group"] = "different"
    with pytest.raises(ValueError, match="differ"):
        average_probabilities([a, b], ["a", "b"])
    with pytest.raises(ValueError, match="weight"):
        average_probabilities([a, a], ["a", "b"], [-1, 2])


def test_proposed_checkpoint_remains_loadable(tmp_path):
    from wbc.model import WBCModel

    config = {"classes": ["a", "b"], "model": {"backbone": "debug", "factors": [1.0]}}
    original = WBCModel(2, **config["model"]).eval()
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"state_dict": original.state_dict(), "config": config, "epoch": 1}, checkpoint)
    restored, _, _ = load_model(checkpoint)
    for key, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)


def test_proposed_resume_preserves_fitted_statistics(tiny_config, tmp_path):
    tiny_config["model"] = {"backbone": "debug", "factors": [1.0], "pretrained": False}
    full, paused = tmp_path / "proposed_full", tmp_path / "proposed_paused"
    train(tiny_config, full)
    train(tiny_config, paused, stop_after_epoch=1)
    saved = torch.load(paused / "resume.pt", weights_only=True)
    # Resume into a new directory, using only the self-contained checkpoint.
    resumed = tmp_path / "proposed_restored"
    train(tiny_config, resumed, resume=paused / "resume.pt")
    a = torch.load(full / "last.pt", weights_only=True)
    b = torch.load(resumed / "last.pt", weights_only=True)
    for key, tensor in a["state_dict"].items():
        torch.testing.assert_close(tensor, b["state_dict"][key], rtol=0, atol=0)
    for key in ["basis", "descriptor_mean", "descriptor_std"]:
        torch.testing.assert_close(saved["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)
    assert (resumed / "history.csv").exists()


def test_atomic_checkpoint_failure_keeps_previous_file(tmp_path, monkeypatch):
    from wbc.checkpoints import atomic_save

    checkpoint = tmp_path / "resume.pt"
    atomic_save({"epoch": 1}, checkpoint)

    def failed_save(payload, destination):
        destination.write_bytes(b"interrupted write")
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", failed_save)
    with pytest.raises(OSError, match="disk failure"):
        atomic_save({"epoch": 2}, checkpoint)
    assert torch.load(checkpoint, weights_only=True)["epoch"] == 1
    assert list(tmp_path.iterdir()) == [checkpoint]

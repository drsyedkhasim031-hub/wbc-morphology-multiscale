import numpy as np
import pandas as pd
import pytest
from wbc.data import connect_groups, greedy_allocate, assert_no_leakage, split_manifest
from wbc.labels import EXPECTED, harmonize, maturation
from wbc.metrics import classification, calibration, difficult_pair, fit_temperature
from wbc.statistics import paired_bootstrap, holm, mcnemar_exact


def frame():
    return pd.DataFrame(
        {
            "id": [f"id{i}" for i in range(60)],
            "dataset": ["pbc"] * 60,
            "label": [str(i % 3) for i in range(60)],
            "patient_id": [str(i // 3) for i in range(60)],
            "slide_id": [""] * 60,
            "exact_hash": [f"hash{i}" for i in range(60)],
        }
    )


def test_group_union_is_transitive_and_splits_stay_together():
    f = frame()
    f.loc[3, "exact_hash"] = f.loc[0, "exact_hash"]
    f = connect_groups(f)
    assert f.iloc[:6].group.nunique() == 1
    f["split"] = greedy_allocate(f, {"train": 0.7, "val": 0.15, "test": 0.15})
    assert_no_leakage(f)
    assert set(f.split) == {"train", "val", "test"}
    assert (f.split == greedy_allocate(f, {"train": 0.7, "val": 0.15, "test": 0.15})).all()
    f.loc[0, "split"] = "test" if f.loc[1, "split"] != "test" else "train"
    with pytest.raises(ValueError):
        assert_no_leakage(f)


def test_expected_counts_and_harmonization():
    assert [sum(EXPECTED[d].values()) for d in ["raabin", "pbc", "aml"]] == [14514, 13193, 18365]
    assert sum(v for k, v in EXPECTED["aml"].items() if harmonize(k, "aml")) == 14833
    assert harmonize("MYB", "aml") is None
    assert harmonize("LYA", "aml") == "lymphocyte"
    assert maturation("MMZ", "aml") == 0
    assert maturation("NGB", "aml") == 0.5
    assert np.isnan(maturation("neutrophil", "pbc"))


def test_fixed_classes_and_outside_pair_errors():
    result = classification([0, 0, 1], [0, 2, 1], ["a", "b", "c"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["balanced_accuracy"] == 0.5
    pairs = difficult_pair([0, 1], [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]], 0, 1)
    assert pairs["balanced_accuracy"] == 0.5
    assert pairs["outside_pair_predictions"] == 1


def test_temperature_does_not_change_class_and_ties_enter_together():
    from scipy.special import softmax

    logits = np.array([[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
    t = fit_temperature(logits, [0, 1, 1, 1])
    assert t > 0
    assert (softmax(logits / t, axis=1).argmax(1) == logits.argmax(1)).all()
    result = calibration([0, 1], [[0.8, 0.2], [0.8, 0.2]])
    assert result["coverage"].tolist() == [0.0, 1.0]
    assert result["risk"].tolist() == [0.0, 0.5]
    assert result["aurc"] == 0.25
    assert sum(x["n"] for x in result["bins"]) == 2


def predictions():
    return pd.DataFrame(
        {
            "id": [f"x{i}" for i in range(8)],
            "group": [f"g{i // 2}" for i in range(8)],
            "dataset": ["a"] * 8,
            "seed": [1729] * 8,
            "y_true": [0, 1] * 4,
            "prediction": [0, 1] * 4,
        }
    )


def test_bootstrap_identical_and_misalignment_rejected():
    a = predictions()
    result = paired_bootstrap(a, a, ["a", "b"], draws=30)
    assert result["accuracy"]["ci95"] == [0.0, 0.0]
    b = a.copy()
    b.loc[0, "y_true"] = 1
    with pytest.raises(ValueError):
        paired_bootstrap(a, b, ["a", "b"], draws=30)
    with pytest.raises(ValueError):
        mcnemar_exact(a, a, ["a", "b"])
    assert holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])


def test_raabin_preserves_test_a_excludes_linked_training():
    f = frame()
    f["dataset"] = "raabin"
    f["original_label"] = f.label
    f["collection"] = "train"
    f.loc[0, "collection"] = "test_a"
    result = split_manifest(f, "raabin")
    assert result.loc[0, "split"] == "test"
    assert (result.loc[1:2, "split"] == "excluded").all()

import numpy as np
import pytest
import torch
from scipy.optimize import nnls
from PIL import Image
from wbc.stain import nnls_two, decompose, fit_basis, optical_density
from wbc.crops import make_crops
from wbc.morphology import descriptors
from wbc.segmentation import overlap, reference_grid
from wbc.corruptions import corrupt


def test_nnls_matches_scipy_active_sets():
    rng = np.random.default_rng(27)
    b = np.array([[0.7, 0.2], [0.65, 0.7], [0.25, 0.68]])
    x = rng.uniform(0, 4, (100, 3))
    actual = nnls_two(x, b)
    expected = np.stack([nnls(b, row)[0] for row in x])
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_stain_reconstruction_matches_nnls():
    rgb = torch.rand(2, 3, 12, 12)
    b = torch.tensor([[0.7, 0.2], [0.65, 0.7], [0.25, 0.68]])
    actual = decompose(rgb, b).numpy()
    od = optical_density(rgb.permute(0, 2, 3, 1).numpy() * 255).reshape(-1, 3)
    c = nnls_two(od, b.numpy())
    expected = np.clip(256 * np.exp(-c[:, :, None] * b.numpy().T) - 1, 0, 255) / 255
    expected = expected.reshape(2, 12, 12, 2, 3).transpose(0, 3, 4, 1, 2)
    np.testing.assert_allclose(actual, expected, atol=3e-6)


def test_basis_reproducible_and_empty_rejected():
    with pytest.raises(ValueError):
        fit_basis([np.full((20, 3), 255.0)])
    rgb = np.random.default_rng(2).integers(30, 230, (1000, 3))
    a, _ = fit_basis([rgb], max_pixels=500, iterations=4)
    b, _ = fit_basis([rgb], max_pixels=500, iterations=4)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=0), 1, atol=1e-6)


def test_expansion_padding_is_invalid_and_white():
    rgb = Image.new("RGB", (40, 30), (70, 90, 140))
    crops, validity, extent = make_crops(rgb, [0.8, 1, 1.2], 32)
    assert extent.tolist() == [[32, 24], [40, 30], [48, 36]]
    assert validity[1].all()
    assert not validity[2, 0, 0, 0]
    assert crops[2, :, 0, 0].tolist() == [1.0, 1.0, 1.0]


def test_ratio_cytoplasm_not_whole_cell_and_invalid_rules():
    labels = np.full((12, 12), 2)
    labels[3:9, 3:9] = 1
    labels[4:8, 4:8] = 0
    values = descriptors(labels, np.ones_like(labels, bool), [12, 12])
    assert values[0] == pytest.approx(16 / 20)
    assert values[3] == pytest.approx(4.0)
    assert values[6] == pytest.approx(0.0)
    labels[0:4, 5] = 1
    assert np.isnan(descriptors(labels, np.ones_like(labels, bool), [12, 12])).all()


def test_compartment_empty_conventions_and_area_majority():
    a = np.zeros((8, 8), bool)
    assert overlap(a, a) == {"dice": 1.0, "iou": 1.0, "boundary_f1": 1.0}
    b = a.copy()
    b[2:4, 2:4] = True
    assert overlap(a, b)["dice"] == 0
    labels = reference_grid(b, np.ones_like(a), size=4)
    assert labels[1, 1] == 0
    assert labels[0, 0] == 1
    with pytest.raises(ValueError):
        reference_grid(b, a)


@pytest.mark.parametrize("kind", ["stain", "gamma", "blur", "jpeg"])
def test_corruptions_deterministic(kind):
    image = Image.fromarray(np.random.default_rng(1).integers(0, 255, (24, 24, 3), dtype=np.uint8))
    b = np.array([[0.7, 0.2], [0.65, 0.7], [0.25, 0.68]])
    first = np.asarray(corrupt(image, kind, 2, basis=b))
    second = np.asarray(corrupt(image, kind, 2, basis=b))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (24, 24, 3)

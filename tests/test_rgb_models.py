import pytest
import torch

from wbc.engine import parameter_groups
from wbc.losses import objective
from wbc.models import MODEL_CATALOG, RGBClassifier, build_model


@pytest.mark.parametrize("name", list(MODEL_CATALOG))
def test_every_rgb_encoder_produces_finite_class_logits(name):
    model = RGBClassifier(3, backbone=name).eval()
    size = 224 if name == "vit_b_16" else 64
    with torch.no_grad():
        result = model(torch.rand(1, 1, 3, size, size))
    assert result["logits"].shape == (1, 3)
    assert result["tokens"].shape == (1, 1, MODEL_CATALOG[name]["features"])
    assert torch.isfinite(result["logits"]).all()
    assert not hasattr(model, "prototypes")


def test_rgb_multiscale_gradient_and_classification_only_objective():
    model = RGBClassifier(3, backbone="compact_cnn", factors=[0.8, 1.0, 1.2])
    result = model(torch.rand(2, 3, 3, 32, 32))
    loss, terms = objective(result, torch.tensor([0, 1]), torch.zeros(2), model, torch.ones(3), {})
    loss.backward()
    assert model.backbone[0].body[0].weight.grad.abs().sum() > 0
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert terms["classification"] == loss
    assert all(terms[key].item() == 0 for key in ["scale", "prototype", "maturation"])


def test_frozen_encoder_keeps_batchnorm_and_optimizer_frozen():
    model = RGBClassifier(2, backbone="resnet18", freeze_backbone=True)
    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    model.train()
    optimizer = torch.optim.AdamW(parameter_groups(model, {}))
    loss = model(torch.rand(2, 1, 3, 32, 32))["logits"].square().mean()
    loss.backward()
    optimizer.step()
    for key, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in model.backbone.parameters())
    assert sum(len(g["params"]) for g in optimizer.param_groups) == 2


def test_factory_rejects_unknown_architecture_and_invalid_vit_size():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model({"classes": ["a", "b"], "model": {"architecture": "unknown"}})
    with pytest.raises(ValueError, match="no pretrained weights"):
        RGBClassifier(2, backbone="compact_cnn", pretrained=True)
    with pytest.raises(ValueError, match="requires"):
        build_model(
            {
                "classes": ["a", "b"],
                "image_size": 64,
                "model": {"architecture": "rgb", "backbone": "vit_b_16"},
            }
        )

import pytest
import torch
from wbc.model import WBCModel
from wbc.losses import objective, effective_weights
from wbc.engine import update_lr

torch.set_num_threads(2)


def test_gradient_path_and_all_loss_terms():
    model = WBCModel(3, backbone="debug", pretrained=False, seed=1729)
    image = torch.rand(2, 3, 3, 64, 64)
    valid = torch.ones(2, 3, 1, 64, 64)
    extent = torch.tensor([64.0, 64.0]).expand(2, 3, 2)
    output = model(image, valid, extent)
    assert output["logits"].shape == (2, 3)
    assert output["masks"].shape == (2, 3, 3, 8, 8)
    torch.testing.assert_close(output["masks"].sum(2), torch.ones(2, 3, 8, 8))
    loss, parts = objective(
        output,
        torch.tensor([0, 1]),
        torch.tensor([0.0, float("nan")]),
        model,
        effective_weights([3, 4, 5]),
        {},
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.prototypes.grad.abs().sum() > 0
    assert model.spatial.weight.grad.abs().sum() > 0
    assert parts["maturation"] >= 0
    assert output["raw_descriptors"].requires_grad is False


def test_full_convnext_tensor_specification():
    model = WBCModel(5, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(
            torch.rand(1, 3, 3, 224, 224), torch.ones(1, 3, 1, 224, 224), torch.full((1, 3, 2), 224.0)
        )
    assert output["masks"].shape == (1, 3, 3, 28, 28)
    assert output["tokens"].shape == (1, 3, 384)
    assert model.prototypes.numel() == 2304
    assert sum(p.numel() for p in model.projection.parameters()) == 246144


def test_missing_class_rejected_and_scheduler_endpoints():
    with pytest.raises(ValueError):
        effective_weights([10, 0, 3])
    weight = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([{"params": [weight], "peak_lr": 0.001}])
    update_lr(optimizer, 1, 5, 20)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0002)
    update_lr(optimizer, 5, 5, 20)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    update_lr(optimizer, 20, 5, 20)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.000001)


def test_unweighted_missing_maturation_and_single_scale():
    model = WBCModel(2, backbone="debug", factors=[1.0])
    output = model(torch.rand(2, 1, 3, 64, 64), torch.ones(2, 1, 1, 64, 64), torch.full((2, 1, 2), 64.0))
    loss, terms = objective(
        output, torch.tensor([0, 1]), torch.full((2,), float("nan")), model, torch.ones(2), {}
    )
    assert terms["maturation"].item() == 0
    assert terms["scale"].item() == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "name", ["resnet50", "mobilenet_v3_large", "efficientnet_b0", "efficientnet_v2_s", "swin_t"]
)
def test_comparator_adapters(name):
    from wbc.backbones import TorchvisionBackbone

    backbone = TorchvisionBackbone(name, pretrained=False).eval()
    with torch.no_grad():
        local, global_map = backbone(torch.rand(1, 3, 64, 64))
    assert local.shape == (1, backbone.local_channels, 8, 8)
    assert global_map.shape[1] == backbone.global_channels

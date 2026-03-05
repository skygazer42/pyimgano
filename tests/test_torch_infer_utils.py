def test_torch_inference_sets_eval_then_restores_train_mode() -> None:
    import torch

    from pyimgano.utils.torch_infer import torch_inference

    m = torch.nn.Linear(4, 3)
    m.train()
    assert m.training is True

    with torch_inference(m):
        assert m.training is False
        x = torch.randn(2, 4)
        y = m(x)
        assert y.shape == (2, 3)

    assert m.training is True


def test_resolve_torch_device_accepts_cpu() -> None:
    from pyimgano.utils.torch_infer import resolve_torch_device

    dev = resolve_torch_device("cpu")
    assert str(dev) == "cpu"

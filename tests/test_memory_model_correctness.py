from __future__ import annotations

import pytest


def test_memory_modules_do_not_use_random_untrained_entries() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.memae import MemoryModule
    from pyimgano.models.memseg import MemoryBank

    module = MemoryModule(mem_dim=4, fea_dim=3)
    assert isinstance(module.memory, torch.nn.Parameter)

    bank = MemoryBank(memory_size=4, feature_dim=2)
    bank.memory.copy_(torch.tensor([[1.0, 0.0], [100.0, 100.0], [100.0, 100.0], [100.0, 100.0]]))
    bank.memory_filled = 1
    distances, indices = bank.query(torch.tensor([[1.0, 0.0]]), k=3)
    assert distances.shape == (1, 1)
    assert indices.item() == 0


def test_memae_hard_shrink_preserves_weights_above_threshold() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.memae import hard_shrink_relu

    weights = torch.tensor([0.2, 0.4], dtype=torch.float32)
    shrunk = hard_shrink_relu(weights, threshold=0.3)
    torch.testing.assert_close(shrunk, torch.tensor([0.0, 0.4]))


def test_memae_memory_uses_paper_cosine_attention_and_shape() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.memae import MemoryModule

    module = MemoryModule(mem_dim=2, fea_dim=2, shrink_thres=0.0)
    with torch.no_grad():
        module.memory.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0]]))

    retrieved, attention = module(torch.tensor([[[[1.0]], [[0.0]]]]))
    expected_attention = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)

    assert attention.shape == (1, 2, 1, 1)
    torch.testing.assert_close(attention[0, :, 0, 0], expected_attention)
    torch.testing.assert_close(
        retrieved[0, :, 0, 0],
        torch.tensor([2.0, 0.0]) * expected_attention[0]
        + torch.tensor([0.0, 1.0]) * expected_attention[1],
    )


def test_memae_entropy_is_averaged_per_spatial_query() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.memae import memory_entropy

    attention = torch.tensor([[[[1.0, 0.5]], [[0.0, 0.5]]]])
    assert memory_entropy(attention).item() == pytest.approx(
        float(torch.log(torch.tensor(2.0))) / 2
    )


def test_memae_rgb_network_matches_paper_cifar_topology() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.memae import MemAE, MemAENetwork

    detector = MemAE(device="cpu")
    assert detector.mem_dim == 500
    assert detector.learning_rate == pytest.approx(1e-4)

    network = MemAENetwork(in_channels=3, mem_dim=5)
    encoder_convs = [layer for layer in network.encoder if isinstance(layer, torch.nn.Conv2d)]
    decoder_convs = [
        layer for layer in network.decoder if isinstance(layer, torch.nn.ConvTranspose2d)
    ]
    assert [(layer.in_channels, layer.out_channels) for layer in encoder_convs] == [
        (3, 64),
        (64, 128),
        (128, 128),
        (128, 256),
    ]
    assert [(layer.in_channels, layer.out_channels) for layer in decoder_convs] == [
        (256, 256),
        (256, 128),
        (128, 128),
        (128, 3),
    ]
    assert all(layer.kernel_size == (3, 3) for layer in encoder_convs + decoder_convs)
    assert not any(isinstance(layer, torch.nn.Sigmoid) for layer in network.decoder)

    reconstruction, encoded, retrieved, attention = network(torch.randn(2, 3, 32, 32))
    assert reconstruction.shape == (2, 3, 32, 32)
    assert encoded.shape == retrieved.shape == (2, 256, 2, 2)
    assert attention.shape == (2, 5, 2, 2)

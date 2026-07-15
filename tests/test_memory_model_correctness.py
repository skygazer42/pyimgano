from __future__ import annotations

import pytest


def test_memory_modules_do_not_use_random_untrained_entries() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.memae import MemoryModule
    from pyimgano.models.memseg import MemoryBank

    module = MemoryModule(mem_dim=4, fea_dim=3)
    assert isinstance(module.memory, torch.nn.Parameter)

    bank = MemoryBank(memory_size=4, feature_dim=2)
    bank.memory.copy_(
        torch.tensor([[1.0, 0.0], [100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])
    )
    bank.memory_filled = 1
    distances, indices = bank.query(torch.tensor([[1.0, 0.0]]), k=3)
    assert distances.shape == (1, 1)
    assert indices.item() == 0

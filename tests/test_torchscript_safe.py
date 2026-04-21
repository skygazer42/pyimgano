from __future__ import annotations

import warnings
from pathlib import Path

import pytest


def test_torchscript_safe_wrappers_suppress_deprecation_warnings(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    from pyimgano.utils.torchscript_safe import freeze_module, load_module, trace_module

    class ToyEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):  # noqa: ANN001, ANN201 - torchscript signature
            y = self.pool(self.conv(x))
            return y.flatten(1)

    model = ToyEmbed().eval()
    example = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    path = tmp_path / "toy_embed.pt"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scripted = trace_module(model, example)
        frozen = freeze_module(scripted)
        frozen.save(str(path))
        loaded = load_module(path, map_location=torch.device("cpu"))

    assert loaded is not None
    messages = [str(item.message) for item in caught]
    assert not any("deprecated" in message.lower() for message in messages), messages

from __future__ import annotations

from pathlib import Path


def test_padim_checkpoint_loading_prefers_safe_torch_load_helper() -> None:
    source = Path("pyimgano/models/padim.py").read_text(encoding="utf-8")

    assert "safe_torch_load(" in source

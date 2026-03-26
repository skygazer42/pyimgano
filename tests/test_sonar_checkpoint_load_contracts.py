from __future__ import annotations

from pathlib import Path


def test_draem_and_fastflow_checkpoint_loads_do_not_force_unsafe_pickle_mode() -> None:
    for rel_path in ("pyimgano/models/draem.py", "pyimgano/models/fastflow.py"):
        source = Path(rel_path).read_text(encoding="utf-8")
        assert "weights_only=False" not in source


def test_padim_checkpoint_loading_does_not_call_torch_load_directly() -> None:
    source = Path("pyimgano/models/padim.py").read_text(encoding="utf-8")

    assert "torch.load(" not in source

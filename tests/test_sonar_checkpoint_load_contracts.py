from __future__ import annotations

from pathlib import Path


def test_draem_and_fastflow_checkpoint_loads_do_not_force_unsafe_pickle_mode() -> None:
    for rel_path in ("pyimgano/models/draem.py", "pyimgano/models/fastflow.py"):
        source = Path(rel_path).read_text(encoding="utf-8")
        assert "weights_only=False" not in source

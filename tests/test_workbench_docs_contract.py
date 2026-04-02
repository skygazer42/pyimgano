from __future__ import annotations

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_workbench_docs_cover_weights_cache_and_checkpoint_reuse_policy() -> None:
    text = _read_text("docs/WORKBENCH.md")

    assert "does **not** ship model weights inside the wheel" in text
    assert "checkpoints/<cat>/..." in text
    assert "pyimgano-infer --from-run" in text
    assert "TORCH_HOME" in text
    assert "HF_HOME" in text
    assert "XDG_CACHE_HOME" in text

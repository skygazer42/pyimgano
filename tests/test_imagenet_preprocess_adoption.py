from __future__ import annotations

from pathlib import Path

_TARGETS = [
    "pyimgano/models/oneformore.py",
    "pyimgano/models/gcad.py",
    "pyimgano/models/realnet.py",
    "pyimgano/models/regad.py",
    "pyimgano/models/bayesianpf.py",
    "pyimgano/models/glad.py",
    "pyimgano/models/panda.py",
    "pyimgano/models/inctrl.py",
    "pyimgano/models/ast.py",
    "pyimgano/models/dst.py",
    "pyimgano/models/favae.py",
    "pyimgano/models/promptad.py",
]


def test_selected_models_use_shared_imagenet_preprocess_helper() -> None:
    for rel_path in _TARGETS:
        text = Path(rel_path).read_text(encoding="utf-8")
        assert "preprocess_imagenet_batch" in text, rel_path
        assert "mean = np.array([0.485, 0.456, 0.406])" not in text, rel_path
        assert "std = np.array([0.229, 0.224, 0.225])" not in text, rel_path

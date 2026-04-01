from __future__ import annotations

import re
from pathlib import Path


_TARGETS = [
    "pyimgano/models/spade.py",
    "pyimgano/models/padim.py",
    "pyimgano/models/dfm.py",
    "pyimgano/models/cflow.py",
    "pyimgano/models/stfpm.py",
    "pyimgano/models/simplenet.py",
    "pyimgano/models/patchcore.py",
    "pyimgano/models/one_svm_cnn.py",
    "pyimgano/models/realnet.py",
    "pyimgano/models/regad.py",
    "pyimgano/models/oneformore.py",
    "pyimgano/models/gcad.py",
    "pyimgano/models/promptad.py",
    "pyimgano/models/fastflow.py",
    "pyimgano/models/bgad.py",
    "pyimgano/models/dsr.py",
    "pyimgano/models/pni.py",
    "pyimgano/models/ast.py",
    "pyimgano/models/dst.py",
    "pyimgano/models/favae.py",
    "pyimgano/models/csflow.py",
    "pyimgano/models/rdplusplus.py",
    "pyimgano/models/bayesianpf.py",
    "pyimgano/models/glad.py",
    "pyimgano/models/panda.py",
    "pyimgano/models/inctrl.py",
]


def test_selected_models_use_shared_torchvision_loader() -> None:
    for rel_path in _TARGETS:
        text = Path(rel_path).read_text(encoding="utf-8")
        assert "load_torchvision_model" in text, rel_path
        assert re.search(r"models\.[A-Za-z0-9_]+\(pretrained=", text) is None, rel_path

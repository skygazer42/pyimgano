from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _write_score_model(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["input"], ["score"], axes=[1, 2, 3], keepdims=0)],
        "portable-score-model",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 4, 4])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [None])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save_model(model, str(path))


def _import_contract() -> dict[str, object]:
    return {
        "schema_family": "pyimgano-onnx-import",
        "schema_version": 1,
        "input": {
            "name": "input",
            "dtype": "float32",
            "layout": "NCHW",
            "color_space": "RGB",
            "size": [4, 4],
            "dynamic_batch": True,
            "resize": {"mode": "stretch", "interpolation": "bilinear"},
            "scale": {"divisor": 255.0},
            "normalize": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        },
        "outputs": {
            "score": {
                "name": "score",
                "transform": "identity",
                "score_order": "higher_is_more_anomalous",
            }
        },
    }


@pytest.mark.integration
def test_imported_onnx_artifact_survives_relocation_source_deletion_and_subprocess_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("onnxruntime")
    from pyimgano.artifacts import import_onnx

    source_run = tmp_path / "temporary source run"
    source_run.mkdir()
    source_model = source_run / "source-model.onnx"
    _write_score_model(source_model)
    source_artifact = source_run / "exports" / "onnx-artifact"
    import_onnx(source_model, contract=_import_contract(), out=source_artifact)

    relocated = tmp_path / "release artifacts" / "质检 模型" / "portable artifact"
    relocated.parent.mkdir(parents=True)
    shutil.copytree(source_artifact, relocated)
    shutil.rmtree(source_run)
    assert not source_run.exists()

    unrelated_cwd = tmp_path / "unrelated working directory" / "完全不同"
    unrelated_cwd.mkdir(parents=True)
    monkeypatch.chdir(unrelated_cwd)

    script = textwrap.dedent("""
        import json
        import os
        import sys

        import numpy as np
        import pyimgano
        from pyimgano.inference import infer, load_artifact

        artifact = sys.argv[1]
        runtime = load_artifact(artifact, format="onnx", backend="onnxruntime")
        try:
            result = infer(
                runtime,
                [np.full((6, 8, 3), 255, dtype=np.uint8)],
                input_format="rgb_u8_hwc",
                include_maps=False,
            )[0]
            print(json.dumps({
                "artifact_root": str(runtime.artifact_root),
                "cwd": os.getcwd(),
                "package_file": str(pyimgano.__file__),
                "score": result.score,
            }, ensure_ascii=False))
        finally:
            runtime.close()
        """)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(relocated)],
        cwd=unrelated_cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(payload["artifact_root"]).resolve() == relocated.resolve()
    assert Path(payload["cwd"]).resolve() == unrelated_cwd.resolve()
    package_file = Path(payload["package_file"]).resolve()
    assert package_file.is_file()
    if os.environ.get("PYIMGANO_E2E_EXPECT_WHEEL") == "1":
        checkout_root = Path(__file__).resolve().parents[1]
        try:
            package_file.relative_to(checkout_root)
        except ValueError:
            pass
        else:
            pytest.fail(f"subprocess imported checkout code instead of the wheel: {package_file}")
    assert payload["score"] == pytest.approx(1.0)

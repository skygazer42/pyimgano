from __future__ import annotations


class _FakeDim:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeExportAPI:
    Dim = _FakeDim


class _FakeONNX:
    def __init__(self) -> None:
        self.kwargs = None

    def export(
        self,
        model,
        dummy_input,
        output_path,
        *,
        input_names,
        output_names,
        opset_version,
        do_constant_folding,
        dynamo=None,
        dynamic_shapes=None,
        dynamic_axes=None,
    ) -> None:
        self.kwargs = {
            "model": model,
            "dummy_input": dummy_input,
            "output_path": output_path,
            "input_names": input_names,
            "output_names": output_names,
            "opset_version": opset_version,
            "do_constant_folding": do_constant_folding,
            "dynamo": dynamo,
            "dynamic_shapes": dynamic_shapes,
            "dynamic_axes": dynamic_axes,
        }


class _FakeTorch:
    export = _FakeExportAPI()

    def __init__(self) -> None:
        self.onnx = _FakeONNX()


def _export(fake_torch: _FakeTorch, *, opset: int) -> dict:
    from pyimgano.utils.onnx_export import export_torch_model

    export_torch_model(
        torch=fake_torch,
        model="model",
        dummy_input="input",
        output_path="model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
    )
    assert fake_torch.onnx.kwargs is not None
    return fake_torch.onnx.kwargs


def test_current_opset_uses_dynamo_dynamic_shapes() -> None:
    kwargs = _export(_FakeTorch(), opset=18)

    assert kwargs["dynamo"] is True
    assert kwargs["dynamic_axes"] is None
    assert kwargs["dynamic_shapes"][0][0].name == "batch"


def test_legacy_opset_preserves_dynamic_axes() -> None:
    kwargs = _export(_FakeTorch(), opset=17)

    assert kwargs["dynamo"] is False
    assert kwargs["dynamic_shapes"] is None
    assert kwargs["dynamic_axes"] == {
        "input": {0: "batch"},
        "output": {0: "batch"},
    }

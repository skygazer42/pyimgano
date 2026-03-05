from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyimgano-export-onnx")
    parser.add_argument(
        "--backbone",
        default="resnet18",
        help="Torchvision model name to export (classification head is stripped). Default: resnet18",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether to use torchvision pretrained weights. Default: false (offline-safe). "
            "When true, torchvision may download weights if not cached."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size used for ONNX export. Default: 224",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version. Default: 17",
    )
    parser.add_argument(
        "--dynamic-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export with dynamic batch dimension. Default: true",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify exported ONNX by loading via onnx + onnxruntime. Default: true",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .onnx path to write",
    )
    args = parser.parse_args(argv)

    image_size = int(args.image_size)
    if image_size <= 0:
        raise ValueError(f"--image-size must be > 0, got {args.image_size!r}")

    opset = int(args.opset)
    if opset <= 0:
        raise ValueError(f"--opset must be > 0, got {args.opset!r}")

    out_path = Path(str(args.out)).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from pyimgano.utils.optional_deps import require

    torch = require("torch", extra="torch", purpose="pyimgano-export-onnx")

    from pyimgano.utils.torchvision_safe import load_torchvision_backbone

    model, _transform = load_torchvision_backbone(
        str(args.backbone), pretrained=bool(args.pretrained)
    )
    model.eval()
    model.to(torch.device("cpu"))

    dummy = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)

    input_names = ["input"]
    output_names = ["embeddings"]
    dynamic_axes = None
    if bool(args.dynamic_batch):
        dynamic_axes = {
            "input": {0: "batch"},
            "embeddings": {0: "batch"},
        }

    try:
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=int(opset),
            do_constant_folding=True,
        )
    except ModuleNotFoundError as exc:
        if "onnxscript" in str(exc):
            raise ModuleNotFoundError(
                "torch.onnx.export requires the 'onnxscript' package in PyTorch >= 2.10. "
                "Install it with: pip install 'pyimgano[onnx]'"
            ) from exc
        raise

    if bool(args.verify):
        import numpy as np

        onnx = require("onnx", extra="onnx", purpose="pyimgano-export-onnx --verify")
        ort = require("onnxruntime", extra="onnx", purpose="pyimgano-export-onnx --verify")

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        x = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
        outputs = sess.run(None, {in_name: x})
        if not outputs:
            raise RuntimeError("ONNX runtime produced no outputs during --verify.")

        out0 = outputs[0]
        shape = getattr(out0, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) != 1:
            raise RuntimeError(
                "ONNX verification failed: expected batch dim 1, " f"got output[0].shape={shape}"
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

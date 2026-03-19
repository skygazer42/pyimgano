from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyimgano-export-torchscript")
    parser.add_argument(
        "--backbone",
        default="resnet18",
        help="Torchvision model name to export (classification head is stripped). Default: resnet18",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use torchvision pretrained weights. Default: false (offline-safe). When true, torchvision may download weights if not cached.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size used for tracing/scripting. Default: 224",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu|cuda. Default: cpu",
    )
    parser.add_argument(
        "--method",
        default="trace",
        choices=["trace", "script"],
        help="TorchScript export method. Default: trace",
    )
    parser.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply TorchScript inference optimizations (freeze; best-effort). Default: true",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output .pt path to write",
    )
    args = parser.parse_args(argv)

    image_size = int(args.image_size)
    if image_size <= 0:
        raise ValueError(f"--image-size must be > 0, got {args.image_size!r}")

    out_path = Path(str(args.out)).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from pyimgano.utils.optional_deps import require

    torch = require("torch", extra="torch", purpose="pyimgano-export-torchscript")

    dev = str(args.device).strip().lower()
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    device = torch.device(dev)

    from pyimgano.utils.torchvision_safe import load_torchvision_backbone

    model, _transform = load_torchvision_backbone(
        str(args.backbone), pretrained=bool(args.pretrained)
    )
    model.eval()
    model.to(device)

    example = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32, device=device)
    with torch.no_grad():
        if str(args.method) == "script":
            exported = torch.jit.script(model)
        else:
            exported = torch.jit.trace(model, example)

        if bool(args.optimize):
            # NOTE: torch.jit.optimize_for_inference currently produces artifacts
            # that fail to load on some torch versions (e.g. torch 2.4.0).
            # `freeze` gives most of the inference wins and remains loadable.
            exported = torch.jit.freeze(exported)

    exported.save(str(out_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

from __future__ import annotations


def test_benchmark_cli_pretrained_default_is_false() -> None:
    """Industrial default: do not enable pretrained weights implicitly.

    This prevents accidental network downloads (torchvision/openclip/diffusers).
    """

    from pyimgano.cli import _build_parser

    args = _build_parser().parse_args([])
    assert args.pretrained is False


def test_robust_cli_pretrained_default_is_false() -> None:
    """Robustness CLI should also be offline-safe by default."""

    from pyimgano.robust_cli import _build_parser

    args = _build_parser().parse_args([])
    assert args.pretrained is False


def test_infer_cli_direct_mode_default_does_not_enable_pretrained(capsys) -> None:
    """Regression: `pyimgano-infer --model ...` must not silently enable pretrained.

    We use `vision_anomalydino` because it only auto-loads a torch.hub DINOv2
    embedder when pretrained=True. With the offline-safe default, it should
    fail fast and ask for an explicit embedder or `--pretrained`.
    """

    from pyimgano.infer_cli import main

    rc = main(
        [
            "--model",
            "vision_anomalydino",
            "--input",
            "/does/not/exist.png",
        ]
    )

    out = capsys.readouterr()
    assert rc != 0
    assert "requires a patch embedder" in (out.err + out.out).lower()

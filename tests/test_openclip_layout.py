from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_openclip_runner_detects_sequence_first_attention_without_exception_probe() -> None:
    from pyimgano.models.openclip_backend import _run_openclip_transformer

    torch.manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=4,
        nhead=2,
        dropout=0.0,
        batch_first=False,
    ).eval()
    tokens = torch.randn(2, 5, 4)

    expected = layer(tokens.permute(1, 0, 2)).permute(1, 0, 2)
    actual = _run_openclip_transformer(layer, tokens)
    silently_wrong = layer(tokens)

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, silently_wrong)


def test_openclip_runner_detects_current_batch_first_custom_attention() -> None:
    from pyimgano.models.openclip_backend import _run_openclip_transformer

    class CurrentAttention:
        use_sdpa = True

    class CurrentBlock:
        attn = CurrentAttention()

    class CurrentTransformer:
        resblocks = [CurrentBlock()]

        def __call__(self, tokens):  # noqa: ANN001, ANN204
            assert tuple(tokens.shape[:2]) == (2, 5)
            return tokens + 1.0

    tokens = torch.randn(2, 5, 4)
    torch.testing.assert_close(
        _run_openclip_transformer(CurrentTransformer(), tokens),
        tokens + 1.0,
    )


def test_openclip_layout_detection_fails_closed_for_unknown_attention() -> None:
    from pyimgano.models.openclip_backend import _run_openclip_transformer

    transformer = SimpleNamespace(
        resblocks=[SimpleNamespace(attn=SimpleNamespace())],
    )
    with pytest.raises(RuntimeError, match="Cannot determine OpenCLIP token layout"):
        _run_openclip_transformer(transformer, torch.randn(2, 5, 4))

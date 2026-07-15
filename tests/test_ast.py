from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyimgano.models.ast import (  # noqa: E402
    ASTCouplingBlock,
    ASTStudent,
    ASTTeacherFlow,
    VisionAST,
)


def test_ast_defaults_match_paper_rgb_protocol() -> None:
    detector = VisionAST(device="cpu", verbose=0)

    assert detector.backbone == "efficientnet_b5"
    assert detector.image_size == 768
    assert detector.feature_dim == 304
    assert detector.condition_dim == 32
    assert detector.kernel_sizes == (3, 3, 3, 5)
    assert detector.n_coupling_blocks == 4
    assert detector.teacher_hidden_channels == 1024
    assert detector.student_hidden_channels == 1024
    assert detector.student_blocks == 4
    assert detector.clamp == 3.0
    assert detector.negative_slope == 0.2
    assert detector.learning_rate == 2e-4
    assert detector.weight_decay == 1e-5
    assert detector.batch_size == 8
    assert detector.epochs == 240
    assert detector.pretrained_backbone is False


def test_ast_coupling_block_is_invertible() -> None:
    torch.manual_seed(3)
    block = ASTCouplingBlock(
        6,
        4,
        hidden_channels=8,
        kernel_size=3,
        clamp=3.0,
        seed=0,
        gamma_trick=False,
    ).eval()
    value = torch.randn(2, 6, 5, 5)
    condition = torch.randn(2, 4, 5, 5)

    output, logdet = block(value, condition)
    restored, inverse_logdet = block(output, condition, reverse=True)

    torch.testing.assert_close(restored, value, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(inverse_logdet, -logdet, atol=2e-5, rtol=2e-5)


def test_ast_teacher_and_student_match_paper_architecture() -> None:
    teacher = ASTTeacherFlow(
        feature_dim=6,
        condition_dim=4,
        hidden_channels=8,
        kernel_sizes=(3, 3, 3, 5),
        clamp=3.0,
    )
    student = ASTStudent(
        feature_dim=6,
        condition_dim=4,
        hidden_channels=8,
        n_blocks=4,
        negative_slope=0.2,
    )

    assert len(teacher.blocks) == 4
    assert [block.subnet_1.conv1.kernel_size for block in teacher.blocks] == [
        (3, 3),
        (3, 3),
        (3, 3),
        (5, 5),
    ]
    assert all(block.subnet_1.gamma.item() == 0 for block in teacher.blocks)
    assert student.input_conv.in_channels == 10
    assert student.input_conv.out_channels == 8
    assert len(student.residual_blocks) == 4
    assert student.activation.negative_slope == 0.2

    feature = torch.randn(2, 6, 4, 4)
    condition = torch.randn(2, 4, 4, 4)
    assert student(feature, condition).shape == feature.shape


def test_ast_losses_follow_author_equations() -> None:
    latent = torch.ones(1, 4, 2, 2)
    logdet = torch.ones(1, 2, 2)
    assert VisionAST._teacher_loss(latent, logdet).item() == pytest.approx(1.0)

    target = torch.ones(1, 4, 2, 2)
    output = torch.zeros_like(target)
    torch.testing.assert_close(VisionAST._student_map(target, output), torch.ones(1, 2, 2))


def test_ast_tiny_fit_map_and_checkpoint_roundtrip(monkeypatch, tmp_path) -> None:
    import pyimgano.models.ast as ast_module

    class _TinyFeatureExtractor(torch.nn.Module):
        out_channels = 4

        def __init__(self, backbone: str, *, pretrained: bool) -> None:
            super().__init__()
            assert backbone == "efficientnet_b5"
            assert pretrained is False
            self.projection = torch.nn.Conv2d(3, 4, kernel_size=1, bias=False)
            with torch.no_grad():
                self.projection.weight.fill_(0.25)
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        def train(self, mode: bool = True):
            del mode
            return super().train(False)

        def forward(self, images):  # noqa: ANN001, ANN201
            return torch.nn.functional.adaptive_avg_pool2d(self.projection(images), (4, 4))

    monkeypatch.setattr(ast_module, "ASTFeatureExtractor", _TinyFeatureExtractor)

    kwargs = {
        "pretrained_backbone": False,
        "image_size": 16,
        "feature_dim": 4,
        "condition_dim": 4,
        "n_coupling_blocks": 1,
        "teacher_hidden_channels": 4,
        "student_hidden_channels": 4,
        "student_blocks": 1,
        "kernel_sizes": (3,),
        "batch_size": 2,
        "epochs": 1,
        "device": "cpu",
        "verbose": 0,
        "random_state": 7,
    }
    rng = np.random.default_rng(7)
    images = rng.integers(0, 256, size=(4, 16, 16, 3), dtype=np.uint8)

    detector = VisionAST(**kwargs).fit(images)
    scores = detector.decision_function(images)
    maps = detector.predict_anomaly_map(images)

    assert scores.shape == (4,)
    assert maps.shape == (4, 16, 16)
    assert np.isfinite(scores).all()
    assert np.isfinite(maps).all()

    checkpoint = detector.save_checkpoint(tmp_path / "ast.pt")
    restored = VisionAST(**kwargs)
    restored.load_checkpoint(checkpoint)

    np.testing.assert_allclose(restored.decision_function(images), scores, rtol=0, atol=0)
    np.testing.assert_allclose(restored.predict_anomaly_map(images), maps, rtol=0, atol=0)


def test_ast_rejects_removed_synthetic_proxy_parameters() -> None:
    with pytest.raises(TypeError, match="Synthetic anomalies"):
        VisionAST(anomaly_ratio=0.3, device="cpu")

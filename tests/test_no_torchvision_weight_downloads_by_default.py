from __future__ import annotations


def test_no_torchvision_weight_downloads_by_default(monkeypatch) -> None:
    """Guardrail: our defaults must not trigger implicit model-weight downloads.

    Torchvision downloads weights via `torch.hub.load_state_dict_from_url`.
    In unit tests (and in many offline/airgapped industrial settings), this must
    never happen unless the user explicitly opts in.
    """

    import torch.hub

    def _blocked(*_args, **_kwargs):  # noqa: ANN001, ANN201 - test helper
        raise AssertionError("Torchvision weight download is forbidden by default.")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _blocked, raising=True)

    # Feature extractors should be safe by default.
    from pyimgano.features import create_feature_extractor

    ext = create_feature_extractor("torchvision_backbone")
    # Trigger model construction.
    ext.extract([__import__("numpy").zeros((32, 32, 3), dtype="uint8")])

    ext2 = create_feature_extractor("torchvision_multilayer")
    ext2.extract([__import__("numpy").zeros((32, 32, 3), dtype="uint8")])

    ext3 = create_feature_extractor("torchvision_vit_tokens")
    ext3.extract([__import__("numpy").zeros((32, 32, 3), dtype="uint8")])

    ext4 = create_feature_extractor("torchvision_backbone_gem", image_size=32)
    ext4.extract([__import__("numpy").zeros((32, 32, 3), dtype="uint8")])

    # Selected detectors historically used pretrained=True defaults. Instantiating
    # them must not attempt downloads unless explicitly enabled.
    from pyimgano.models import create_model

    create_model("efficient_ad")
    create_model("ae_resnet_unet")

    # CrossMAD should also be safe-by-default even when fitting (which triggers
    # embedding extraction).
    import numpy as np

    det = create_model(
        "vision_crossmad",
        backbone="resnet18",
        image_size=32,
        device="cpu",
        num_prototypes=2,
        pretrained=False,
        contamination=0.5,
    )
    det.fit([np.zeros((32, 32, 3), dtype="uint8")])

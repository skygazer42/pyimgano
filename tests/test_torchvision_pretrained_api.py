from __future__ import annotations

import warnings

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.mark.parametrize(
    ("model_name", "kwargs"),
    [
        (
            "vision_cutpaste",
            {
                "pretrained": False,
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
            },
        ),
        (
            "vision_devnet",
            {
                "pretrained": False,
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
            },
        ),
        (
            "vision_differnet",
            {
                "pretrained": False,
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
                "random_state": 0,
            },
        ),
        (
            "vision_memseg",
            {
                "pretrained": False,
                "device": "cpu",
                "memory_size": 32,
                "k_neighbors": 1,
                "use_segmentation_head": False,
            },
        ),
    ],
)
def test_selected_torchvision_detectors_do_not_use_deprecated_pretrained_api(
    model_name: str,
    kwargs: dict[str, object],
) -> None:
    from pyimgano.models import create_model

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_model(model_name, **kwargs)

    messages = [str(item.message) for item in caught]
    assert not any(
        "parameter 'pretrained' is deprecated" in message.lower()
        or "arguments other than a weight enum or `none`" in message.lower()
        for message in messages
    ), messages

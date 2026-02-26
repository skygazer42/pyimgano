import pytest


def test_get_activation_by_name_known_values() -> None:
    import torch.nn as nn

    from pyimgano.utils.torch_activations import get_activation_by_name

    assert isinstance(get_activation_by_name("relu"), nn.ReLU)
    assert isinstance(get_activation_by_name("ReLU"), nn.ReLU)
    assert isinstance(get_activation_by_name("tanh"), nn.Tanh)
    assert isinstance(get_activation_by_name("sigmoid"), nn.Sigmoid)
    assert isinstance(get_activation_by_name("identity"), nn.Identity)


def test_get_activation_by_name_rejects_unknown() -> None:
    from pyimgano.utils.torch_activations import get_activation_by_name

    with pytest.raises(ValueError, match="Unknown activation"):
        get_activation_by_name("__does_not_exist__")


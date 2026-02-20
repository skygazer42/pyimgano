import pytest

from pyimgano.models import create_model


def test_anomalib_backend_missing_dep():
    # The model is registered, but should raise a helpful ImportError unless
    # pyimgano[anomalib] is installed.
    with pytest.raises(ImportError):
        create_model("vision_patchcore_anomalib", checkpoint_path="dummy.ckpt")


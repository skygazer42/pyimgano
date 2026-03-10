import pyimgano.models as models


def test_parallel_algorithm_family_placeholders_register():
    available = set(models.list_models())
    expected = {
        "vision_visionad",
        "vision_univad",
        "vision_filopp",
        "vision_adaclip",
        "vision_aaclip",
        "vision_one_to_normal",
        "vision_logsad",
        "vision_anogen_adapter",
    }
    assert expected.issubset(available)

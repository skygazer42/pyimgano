from pyimgano.defects import extract_defects_from_anomaly_map


def test_defects_public_imports() -> None:
    assert callable(extract_defects_from_anomaly_map)


from __future__ import annotations


def test_dataset_catalog_mvtec_prefers_on_disk_subset(tmp_path) -> None:
    from pyimgano.datasets.catalog import list_dataset_categories

    root = tmp_path / "mvtec"
    (root / "bottle").mkdir(parents=True)
    (root / "not_a_category").mkdir(parents=True)

    cats = list_dataset_categories(dataset="mvtec", root=str(root))
    assert cats == ["bottle"]


def test_dataset_catalog_visa_uses_visa_pytorch_root(tmp_path) -> None:
    from pyimgano.datasets.catalog import list_dataset_categories

    root = tmp_path / "visa"
    (root / "visa_pytorch" / "candle").mkdir(parents=True)
    (root / "visa_pytorch" / "capsules").mkdir(parents=True)

    cats = list_dataset_categories(dataset="visa", root=str(root))
    assert cats == ["candle", "capsules"]


def test_dataset_catalog_unknown_dataset_raises(tmp_path) -> None:
    from pyimgano.datasets.catalog import list_dataset_categories

    try:
        list_dataset_categories(dataset="unknown", root=str(tmp_path))
    except ValueError as exc:
        assert "unknown dataset" in str(exc).lower()
        return
    raise AssertionError("Expected ValueError")


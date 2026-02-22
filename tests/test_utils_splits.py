from __future__ import annotations

from pyimgano.utils.splits import split_train_calibration


def test_split_train_calibration_is_deterministic_and_disjoint() -> None:
    paths = [f"img_{i}.png" for i in range(10)]

    train1, cal1 = split_train_calibration(paths, calibration_fraction=0.2, seed=0)
    train2, cal2 = split_train_calibration(paths, calibration_fraction=0.2, seed=0)

    assert train1 == train2
    assert cal1 == cal2
    assert set(train1).isdisjoint(set(cal1))
    assert sorted(train1 + cal1) == sorted(paths)
    assert len(cal1) == 2


def test_split_train_calibration_single_item_keeps_train_nonempty() -> None:
    paths = ["only.png"]
    train, cal = split_train_calibration(paths, calibration_fraction=0.2, seed=0)
    assert train == paths
    assert cal == []

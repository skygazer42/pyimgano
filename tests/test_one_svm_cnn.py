from __future__ import annotations


def test_one_class_cnn_exposes_seeded_pca_random_state() -> None:
    from pyimgano.models.one_svm_cnn import ImageAnomalyDetector

    detector = ImageAnomalyDetector(feature_type="histogram", random_state=7)

    assert detector.pca.random_state == 7

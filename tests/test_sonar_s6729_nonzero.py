import numpy as np


def test_defect_bank_crop_to_mask_basic():
    from pyimgano.synthesis.defect_bank import _crop_to_mask

    img = np.zeros((10, 12, 3), dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 3:7] = 255

    crop_img, crop_mask, meta = _crop_to_mask(img, mask)

    assert crop_img.dtype == np.uint8
    assert crop_mask.dtype == np.uint8
    assert crop_img.shape[:2] == crop_mask.shape[:2]
    assert crop_img.shape[0] == 3
    assert crop_img.shape[1] == 4
    assert meta["crop_xyxy"] == (3, 2, 6, 4)


def test_select_templates_executes_nonzero_line():
    # These modules have identical "select templates via KMeans then index
    # labels==k" logic. This test ensures the codepath executes (and is covered)
    # for each file where we replaced np.where(...) with np.nonzero(...).
    from pyimgano.models import phase_corr_map, ssim, ssim_map, ssim_struct, template_ncc_map

    imgs = [
        np.zeros((16, 16), dtype=np.uint8),
        np.full((16, 16), 32, dtype=np.uint8),
        np.full((16, 16), 200, dtype=np.uint8),
    ]

    modules = (phase_corr_map, template_ncc_map, ssim_map, ssim, ssim_struct)
    for m in modules:
        templates = m._select_templates(imgs, n_templates=2, random_state=0)  # type: ignore[attr-defined]
        assert templates
        assert 1 <= len(templates) <= 2
        assert all(isinstance(t, np.ndarray) for t in templates)
        assert all(t.dtype == np.uint8 for t in templates)


def test_lscp_get_competent_detectors_nonzero_line():
    from pyimgano.models.lscp import CoreLSCP

    det = CoreLSCP(detector_list=[object(), object()], n_bins=1)
    scores = np.asarray([0.1, 0.2, 0.15, 0.18], dtype=np.float64)
    idx = det._get_competent_detectors(scores)

    assert idx.dtype == np.int64
    assert idx.shape == (scores.shape[0],)
    assert np.array_equal(idx, np.arange(scores.shape[0], dtype=np.int64))


def test_postprocessing_nms_executes_nonzero_line():
    from pyimgano.utils.dl_integration import PostProcessing

    boxes = np.asarray(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 9.0, 9.0],  # overlaps heavily with box 0
            [20.0, 20.0, 30.0, 30.0],  # far away
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

    keep = PostProcessing.non_max_suppression(
        boxes, scores, iou_threshold=0.5, score_threshold=0.0
    )
    assert isinstance(keep, np.ndarray)
    assert keep.dtype == np.int32
    assert keep.size >= 1

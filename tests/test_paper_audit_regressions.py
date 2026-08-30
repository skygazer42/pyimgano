from __future__ import annotations

import numpy as np
import pytest


def test_lid_uses_reciprocal_hill_mle() -> None:
    from pyimgano.models.lid import _lid_from_knn_distances

    distances = np.asarray([[1.0, 2.0, 4.0]], dtype=np.float64)
    actual = _lid_from_knn_distances(distances, eps=1e-12)[0]
    expected = -1.0 / np.mean(np.log(distances / distances[:, [-1]]))

    assert actual == pytest.approx(float(expected))


def test_extra_trees_density_is_query_batch_invariant() -> None:
    from pyimgano.models.extra_trees_density import CoreExtraTreesDensity

    train = np.random.default_rng(0).normal(size=(40, 6))
    query = np.zeros((1, 6), dtype=np.float64)
    detector = CoreExtraTreesDensity(n_estimators=10, random_state=0).fit(train)

    single = detector.decision_function(query)[0]
    repeated = detector.decision_function(np.repeat(query, 10, axis=0))[0]

    assert repeated == pytest.approx(single)


def test_pca_default_keeps_a_residual_direction() -> None:
    from pyimgano.models.pca import CorePCA

    train = np.random.default_rng(0).normal(size=(40, 5))
    detector = CorePCA().fit(train)

    assert detector.n_selected_components_ is not None
    assert detector.n_selected_components_ < train.shape[1]
    assert float(np.max(detector.decision_scores_)) > 1e-6


def test_qmcd_far_out_of_support_scores_higher() -> None:
    from pyimgano.models.qmcd import CoreQMCD

    train = np.random.default_rng(0).normal(size=(40, 6))
    detector = CoreQMCD().fit(train)
    far = np.full((1, 6), 12.0, dtype=np.float64)

    assert np.all(detector.decision_scores_ >= 0.0)
    assert detector.decision_function(far)[0] > float(np.max(detector.decision_scores_))


def test_hbos_out_of_support_value_does_not_inherit_edge_density() -> None:
    from pyimgano.models.hbos import CoreHBOS

    train = np.concatenate((np.zeros(10), np.ones(90))).reshape(-1, 1)
    detector = CoreHBOS(n_bins=2).fit(train)

    dense_edge, far = detector.decision_function(np.asarray([[1.0], [100.0]]))

    assert far > dense_edge


def test_inne_rejects_singleton_subsamples() -> None:
    from pyimgano.models.inne import CoreINNE

    train = np.random.default_rng(0).normal(size=(10, 3))

    with pytest.raises(ValueError, match="at least 2"):
        CoreINNE(max_samples=1, random_state=0).fit(train)


def test_sod_uses_actual_reference_set_size() -> None:
    from pyimgano.models.sod import CoreSOD

    detector = CoreSOD(n_neighbors=10, ref_set=9, alpha=0.8)
    reference = np.asarray([[0.0, 0.0], [2.0, 1.0], [4.0, 2.0]], dtype=np.float64)
    observation = np.asarray([2.0, 4.0], dtype=np.float64)

    actual = detector._score_one(observation, reference)
    means = np.mean(reference, axis=0)
    var_total = float(np.sum(np.square(reference - means)) / reference.shape[0])
    var_expect = detector.alpha * var_total / reference.shape[1]
    mask = np.var(reference, axis=0) < var_expect
    expected = np.sqrt(np.sum(np.square(observation - means)[mask]) / np.sum(mask))

    assert actual == pytest.approx(float(expected))


def test_cblof_rejects_cluster_partition_without_alpha_beta_intersection() -> None:
    from pyimgano.models.cblof import CoreCBLOF

    detector = CoreCBLOF(n_clusters=3, alpha=0.99, beta=100.0)
    detector.cluster_sizes_ = np.asarray([4, 3, 3])
    detector.n_clusters_ = 3

    with pytest.raises(ValueError, match="could not separate"):
        detector._set_small_large_clusters(n_samples=10)


def test_feature_bagging_never_draws_the_full_feature_space() -> None:
    from pyimgano.models.feature_bagging import CoreFeatureBagging

    detector = CoreFeatureBagging(n_estimators=20, max_features=1.0, random_state=0)
    detector.fit(np.arange(240, dtype=np.float64).reshape(40, 6))

    assert max(len(features) for features in detector.estimators_features_) <= 5


def test_feature_bagging_implements_paper_combination_rules() -> None:
    from pyimgano.models.feature_bagging import CoreFeatureBagging

    score_matrix = np.asarray(
        [[10.0, 0.0], [8.0, 5.0], [1.0, 10.0], [0.0, 8.0]],
        dtype=np.float64,
    )

    cumulative = CoreFeatureBagging(combination="cumulative_sum")
    breadth_first = CoreFeatureBagging(combination="breadth_first")

    assert np.array_equal(cumulative._combine_scores(score_matrix), [10.0, 13.0, 11.0, 8.0])
    assert np.array_equal(breadth_first._combine_scores(score_matrix), [4.0, 2.0, 3.0, 1.0])


@pytest.mark.parametrize("name", ["loci", "sos", "imdd"])
def test_dataset_level_novelty_extensions_are_batch_invariant(name: str) -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(24, 3))
    query = np.full((1, 3), 7.0, dtype=np.float64)
    companions = rng.normal(size=(5, 3))

    if name == "loci":
        from pyimgano.models.loci import CoreLOCI

        detector = CoreLOCI().fit(train)
    elif name == "sos":
        from pyimgano.models.sos import CoreSOS

        detector = CoreSOS(perplexity=4.5).fit(train)
    else:
        from pyimgano.models.imdd import CoreIMDD

        detector = CoreIMDD(n_iter=5, random_state=0).fit(train)

    single = detector.decision_function(query)[0]
    with_companions = detector.decision_function(np.vstack((query, companions)))[0]
    repeated = detector.decision_function(query)[0]

    assert with_companions == pytest.approx(single)
    assert repeated == pytest.approx(single)


def test_loci_optimized_novelty_score_matches_augmented_dataset_definition() -> None:
    from pyimgano.models.loci import CoreLOCI

    train = np.random.default_rng(7).normal(size=(25, 3))
    query = np.asarray([[4.0, -3.0, 2.0]], dtype=np.float64)
    detector = CoreLOCI().fit(train)

    expected = detector._calculate_scores(np.vstack((train, query)))[-1]

    assert detector.decision_function(query)[0] == pytest.approx(float(expected))


def test_loda_score_is_single_weighted_projection_average() -> None:
    from pyimgano.models.loda import CoreLODA

    train = np.random.default_rng(0).normal(size=(30, 4))
    detector = CoreLODA(n_bins=5, n_random_cuts=7, random_state=0).fit(train)
    query = train[:1]

    manual = 0.0
    for index in range(detector.n_random_cuts):
        projected = detector.projections_[index].dot(query.T)
        limits = detector.limits_[index]
        histogram = detector.histograms_[index]
        bin_index = np.searchsorted(limits, projected, side="right") - 1
        bin_index = np.clip(bin_index, 0, histogram.size - 1)
        manual += float(-detector.weights[index] * np.log(histogram[bin_index][0]))

    assert detector.decision_function(query)[0] == pytest.approx(manual)


def test_loop_nplof_uses_zero_mean_rms() -> None:
    from pyimgano.models.loop import CoreLoOP

    train = np.random.default_rng(0).normal(size=(30, 4))
    detector = CoreLoOP(n_neighbors=5, lambda_=3.0).fit(train)
    _distances, indices = detector._nn.kneighbors(train, n_neighbors=6, return_distance=True)
    neighbor_indices = np.asarray(indices[:, 1:], dtype=np.int64)
    mean_neighbor_pdist = np.mean(detector._pdist_train[neighbor_indices], axis=1)
    plof = detector._pdist_train / (mean_neighbor_pdist + detector.eps) - 1.0
    expected = detector.lambda_ * np.sqrt(np.mean(np.square(plof))) + detector.eps

    assert detector._nplof == pytest.approx(float(expected))


def test_dbscan_no_core_fallback_remains_a_novelty_score() -> None:
    from pyimgano.models.dbscan import CoreDBSCAN

    train = np.random.default_rng(3).normal(size=(8, 3))
    detector = CoreDBSCAN(eps=1e-9, min_samples=4).fit(train)

    near, far = detector.decision_function(
        np.asarray([[0.1, 1.1, 2.1], [100.0, 100.0, 100.0]], dtype=np.float64)
    )

    assert detector.used_training_fallback_ is True
    assert np.ptp(detector.decision_scores_) > 0.0
    assert far > near


def test_neighborhood_entropy_preserves_absolute_isolation() -> None:
    from pyimgano.models.neighborhood_entropy import CoreNeighborhoodEntropy

    train = np.random.default_rng(0).normal(scale=0.2, size=(40, 3))
    detector = CoreNeighborhoodEntropy(n_neighbors=5).fit(train)
    near, far = detector.decision_function(
        np.asarray([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]], dtype=np.float64)
    )

    assert far > near


def test_rgraph_uses_sparse_self_representation_transition() -> None:
    from pyimgano.models.rgraph import CoreRGraph

    rng = np.random.default_rng(0)
    inliers = np.column_stack([rng.normal(size=(40, 2)), np.zeros(40)])
    train = np.vstack([inliers, np.asarray([[0.0, 0.0, 5.0]])])
    detector = CoreRGraph(
        gamma=50.0,
        n_nonzero=5,
        transition_steps=20,
        preprocessing=False,
    ).fit(train)

    representation = detector.representation_matrix_
    transition = detector.transition_matrix_
    assert representation is not None
    assert transition is not None
    assert np.all(np.diag(representation) == 0.0)
    assert np.all(np.count_nonzero(representation, axis=1) <= 5)
    row_sums = np.sum(transition, axis=1)
    assert np.all(np.isclose(row_sums, 0.0) | np.isclose(row_sums, 1.0))
    assert detector.decision_scores_ is not None
    assert detector.decision_scores_[-1] > float(np.mean(detector.decision_scores_[:-1]))


def test_rod_enumerates_all_subspaces_by_default() -> None:
    from math import comb

    from pyimgano.models.rod import CoreROD

    train = np.random.default_rng(0).normal(size=(20, 6))
    detector = CoreROD().fit(train)

    assert detector.subspaces_ is not None
    assert len(detector.subspaces_) == comb(6, 3)

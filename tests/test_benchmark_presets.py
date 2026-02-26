from __future__ import annotations


def test_build_benchmark_configs_supports_new_classical_detectors() -> None:
    from pyimgano.benchmark import build_benchmark_configs

    cfg = build_benchmark_configs(
        [
            "ecod",
            "copod",
            "knn",
            "loop",
            "ldof",
            "odin",
            "rrcf",
            "hst",
            "mahalanobis",
            "dtc",
            "rzscore",
            "pca_md",
        ]
    )

    # Existing
    assert cfg["ECOD"]["model_name"] == "vision_ecod"
    assert cfg["COPOD"]["model_name"] == "vision_copod"

    # New classical baselines
    assert cfg["LoOP"]["model_name"] == "vision_loop"
    assert cfg["LDOF"]["model_name"] == "vision_ldof"
    assert cfg["ODIN"]["model_name"] == "vision_odin"
    assert cfg["RRCF"]["model_name"] == "vision_rrcf"
    assert cfg["HST"]["model_name"] == "vision_hst"
    assert cfg["Mahalanobis"]["model_name"] == "vision_mahalanobis"
    assert cfg["DTC"]["model_name"] == "vision_dtc"
    assert cfg["RZScore"]["model_name"] == "vision_rzscore"
    assert cfg["PCA-MD"]["model_name"] == "vision_pca_md"


def test_build_benchmark_configs_skips_unknown_algorithms() -> None:
    from pyimgano.benchmark import build_benchmark_configs

    cfg = build_benchmark_configs(["ecod", "definitely-not-a-real-model"])
    assert "ECOD" in cfg
    assert "definitely-not-a-real-model" not in cfg


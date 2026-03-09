def test_registry_model_info_includes_capabilities_payload() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import model_info

    info = model_info("vision_ecod")

    assert info["name"] == "vision_ecod"
    assert "capabilities" in info
    caps = info["capabilities"]
    assert "input_modes" in caps
    assert "paths" in caps["input_modes"]
    assert isinstance(caps["supports_pixel_map"], bool)
    assert isinstance(caps["supports_checkpoint"], bool)
    assert isinstance(caps["requires_checkpoint"], bool)
    assert isinstance(caps["supports_save_load"], bool)

    # Convenience aliases
    assert info["input_modes"] == caps["input_modes"]
    assert info["supports_save_load"] == caps["supports_save_load"]


def test_core_models_are_reported_as_features_input_mode() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import model_info

    info = model_info("core_deep_svdd")
    assert info["name"] == "core_deep_svdd"
    assert "capabilities" in info
    assert info["capabilities"]["input_modes"] == ["features"]
    assert info["input_modes"] == ["features"]


def test_registry_model_info_exposes_verified_paper_year_metadata() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import model_info

    ocsvm = model_info("vision_ocsvm")
    fastflow = model_info("vision_fastflow")
    dbscan = model_info("vision_dbscan")
    deep_svdd = model_info("vision_deep_svdd")

    assert ocsvm["metadata"]["year"] == 2001
    assert "Support" in ocsvm["metadata"]["paper"] or "One-Class SVM" in ocsvm["metadata"]["paper"]

    assert fastflow["metadata"]["year"] == 2021
    assert "FastFlow" in fastflow["metadata"]["paper"]

    assert dbscan["metadata"]["year"] == 1996
    assert "Density-Based Algorithm" in dbscan["metadata"]["paper"]

    assert deep_svdd["metadata"]["year"] == 2018
    assert "Deep One-Class Classification" in deep_svdd["metadata"]["paper"]


def test_backend_alias_model_info_inherits_verified_algorithm_metadata() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import model_info

    cfa_alias = model_info("vision_cfa_anomalib")
    cflow_alias = model_info("vision_cflow_anomalib")
    csflow_alias = model_info("vision_csflow_anomalib")
    anomalydino = model_info("vision_anomalydino")
    dfm = model_info("vision_dfm")
    dfm_alias = model_info("vision_dfm_anomalib")
    dinomaly_alias = model_info("vision_dinomaly_anomalib")
    dsr_alias = model_info("vision_dsr_anomalib")
    fre_alias = model_info("vision_fre_anomalib")
    uflow_alias = model_info("vision_uflow_anomalib")
    patchcore_alias = model_info("vision_patchcore_anomalib")
    patchcore_inspection = model_info("vision_patchcore_inspection_checkpoint")
    patchcore_lite = model_info("vision_patchcore_lite")
    patchcore_online = model_info("vision_patchcore_online")
    padim_alias = model_info("vision_padim_anomalib")
    reverse_dist = model_info("vision_reverse_distillation")
    reverse_dist_alias = model_info("vision_reverse_dist")
    rd_alias = model_info("vision_reverse_distillation_anomalib")
    softpatch = model_info("vision_softpatch")
    stfpm_alias = model_info("vision_stfpm_anomalib")
    draem_alias = model_info("vision_draem_anomalib")
    fastflow_alias = model_info("vision_fastflow_anomalib")
    efficientad_alias = model_info("vision_efficientad_anomalib")
    ganomaly_alias = model_info("vision_ganomaly_anomalib")
    rkde_alias = model_info("vision_rkde_anomalib")
    supersimplenet_alias = model_info("vision_supersimplenet_anomalib")
    winclip_alias = model_info("vision_winclip_anomalib")
    ssim_template = model_info("ssim_template")
    ssim_template_map = model_info("ssim_template_map")
    ssim_struct = model_info("ssim_struct")
    ssim_struct_map = model_info("ssim_struct_map")

    assert patchcore_alias["metadata"]["year"] == 2022
    assert "Total Recall" in patchcore_alias["metadata"]["paper"]

    assert patchcore_inspection["metadata"]["year"] == 2022
    assert "Total Recall" in patchcore_inspection["metadata"]["paper"]

    assert patchcore_lite["metadata"]["year"] == 2022
    assert "Total Recall" in patchcore_lite["metadata"]["paper"]

    assert patchcore_online["metadata"]["year"] == 2022
    assert "Total Recall" in patchcore_online["metadata"]["paper"]

    assert padim_alias["metadata"]["year"] == 2020
    assert "PaDiM" in padim_alias["metadata"]["paper"]

    assert softpatch["metadata"]["year"] == 2022
    assert "SoftPatch" in softpatch["metadata"]["paper"]

    assert cfa_alias["metadata"]["year"] == 2022
    assert "Coupled-hypersphere-based Feature Adaptation" in cfa_alias["metadata"]["paper"]

    assert cflow_alias["metadata"]["year"] == 2022
    assert "Real-Time Unsupervised Anomaly Detection" in cflow_alias["metadata"]["paper"]

    assert csflow_alias["metadata"]["year"] == 2022
    assert "Cross-Scale" in csflow_alias["metadata"]["paper"]

    assert anomalydino["metadata"]["year"] == 2025
    assert "AnomalyDINO" in anomalydino["metadata"]["paper"]

    assert dfm["metadata"]["year"] == 2019
    assert "Probabilistic Modeling of Deep Features" in dfm["metadata"]["paper"]

    assert dfm_alias["metadata"]["year"] == 2019
    assert "Probabilistic Modeling of Deep Features" in dfm_alias["metadata"]["paper"]

    assert dinomaly_alias["metadata"]["year"] == 2025
    assert "Dinomaly" in dinomaly_alias["metadata"]["paper"]

    assert dsr_alias["metadata"]["year"] == 2022
    assert "Dual Subspace Re-Projection" in dsr_alias["metadata"]["paper"]

    assert fre_alias["metadata"]["year"] == 2023
    assert "FRE:" in fre_alias["metadata"]["paper"]

    assert reverse_dist["metadata"]["year"] == 2022
    assert "Reverse Distillation" in reverse_dist["metadata"]["paper"]

    assert reverse_dist_alias["metadata"]["year"] == 2022
    assert "Reverse Distillation" in reverse_dist_alias["metadata"]["paper"]

    assert rd_alias["metadata"]["year"] == 2022
    assert "Reverse Distillation" in rd_alias["metadata"]["paper"]

    assert stfpm_alias["metadata"]["year"] == 2021
    assert "Student-Teacher Feature Pyramid Matching" in stfpm_alias["metadata"]["paper"]

    assert draem_alias["metadata"]["year"] == 2021
    assert "DRAEM" in draem_alias["metadata"]["paper"]

    assert fastflow_alias["metadata"]["year"] == 2021
    assert "FastFlow" in fastflow_alias["metadata"]["paper"]

    assert efficientad_alias["metadata"]["year"] == 2024
    assert "EfficientAD" in efficientad_alias["metadata"]["paper"]

    assert ganomaly_alias["metadata"]["year"] == 2018
    assert "GANomaly" in ganomaly_alias["metadata"]["paper"]

    assert rkde_alias["metadata"]["year"] == 2019
    assert "Region Based Anomaly Detection" in rkde_alias["metadata"]["paper"]

    assert supersimplenet_alias["metadata"]["year"] == 2024
    assert "SuperSimpleNet" in supersimplenet_alias["metadata"]["paper"]

    assert ssim_template["metadata"]["year"] == 2004
    assert "Image Quality Assessment" in ssim_template["metadata"]["paper"]

    assert ssim_template_map["metadata"]["year"] == 2004
    assert "Image Quality Assessment" in ssim_template_map["metadata"]["paper"]

    assert ssim_struct["metadata"]["year"] == 2004
    assert "Image Quality Assessment" in ssim_struct["metadata"]["paper"]

    assert ssim_struct_map["metadata"]["year"] == 2004
    assert "Image Quality Assessment" in ssim_struct_map["metadata"]["paper"]

    assert uflow_alias["metadata"]["year"] == 2022
    assert "U-Flow" in uflow_alias["metadata"]["paper"]

    assert winclip_alias["metadata"]["year"] == 2023
    assert "WinCLIP" in winclip_alias["metadata"]["paper"]


def test_dfkde_alias_exposes_documented_paper_title_without_invented_year() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import model_info

    dfkde_alias = model_info("vision_dfkde_anomalib")

    assert dfkde_alias["metadata"]["paper"] == "Deep Feature Kernel Density Estimation"
    assert "year" not in dfkde_alias["metadata"]


def test_checkpoint_wrappers_expose_weights_source_defaults() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import model_info

    anomalib_ckpt = model_info("vision_patchcore_anomalib")
    inspection_ckpt = model_info("vision_patchcore_inspection_checkpoint")
    onnx_ckpt = model_info("vision_onnx_ecod")
    torchscript_ckpt = model_info("vision_torchscript_ecod")

    assert anomalib_ckpt["metadata"]["weights_source"] == "upstream-anomalib-checkpoint"
    assert inspection_ckpt["metadata"]["weights_source"] == "upstream-patchcore-inspection-checkpoint"
    assert onnx_ckpt["metadata"]["weights_source"] == "local-exported-onnx"
    assert torchscript_ckpt["metadata"]["weights_source"] == "local-exported-torchscript"

from __future__ import annotations


def test_cli_presets_pixel_stats_baselines_resolve() -> None:
    from pyimgano.cli_presets import list_model_presets, resolve_model_preset

    names = set(list_model_presets())
    for preset_name, expected_model in [
        ("industrial-pixel-mean-absdiff-map", "vision_pixel_mean_absdiff_map"),
        ("industrial-pixel-gaussian-map", "vision_pixel_gaussian_map"),
        ("industrial-pixel-mad-map", "vision_pixel_mad_map"),
        ("industrial-ssim-template-map", "ssim_template_map"),
        ("industrial-ssim-struct-map", "ssim_struct_map"),
        ("industrial-template-ncc-map", "vision_template_ncc_map"),
        ("industrial-phase-correlation-map", "vision_phase_correlation_map"),
    ]:
        assert preset_name in names
        preset = resolve_model_preset(preset_name)
        assert preset is not None
        assert preset.model == expected_model

        kwargs = dict(preset.kwargs)
        assert "resize_hw" in kwargs
        if preset_name.startswith("industrial-pixel-"):
            assert kwargs.get("color") in {"gray", "rgb"}

import json
import subprocess
import sys
from datetime import date
from pathlib import Path


def test_pyim_list_defaults_to_all_sections(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Models" in out
    assert "Families" in out
    assert "Preprocessing Schemes" in out
    assert "vision_patchcore" in out


def test_pyim_list_models_supports_family_filter(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--family", "patchcore"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_patchcore" in out
    assert "vision_abod" not in out


def test_pyim_list_families_outputs_json(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "families", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert any(item["name"] == "neighbors" and item["model_count"] > 0 for item in payload)


def test_pyim_list_families_includes_parallel_algorithm_families(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "families", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    names = {item["name"] for item in payload}
    assert {
        "visionad",
        "univad",
        "filopp",
        "adaclip",
        "aaclip",
        "one_to_normal",
        "logsad",
        "anogen",
    }.issubset(names)


def test_pyim_list_metadata_contract_outputs_json(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "metadata-contract", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(item["name"] == "paper" and item["requirement"] == "recommended" for item in payload)
    assert any(item["name"] == "family" and item["requirement"] == "required" for item in payload)


def test_pyim_audit_metadata_outputs_json_and_nonzero_exit(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--audit-metadata", "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["total_models"] > 0
    assert "required_missing_by_model" in payload
    assert "recommended_missing_by_model" in payload


def test_pyim_list_preprocessing_supports_deployable_filter(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "preprocessing", "--json", "--deployable-only"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert payload
    assert all(item["deployable"] is True for item in payload)
    assert any(item["name"] == "illumination-contrast-balanced" for item in payload)


def test_pyim_list_years_outputs_timeline_json(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "years", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(item["name"] == "2001" and item["model_count"] > 0 for item in payload)
    assert any(item["name"] == str(date.today().year) for item in payload)


def test_pyim_list_types_outputs_json(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "types", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert any(item["name"] == "deep-vision" and item["model_count"] > 0 for item in payload)
    assert any(item["name"] == "density-estimation" and item["model_count"] > 0 for item in payload)
    assert any(item["name"] == "flow-based" and item["model_count"] > 0 for item in payload)
    assert any(item["name"] == "one-class-svm" and item["model_count"] > 0 for item in payload)


def test_pyim_list_models_supports_year_and_type_filters(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2021", "--type", "deep-vision"])
    assert code == 0
    out = capsys.readouterr().out
    assert "cutpaste" in out
    assert "core_qmcd" not in out


def test_pyim_list_models_surfaces_verified_classical_years(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2000", "--type", "classical-vision"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_lof" in out
    assert "vision_ocsvm" not in out


def test_pyim_list_models_surfaces_verified_hbos_year(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2012", "--type", "classical-vision"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_hbos" in out


def test_pyim_list_models_surfaces_verified_dbscan_and_deep_svdd_years(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "1996", "--type", "clustering-based"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_dbscan" in out

    code = main(["--list", "models", "--year", "2018", "--type", "deep-vision"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_deep_svdd" in out


def test_pyim_list_models_supports_specific_method_type_filters(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--type", "flow-based"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_fastflow" in out
    assert "vision_alad" not in out


def test_pyim_flow_and_distillation_filters_include_backend_aliases(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2021", "--type", "flow-based"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_fastflow" in out
    assert "vision_fastflow_anomalib" in out

    code = main(["--list", "models", "--year", "2024", "--type", "distillation"])
    assert code == 0
    out = capsys.readouterr().out
    assert "efficient_ad" in out
    assert "vision_efficientad_anomalib" in out


def test_pyim_family_and_gan_filters_include_verified_backend_aliases(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2022", "--family", "patchcore"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_patchcore" in out
    assert "vision_patchcore_anomalib" in out
    assert "vision_patchcore_inspection_checkpoint" in out

    code = main(["--list", "models", "--year", "2018", "--type", "gan-based"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_alad" in out
    assert "vision_ganomaly_anomalib" in out


def test_pyim_ssim_template_family_is_discoverable_by_year(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2004", "--family", "template"])
    assert code == 0
    out = capsys.readouterr().out
    assert "ssim_template" in out
    assert "ssim_template_map" in out
    assert "ssim_struct" in out
    assert "ssim_struct_map" in out


def test_pyim_flow_and_reconstruction_filters_include_more_verified_backend_aliases(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2022", "--type", "flow-based"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_csflow_anomalib" in out
    assert "vision_uflow_anomalib" in out

    code = main(["--list", "models", "--year", "2022", "--type", "reconstruction"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_dsr_anomalib" in out


def test_pyim_cflow_and_reverse_distillation_aliases_are_discoverable(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2022", "--type", "flow-based"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_cflow_anomalib" in out

    code = main(["--list", "models", "--year", "2022", "--type", "distillation"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_reverse_distillation" in out
    assert "vision_reverse_dist" in out
    assert "vision_reverse_distillation_anomalib" in out


def test_pyim_dinomaly_and_fre_aliases_are_discoverable(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2025", "--type", "reconstruction"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_dinomaly_anomalib" in out

    code = main(["--list", "models", "--year", "2023", "--type", "reconstruction"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_fre_anomalib" in out


def test_pyim_dfm_family_is_discoverable_by_year_and_gaussian_type(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2019", "--type", "gaussian-distance"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_dfm" in out
    assert "vision_dfm_anomalib" in out


def test_pyim_softpatch_and_anomalydino_are_discoverable_by_industrial_filters(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2022", "--type", "memory-bank"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_softpatch" in out
    assert "vision_patchcore" in out
    assert "vision_patchcore_anomalib" in out
    assert "vision_patchcore_inspection_checkpoint" in out
    assert "vision_patchcore_lite" in out
    assert "vision_patchcore_online" in out

    code = main(["--list", "models", "--year", "2025", "--type", "neighbor-based"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_anomalydino" in out


def test_pyim_supersimplenet_alias_is_discoverable_by_year_and_raw_type_tag(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2024", "--type", "supersimplenet"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_supersimplenet_anomalib" in out


def test_pyim_rkde_alias_is_discoverable_by_density_estimation_type(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--year", "2019", "--type", "density-estimation"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_rkde_anomalib" in out


def test_pyim_density_estimation_type_includes_dfkde_alias_without_year_filter(capsys):
    from pyimgano.pyim_cli import main

    code = main(["--list", "models", "--type", "density-estimation"])
    assert code == 0
    out = capsys.readouterr().out
    assert "vision_dfkde_anomalib" in out


def test_pyim_supports_python_module_invocation() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyimgano.pyim_cli",
            "--list",
            "preprocessing",
            "--json",
            "--deployable-only",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert any(item["name"] == "illumination-contrast-balanced" for item in payload)

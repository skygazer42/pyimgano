import json


def test_train_cli_list_recipes_outputs_text(capsys):
    from pyimgano.train_cli import main

    code = main(["--list-recipes"])
    assert code == 0
    out = capsys.readouterr().out
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    assert lines[0].startswith("industrial-adapt")
    assert "industrial-adapt" in out
    assert "deploy_smoke_custom_cpu.json" in out
    assert "classical-structural-ecod" in out


def test_train_cli_list_recipes_text_marks_starterless_recipes(capsys):
    from pyimgano.train_cli import main

    code = main(["--list-recipes"])
    assert code == 0
    out = capsys.readouterr().out
    assert "anomalib-train [manual-only]" in out
    assert "classical-struct-iforest-synth [generated-at-runtime]" in out


def test_train_cli_list_recipes_outputs_json(capsys):
    from pyimgano.train_cli import main

    code = main(["--list-recipes", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert parsed[0] == "industrial-adapt"
    assert "industrial-adapt" in parsed
    assert "classical-structural-ecod" in parsed


def test_train_cli_list_recipes_uses_shared_listing_helper(monkeypatch):
    import pyimgano.train_cli as train_cli

    monkeypatch.setattr(train_cli, "list_recipes", lambda: ["recipe-a", "recipe-b"])

    calls = []
    monkeypatch.setattr(
        train_cli,
        "cli_listing",
        type(
            "_StubCliListing",
            (),
            {
                "emit_listing": staticmethod(
                    lambda items, **kwargs: calls.append((list(items), kwargs)) or 73
                )
            },
        ),
        raising=False,
    )

    code = train_cli.main(["--list-recipes", "--json"])
    assert code == 73
    assert calls == [(["recipe-a", "recipe-b"], {"json_output": True})]


def test_train_cli_recipe_info_outputs_json(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "industrial-adapt", "--json"])
    assert code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["name"] == "industrial-adapt"


def test_train_cli_recipe_info_outputs_curated_text_sections(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "industrial-adapt"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Recipe: industrial-adapt" in out
    assert "Tags: builtin, adaptation" in out
    assert "Default config: examples/configs/deploy_smoke_custom_cpu.json" in out
    assert "Starter configs:" in out
    assert "  - examples/configs/deploy_smoke_custom_cpu.json" in out
    assert "  - examples/configs/industrial_adapt_audited.json" in out
    assert "  - examples/configs/manifest_industrial_workflow_balanced.json" in out
    assert "Runtime profile: cpu-offline" in out
    assert "Expected artifacts:" in out
    assert "  - artifacts/infer_config.json" in out
    assert "  - deploy_bundle/bundle_manifest.json" in out
    assert "  - deploy_bundle/handoff_report.json" in out
    assert (
        "Run command: pyimgano train --config examples/configs/deploy_smoke_custom_cpu.json" in out
    )
    assert "Metadata:" in out
    assert (
        "  description: Adaptation-first industrial workbench recipe "
        "(tiling/postprocess/maps optional)."
    ) in out


def test_train_cli_recipe_info_outputs_text_starter_status_and_reason(capsys):
    from pyimgano.train_cli import main

    code = main(["--recipe-info", "anomalib-train"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Recipe: anomalib-train" in out
    assert "Starter status: manual-only" in out
    assert "Starter reason: Optional placeholder recipe." in out
    assert "Install hint: pip install 'pyimgano[anomalib]'" in out
    assert "Metadata:" in out
    assert "  requires_extra: anomalib" in out

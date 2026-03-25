import pytest


def test_root_cli_prints_help_with_command_index(capsys):
    from pyimgano.root_cli import main

    rc = main([])
    out = capsys.readouterr().out

    assert rc == 0
    assert "pyimgano <command> [args...]" in out
    assert "benchmark" in out
    assert "bundle" in out
    assert "evaluate" in out
    assert "infer" in out
    assert "train" in out
    assert "runs" in out
    assert "--list [KIND]" in out
    assert "benchmark --list-official-configs" in out
    assert "industrial_adapt_audited.json" in out
    assert "runs publication" in out
    assert "runs acceptance" in out
    assert "weights audit-bundle" in out


def test_root_cli_delegates_discovery_shortcuts(monkeypatch):
    import pyimgano.root_cli as root_cli

    calls = []

    monkeypatch.setattr(
        root_cli,
        "_run_discovery_cli",
        lambda argv: calls.append(list(argv)) or 17,
    )

    rc = root_cli.main(["--list", "models", "--json"])

    assert rc == 17
    assert calls == [["--list", "models", "--json"]]


def test_root_cli_accepts_list_alias_without_dashes(monkeypatch):
    import pyimgano.root_cli as root_cli

    calls = []

    monkeypatch.setattr(
        root_cli,
        "_run_discovery_cli",
        lambda argv: calls.append(list(argv)) or 19,
    )

    rc = root_cli.main(["list", "models", "--family", "patchcore"])

    assert rc == 19
    assert calls == [["--list", "models", "--family", "patchcore"]]


def test_root_cli_accepts_discovery_after_double_dash(monkeypatch):
    import pyimgano.root_cli as root_cli

    calls = []

    monkeypatch.setattr(
        root_cli,
        "_run_discovery_cli",
        lambda argv: calls.append(list(argv)) or 29,
    )

    rc = root_cli.main(["--", "list", "models", "--json"])

    assert rc == 29
    assert calls == [["--list", "models", "--json"]]


def test_root_cli_delegates_subcommand(monkeypatch):
    import pyimgano.root_cli as root_cli

    calls = []

    monkeypatch.setattr(
        root_cli,
        "_dispatch_command",
        lambda name, argv: calls.append((name, list(argv))) or 23,
    )

    rc = root_cli.main(["train", "--dry-run", "--config", "cfg.json"])

    assert rc == 23
    assert calls == [("train", ["--dry-run", "--config", "cfg.json"])]


def test_root_cli_delegates_bundle_subcommand(monkeypatch):
    import pyimgano.root_cli as root_cli

    calls = []

    monkeypatch.setattr(
        root_cli,
        "_dispatch_command",
        lambda name, argv: calls.append((name, list(argv))) or 31,
    )

    rc = root_cli.main(["bundle", "validate", "/tmp/deploy_bundle", "--json"])

    assert rc == 31
    assert calls == [("bundle", ["validate", "/tmp/deploy_bundle", "--json"])]


def test_root_cli_delegates_evaluate_subcommand(monkeypatch):
    import pyimgano.root_cli as root_cli

    calls = []

    monkeypatch.setattr(
        root_cli,
        "_dispatch_command",
        lambda name, argv: calls.append((name, list(argv))) or 37,
    )

    rc = root_cli.main(["evaluate", "--config", "eval.json", "--json"])

    assert rc == 37
    assert calls == [("evaluate", ["--config", "eval.json", "--json"])]


def test_root_cli_help_subcommand_routes_to_subcommand_help(monkeypatch):
    import pyimgano.root_cli as root_cli

    calls = []

    monkeypatch.setattr(
        root_cli,
        "_dispatch_command",
        lambda name, argv: calls.append((name, list(argv))) or 0,
    )

    rc = root_cli.main(["help", "train"])

    assert rc == 0
    assert calls == [("train", ["--help"])]


def test_root_cli_accepts_help_after_double_dash(monkeypatch):
    import pyimgano.root_cli as root_cli

    calls = []

    monkeypatch.setattr(
        root_cli,
        "_dispatch_command",
        lambda name, argv: calls.append((name, list(argv))) or 0,
    )

    rc = root_cli.main(["--", "help", "train"])

    assert rc == 0
    assert calls == [("train", ["--help"])]


def test_root_cli_rejects_unknown_command(capsys):
    from pyimgano.root_cli import main

    rc = main(["unknown-command"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Unknown command" in err


def test_root_cli_rejects_help_for_unknown_command(capsys):
    from pyimgano.root_cli import main

    rc = main(["help", "unknown-command"])
    err = capsys.readouterr().err

    assert rc == 2
    assert "Unknown command" in err


def test_root_cli_surfaces_system_exit_codes_from_subcommands(monkeypatch):
    import pyimgano.root_cli as root_cli

    def _raise_system_exit(_name, _argv):
        raise SystemExit(2)

    monkeypatch.setattr(root_cli, "_dispatch_command", _raise_system_exit)

    rc = root_cli.main(["train", "--help"])

    assert rc == 2

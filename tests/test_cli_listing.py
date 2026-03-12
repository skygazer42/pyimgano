from __future__ import annotations


def test_emit_listing_outputs_text_lines(capsys) -> None:
    from pyimgano.cli_listing import emit_listing

    rc = emit_listing(["alpha", "beta"], json_output=False)
    assert rc == 0
    assert capsys.readouterr().out.strip().splitlines() == ["alpha", "beta"]


def test_emit_listing_delegates_json_output(monkeypatch) -> None:
    import pyimgano.cli_listing as cli_listing

    calls = []
    monkeypatch.setattr(
        cli_listing,
        "cli_output",
        type(
            "_StubCliOutput",
            (),
            {
                "emit_json": staticmethod(
                    lambda payload, **kwargs: calls.append((payload, kwargs)) or 17
                )
            },
        ),
        raising=False,
    )

    rc = cli_listing.emit_listing(["alpha", "beta"], json_output=True, sort_keys=False)
    assert rc == 17
    assert calls == [(["alpha", "beta"], {"status_code": 0, "sort_keys": False})]


def test_emit_listing_can_use_distinct_json_payload(monkeypatch) -> None:
    import pyimgano.cli_listing as cli_listing

    calls = []
    monkeypatch.setattr(
        cli_listing,
        "cli_output",
        type(
            "_StubCliOutput",
            (),
            {
                "emit_json": staticmethod(
                    lambda payload, **kwargs: calls.append((payload, kwargs)) or 23
                )
            },
        ),
        raising=False,
    )

    rc = cli_listing.emit_listing(
        ["preset-a", "preset-b"],
        json_output=True,
        json_payload=[{"name": "preset-a", "tags": ["graph"]}],
        sort_keys=False,
    )
    assert rc == 23
    assert calls == [
        ([{"name": "preset-a", "tags": ["graph"]}], {"status_code": 0, "sort_keys": False})
    ]

from __future__ import annotations

from types import SimpleNamespace


def test_run_pyim_command_delegates_list_flow_through_shared_helpers(monkeypatch) -> None:
    import pyimgano.pyim_app as pyim_app

    helper_calls = []
    request_calls = []

    def _forbidden_pyim_list_request(**_kwargs):  # noqa: ANN003, ANN202 - test stub
        raise AssertionError("pyim_contracts should not be used")

    class _Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Options:
        list_kind = "models"
        tags = ["normalized-tag"]
        family = "normalized-family"
        algorithm_type = "normalized-type"
        year = "2032"
        deployable_only = True
        objective = "latency"
        selection_profile = "cpu-screening"
        topk = 3

        def to_request(self):
            request_calls.append(
                {
                    "list_kind": self.list_kind,
                    "tags": self.tags,
                    "family": self.family,
                    "algorithm_type": self.algorithm_type,
                    "year": self.year,
                    "deployable_only": self.deployable_only,
                    "objective": self.objective,
                    "selection_profile": self.selection_profile,
                    "topk": self.topk,
                }
            )
            return _Request(**request_calls[-1])

    monkeypatch.setattr(
        pyim_app,
        "pyim_cli_options",
        type(
            "_StubPyimCliOptions",
            (),
            {
                "resolve_pyim_list_options": staticmethod(
                    lambda **kwargs: helper_calls.append(dict(kwargs)) or _Options()
                )
            },
        ),
        raising=False,
    )

    monkeypatch.setattr(
        pyim_app,
        "pyim_contracts",
        type(
            "_ForbiddenPyimContracts",
            (),
            {"PyimListRequest": staticmethod(_forbidden_pyim_list_request)},
        ),
        raising=False,
    )

    service_calls = []
    monkeypatch.setattr(
        pyim_app,
        "pyim_service",
        type(
            "_StubPyimService",
            (),
            {
                "collect_pyim_listing_payload": staticmethod(
                    lambda request: service_calls.append(dict(request.__dict__))
                    or {"payload": "delegated"}
                ),
                "collect_pyim_model_selection_payload": staticmethod(
                    lambda request: {
                        "selection_context": {
                            "objective": request.objective,
                            "selection_profile": request.selection_profile,
                            "topk": request.topk,
                        },
                        "starter_picks": [{"name": "vision_ecod"}],
                    }
                ),
            },
        ),
        raising=False,
    )

    render_calls = []
    monkeypatch.setattr(
        pyim_app,
        "pyim_cli_rendering",
        type(
            "_StubPyimCliRendering",
            (),
            {
                "emit_pyim_list_payload": staticmethod(
                    lambda payload, **kwargs: render_calls.append((payload, kwargs)) or 91
                )
            },
        ),
        raising=False,
    )

    rc = pyim_app.run_pyim_command(
        pyim_app.PyimCommand(
            list_kind="models",
            tags=None,
            family=None,
            algorithm_type=None,
            year=None,
            deployable_only=False,
            objective="latency",
            selection_profile="cpu-screening",
            topk=3,
            audit_metadata=False,
            json_output=True,
        )
    )

    assert rc == 91
    assert helper_calls == [
        {
            "list_kind": "models",
            "tags": None,
            "family": None,
            "algorithm_type": None,
            "year": None,
            "deployable_only": False,
            "objective": "latency",
            "selection_profile": "cpu-screening",
            "topk": 3,
        }
    ]
    assert request_calls == [
        {
            "list_kind": "models",
            "tags": ["normalized-tag"],
            "family": "normalized-family",
            "algorithm_type": "normalized-type",
            "year": "2032",
            "deployable_only": True,
            "objective": "latency",
            "selection_profile": "cpu-screening",
            "topk": 3,
        }
    ]
    assert service_calls == request_calls
    assert render_calls == [
        (
            {"payload": "delegated"},
            {
                "list_kind": "models",
                "json_output": True,
                "selection_payload": {
                    "selection_context": {
                        "objective": "latency",
                        "selection_profile": "cpu-screening",
                        "topk": 3,
                    },
                    "starter_picks": [{"name": "vision_ecod"}],
                },
            },
        )
    ]


def test_run_pyim_command_audit_metadata_json_preserves_nonzero_exit(monkeypatch) -> None:
    import pyimgano.pyim_app as pyim_app

    payload = {
        "summary": {
            "models_with_required_issues": 1,
            "models_with_recommended_issues": 0,
            "models_with_invalid_fields": 0,
        }
    }

    service_calls = []
    monkeypatch.setattr(
        pyim_app,
        "pyim_audit_service",
        type(
            "_StubPyimAuditService",
            (),
            {
                "collect_pyim_audit_payload": staticmethod(
                    lambda: service_calls.append(True) or payload
                )
            },
        ),
        raising=False,
    )

    calls = []
    monkeypatch.setattr(
        pyim_app,
        "pyim_audit_rendering",
        type(
            "_StubPyimAuditRendering",
            (),
            {
                "emit_pyim_audit_payload": staticmethod(
                    lambda audit_payload, **kwargs: calls.append((audit_payload, kwargs)) or 1
                )
            },
        ),
        raising=False,
    )

    rc = pyim_app.run_pyim_command(
        pyim_app.PyimCommand(
            audit_metadata=True,
            json_output=True,
        )
    )

    assert rc == 1
    assert service_calls == [True]
    assert calls == [(payload, {"json_output": True})]

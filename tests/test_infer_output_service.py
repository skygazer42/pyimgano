from __future__ import annotations

import io
import json

import pytest

from pyimgano.services import infer_output_service


class _TrackedStringIO(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1
        super().flush()


def test_open_infer_output_targets_requires_defects_for_regions_jsonl(tmp_path) -> None:
    with pytest.raises(ValueError, match="--defects-regions-jsonl requires --defects"):
        infer_output_service.open_infer_output_targets(
            infer_output_service.InferOutputTargetsRequest(
                defects_enabled=False,
                defects_regions_jsonl=str(tmp_path / "regions.jsonl"),
            )
        )


def test_open_infer_output_targets_creates_parent_dirs(tmp_path) -> None:
    targets = infer_output_service.open_infer_output_targets(
        infer_output_service.InferOutputTargetsRequest(
            save_jsonl=str(tmp_path / "nested" / "out.jsonl"),
            defects_enabled=True,
            defects_regions_jsonl=str(tmp_path / "nested" / "regions.jsonl"),
        )
    )
    try:
        assert targets.output_file is not None
        assert targets.regions_file is not None
        assert (tmp_path / "nested").exists()
    finally:
        if targets.output_file is not None:
            targets.output_file.close()
        if targets.regions_file is not None:
            targets.regions_file.close()


def test_write_infer_output_payloads_writes_record_and_regions_and_flushes() -> None:
    output_file = _TrackedStringIO()
    regions_file = _TrackedStringIO()

    result = infer_output_service.write_infer_output_payloads(
        infer_output_service.InferOutputWriteRequest(
            record={"b": 1, "a": 2},
            regions_payload={"z": 9, "y": 8},
            output_file=output_file,
            regions_file=regions_file,
            flush_every=1,
            output_written=0,
            regions_written=0,
        )
    )

    assert json.loads(output_file.getvalue().strip()) == {"a": 2, "b": 1}
    assert json.loads(regions_file.getvalue().strip()) == {"y": 8, "z": 9}
    assert result.output_written == 1
    assert result.regions_written == 1
    assert output_file.flush_calls == 1
    assert regions_file.flush_calls == 1


def test_write_infer_output_payloads_prints_record_when_no_output_file() -> None:
    lines: list[str] = []

    result = infer_output_service.write_infer_output_payloads(
        infer_output_service.InferOutputWriteRequest(
            record={"b": 1, "a": 2},
            output_written=3,
            regions_written=4,
        ),
        print_fn=lines.append,
    )

    assert lines == ['{"a": 2, "b": 1}']
    assert result.output_written == 3
    assert result.regions_written == 4


def test_build_infer_error_record_captures_stage_and_exception() -> None:
    record = infer_output_service.build_infer_error_record(
        infer_output_service.InferErrorRecordRequest(
            index=7,
            input_path="sample.png",
            exc=ValueError("boom"),
            stage="infer",
        )
    )

    assert record == {
        "status": "error",
        "index": 7,
        "input": "sample.png",
        "error": {
            "type": "ValueError",
            "message": "boom",
            "stage": "infer",
        },
    }

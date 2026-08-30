from __future__ import annotations

"""Model serialization helpers (joblib).

Many classical `pyimgano` models are Python objects that can be serialized with
joblib/pickle. Deep models often require special checkpoint handling; those are
out of scope for this helper.
"""

from pathlib import Path
from typing import Any


class UntrustedModelArtifactError(ValueError):
    """Raised when executable serialization is loaded without explicit trust."""


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def save_model(model: Any, path: str | Path) -> Path:
    """Serialize a model via joblib and write an integrity sidecar.

    Joblib is an executable pickle-based format. The digest detects accidental
    corruption or replacement, but it does not make the artifact trustworthy.
    """

    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(p))
    digest = _sha256_file(p)
    _digest_path(p).write_text(f"{digest}  {p.name}\n", encoding="ascii")
    return p


def load_model(
    path: str | Path,
    *,
    trusted: bool = False,
    expected_sha256: str | None = None,
) -> Any:
    """Load a trusted model serialized via :func:`save_model`.

    ``trusted=True`` is mandatory because joblib can execute arbitrary code.
    When an explicit digest or a ``.sha256`` sidecar exists, integrity is
    checked before deserialization. A digest proves integrity, not provenance.
    """

    if not trusted:
        raise UntrustedModelArtifactError(
            "Refusing to load executable joblib/pickle without trusted=True. "
            "Only load model artifacts from a verified source."
        )

    import joblib

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model artifact not found: {p}")

    expected = str(expected_sha256).strip().lower() if expected_sha256 is not None else None
    sidecar = _digest_path(p)
    if expected is None and sidecar.is_file():
        fields = sidecar.read_text(encoding="ascii").strip().split()
        if not fields:
            raise ValueError(f"Invalid SHA-256 sidecar: {sidecar}")
        expected = fields[0].lower()
    if expected is not None:
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest.")
        actual = _sha256_file(p)
        if actual != expected:
            raise ValueError(
                "Model artifact SHA-256 mismatch; refusing deserialization. "
                f"expected={expected}, actual={actual}"
            )

    return joblib.load(str(p))  # nosec B301 - explicit trusted=True gate above.


__all__ = ["UntrustedModelArtifactError", "load_model", "save_model"]

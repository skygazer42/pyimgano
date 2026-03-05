import pytest


def test_resolve_n_jobs_defaults_to_one() -> None:
    from pyimgano.utils.parallel import resolve_n_jobs

    assert resolve_n_jobs(None) == 1


def test_resolve_n_jobs_rejects_zero() -> None:
    from pyimgano.utils.parallel import resolve_n_jobs

    with pytest.raises(ValueError, match="n_jobs must be != 0"):
        resolve_n_jobs(0)


def test_resolve_n_jobs_handles_negative_values() -> None:
    from pyimgano.utils.parallel import resolve_n_jobs

    assert resolve_n_jobs(-1) >= 1
    assert resolve_n_jobs(-2) >= 1


def test_parallel_map_runs_and_preserves_order() -> None:
    from pyimgano.utils.parallel import parallel_map

    out = parallel_map(lambda x: x + 1, [1, 2, 3, 4], n_jobs=2, backend="threading")
    assert out == [2, 3, 4, 5]

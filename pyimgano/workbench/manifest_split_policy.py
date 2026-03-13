from __future__ import annotations

from pyimgano.workbench.config import WorkbenchConfig


def build_manifest_split_policy(*, config: WorkbenchConfig):
    from pyimgano.datasets.manifest import ManifestSplitPolicy

    sp = config.dataset.split_policy
    seed = (
        int(sp.seed)
        if sp.seed is not None
        else (int(config.seed) if config.seed is not None else 0)
    )
    return ManifestSplitPolicy(
        mode=str(sp.mode),
        scope=str(sp.scope),
        seed=seed,
        test_normal_fraction=float(sp.test_normal_fraction),
    )


__all__ = ["build_manifest_split_policy"]

"""Unregistered One-to-Normal compatibility marker.

The former ``vision_one_to_normal`` entry was only an injectable pixel-residual
normalizer and did not implement the NeurIPS 2024 method.  The author artifact
at the pinned ModelScope commit is incomplete (its evaluation script imports a
missing ``model.py`` and references unpublished projection/DreamBooth weights),
so pyimgano deliberately does not invent the missing runtime.
"""

PAPER_FIDELITY = "inspired"
IMPLEMENTATION_STATUS = "unregistered-incomplete-author-release"
RELATED_PAPER = "One-to-Normal: Anomaly Personalization for Few-shot Anomaly Detection"
RELATED_PAPER_URL = "https://arxiv.org/abs/2502.01201"
AUTHOR_ARTIFACT = "https://www.modelscope.cn/models/liyiyue/One-to-Normal9"
AUTHOR_ARTIFACT_COMMIT = "1faca331bf876a66f105a8f5aa095e399c21e44d"

PAPER_DIFFUSION_BACKBONE = "Stable Diffusion v1.5 + DreamBooth"
PAPER_CLIP_BACKBONE = "ViT-L/14"
PAPER_IMAGE_SIZE = 240
PAPER_SHOTS = (2, 4, 8)
PAPER_TIMESTEP_RATIO = 0.3
PAPER_MEMORY_SIZE = 30
PAPER_ALPHA = 1.0
PAPER_BETA = 0.5

__all__ = [
    "AUTHOR_ARTIFACT",
    "AUTHOR_ARTIFACT_COMMIT",
    "IMPLEMENTATION_STATUS",
    "PAPER_ALPHA",
    "PAPER_BETA",
    "PAPER_CLIP_BACKBONE",
    "PAPER_DIFFUSION_BACKBONE",
    "PAPER_FIDELITY",
    "PAPER_IMAGE_SIZE",
    "PAPER_MEMORY_SIZE",
    "PAPER_SHOTS",
    "PAPER_TIMESTEP_RATIO",
    "RELATED_PAPER",
    "RELATED_PAPER_URL",
]

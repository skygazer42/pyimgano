"""Unregistered CrossMAD compatibility marker.

The former ``core_crossmad`` and ``vision_crossmad`` entries were a KMeans
distance baseline over independent feature vectors.  CrossMAD instead learns
normal/abnormal global and local visual prototypes per source dataset, then
harmonizes prototypes from other modalities and optionally compares against
normal support images.  The contracts and network are unrelated, so the proxy
is not exposed through the model registry.

The pinned author release consumes precomputed CLIP feature ``.pt`` files but
does not include their extraction pipeline, trained checkpoints, or the
aggregated prototype file.  A faithful adapter therefore cannot be assembled
from that release without inventing missing model state and preprocessing.
"""

PAPER_FIDELITY = "not-applicable"
IMPLEMENTATION_STATUS = "unregistered-incompatible-cross-modal-contract"
RELATED_PAPER = (
    "Beyond Single-Modal Boundary: Cross-Modal Anomaly Detection through "
    "Visual Prototype and Harmonization"
)
RELATED_PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2025/html/"
    "Mao_Beyond_Single-Modal_Boundary_Cross-Modal_Anomaly_Detection_through_"
    "Visual_Prototype_and_CVPR_2025_paper.html"
)
AUTHOR_REPOSITORY = "https://github.com/Kerio99/CMAD"
AUTHOR_REPOSITORY_COMMIT = "c2b8fe7e060a642a247beaec24f1582c54d02cdf"

PAPER_INPUT_CONTRACT = "labeled source-modality datasets plus an unseen-modality query"
PAPER_MODULES = (
    "Transferable Visual Prototype",
    "Prototype Harmonization",
    "Visual Discrepancy Inference",
)
PAPER_IMAGE_ENCODER = "CLIP image encoder"
PAPER_LOCAL_FEATURE_LAYERS = (5, 15, 25)
PAPER_IMAGE_SIZE = 224
PAPER_SUPPORT_SHOTS = (1, 2, 4, 8)
PAPER_BATCH_SIZE = 16
PAPER_LEARNING_RATE = 1e-3
PAPER_TRAINING_ITERATIONS = 1500

AUTHOR_CODE_GLOBAL_PROTOTYPE_SHAPE = (2, 1024)
AUTHOR_CODE_LOCAL_PROTOTYPE_SHAPE = (2, 1280)
AUTHOR_CODE_ANOMALY_TEMPERATURE = 0.07
AUTHOR_CODE_HARMONIZATION_TEMPERATURE = 0.04
AUTHOR_CODE_TOPK = 50
AUTHOR_CODE_ZERO_SHOT_FUSION = {"global": 0.8, "local": 0.2}
AUTHOR_CODE_FEW_SHOT_FUSION = {"zero_shot": 0.5, "discrepancy": 0.5}
AUTHOR_CODE_ADAM_BETAS = (0.5, 0.999)
AUTHOR_CODE_RANDOM_SEED = 10

AUTHOR_RELEASE_HAS_FEATURE_EXTRACTION_PIPELINE = False
AUTHOR_RELEASE_HAS_PROTOTYPE_LIST = False
AUTHOR_RELEASE_HAS_CHECKPOINTS = False
AUTHOR_RELEASE_HAS_LICENSE_FILE = False

"""Unregistered Odd-One-Out compatibility marker.

The former ``core_oddoneout`` and vision wrappers implemented a relative kNN
distance ratio over independent feature vectors.  That is unrelated to the
CVPR 2025 method, whose input is a posed multi-view scene and whose output is
an anomaly label and 3D box for every object instance.  The official release
is complete, but that scene contract does not fit pyimgano's current
single-sample detector registry, so no substitute runtime is invented here.
"""

PAPER_FIDELITY = "not-applicable"
IMPLEMENTATION_STATUS = "unregistered-incompatible-scene-contract"
RELATED_PAPER = "Odd-One-Out: Anomaly Detection by Comparing with Neighbors"
RELATED_PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2025/html/"
    "Bhunia_Odd-One-Out_Anomaly_Detection_by_Comparing_with_Neighbors_CVPR_2025_paper.html"
)
AUTHOR_REPOSITORY = "https://github.com/VICO-UoE/OddOneOutAD"
AUTHOR_REPOSITORY_COMMIT = "5200c918e80628288c4bdc46c5afd036d1e79482"

PAPER_INPUT_CONTRACT = "five posed RGB views plus camera projection matrices"
PAPER_OUTPUT_CONTRACT = "per-instance anomaly labels and 3D bounding boxes"
PAPER_2D_BACKBONE = "ResNet50-FPN"
PAPER_3D_BACKBONE = "four-scale encoder-decoder 3D CNN"
PAPER_DINO_FEATURE_DIM = 128
PAPER_IMAGE_SIZE = 256
PAPER_NUM_VIEWS = 5
PAPER_TRAIN_VIEWS = 10
PAPER_VOXEL_GRID = (96, 96, 16)
PAPER_VOXEL_SIZE_METERS = 0.04
PAPER_POINTS_PER_RAY = 128
PAPER_RENDERED_FEATURE_SIZE = 32
PAPER_DENSITY_THRESHOLD = 0.2
PAPER_ATTENTION_BLOCKS = 3
PAPER_ATTENTION_HEADS = 8
PAPER_ATTENTION_TOPK = 20
PAPER_STAGE_EPOCHS = (50, 50)
PAPER_BATCH_SIZE = 4
PAPER_LEARNING_RATE = 2e-5

AUTHOR_CODE_DINO_MODEL = "dinov2_vits14"
AUTHOR_CODE_OBJECT_VOLUME = (8, 8, 8)
AUTHOR_CODE_VOXEL_DIMS = {"feature": 32, "hidden": 128, "projection": 384}
AUTHOR_CODE_LEARNING_RATES = {"backbone": 1e-5, "matching_net": 2e-5}
AUTHOR_CODE_ADAM_BETAS = (0.0, 0.999)
AUTHOR_CODE_THRESHOLDS = {"toysad8k": 0.5, "partsad15k": 0.3}

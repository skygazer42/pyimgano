"""Unregistered AnoGen workflow marker.

The former ``vision_anogen_adapter`` was a generic generator plus mean-image
residual scorer; neither network nor score exists in the ECCV 2024 method.
AnoGen is a three-stage data-generation/training workflow, not a standalone
detector, so pyimgano does not expose that proxy through the model registry.

Paper values and author-code defaults are kept separate because the pinned
release uses a BERT/1280 LDM configuration while the paper specifies a
CLIP/768 text embedding.
"""

PAPER_FIDELITY = "not-applicable"
IMPLEMENTATION_STATUS = "unregistered-workflow-not-detector"
RELATED_PAPER = "Few-Shot Anomaly-Driven Generation for Anomaly Classification and Segmentation"
RELATED_PAPER_URL = "https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11002.pdf"
AUTHOR_REPOSITORY = "https://github.com/gaobb/AnoGen"
AUTHOR_REPOSITORY_COMMIT = "11ade1bd89ec3bb89646d70b6b95f2c69053f973"

PAPER_PIPELINE = (
    "learn a mask-guided anomaly embedding with a frozen LDM",
    "generate box-guided anomalies by latent inpainting",
    "train weakly supervised DRAEM or DeSTSeg",
)
PAPER_TEXT_ENCODER = "CLIP"
PAPER_EMBEDDING_DIM = 768
PAPER_SUPPORT_SHOTS = 3
PAPER_EMBEDDING_TRAIN_STEPS = 6000
PAPER_EMBEDDING_LEARNING_RATE = 0.005
PAPER_MASKS_PER_NORMAL_IMAGE = 2
PAPER_IMAGES_PER_MASK = 2
PAPER_NORMAL_CONFIDENCE_THRESHOLD = 0.9
PAPER_GENERATED_SAMPLE_PROBABILITY = 0.5
PAPER_GENERATED_DATASET_SIZE = 70_760

AUTHOR_CODE_IMAGE_SIZE = 256
AUTHOR_CODE_TEXT_ENCODER = "BERTEmbedder"
AUTHOR_CODE_CONTEXT_DIM = 1280
AUTHOR_CODE_MAX_STEPS = 6100
AUTHOR_CODE_UNET_MODEL_CHANNELS = 320
AUTHOR_CODE_UNET_CHANNEL_MULTIPLIERS = (1, 2, 4, 4)
AUTHOR_CODE_UNET_RESIDUAL_BLOCKS = 2
AUTHOR_CODE_UNET_ATTENTION_HEADS = 8
AUTHOR_CODE_DDIM_STEPS = 50
AUTHOR_CODE_GUIDANCE_SCALE = 10.0
AUTHOR_CODE_DDIM_ETA = 0.0
AUTHOR_RELEASE_HAS_ANOMALY_EMBEDDINGS = True
AUTHOR_RELEASE_HAS_DETECTOR_CHECKPOINT = False

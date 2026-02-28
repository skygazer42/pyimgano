from __future__ import annotations

# Import builtin recipes for side effects (registration).
from .industrial_adapt import industrial_adapt  # noqa: F401
from .industrial_adapt_fp40 import industrial_adapt_fp40  # noqa: F401
from .industrial_adapt_highres import industrial_adapt_highres  # noqa: F401
from .industrial_embedding_core_fast import industrial_embedding_core_fast  # noqa: F401
from .anomalib_train import anomalib_train  # noqa: F401
from .micro_finetune_autoencoder import micro_finetune_autoencoder  # noqa: F401
from .classical_recipes import (  # noqa: F401
    classical_colorhist_mahalanobis,
    classical_hog_ecod,
    classical_lbp_loop,
    classical_struct_iforest_synth,
)

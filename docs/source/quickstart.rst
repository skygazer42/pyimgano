Quickstart
==========

This page is a **short Sphinx-friendly quickstart**.

For the full Markdown guide (CLI + industrial workflow), see:

- https://github.com/skygazer42/pyimgano/blob/main/docs/QUICKSTART.md

Install
-------

.. code-block:: bash

   pip install pyimgano

Minimal Python usage (feature vectors)
--------------------------------------

.. code-block:: python

   import numpy as np
   from pyimgano.models import create_model

   class IdentityExtractor:
       def extract(self, X):
           return np.asarray(X)

   detector = create_model(
       "vision_ecod",
       feature_extractor=IdentityExtractor(),
       contamination=0.1,
   )

   X_train = np.random.randn(256, 32)
   X_test = np.random.randn(16, 32)
   detector.fit(X_train)
   scores = detector.decision_function(X_test)

CLI (industrial JSONL)
----------------------

.. code-block:: bash

   pyimgano-infer \
     --model vision_patchcore \
     --train-dir /path/to/train/good \
     --input /path/to/images_or_dir \
     --save-jsonl out.jsonl

See also:

- :doc:`api/detectors` (registry API docs)
- https://github.com/skygazer42/pyimgano/blob/main/docs/CLI_REFERENCE.md
- https://github.com/skygazer42/pyimgano/blob/main/docs/WORKBENCH.md

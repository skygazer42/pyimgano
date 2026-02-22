Models / Detectors API Reference
================================

PyImgAno exposes anomaly detection algorithms through a **registry-driven**
factory in ``pyimgano.models``.

Most users should prefer:

- ``pyimgano.models.list_models()`` to discover algorithms
- ``pyimgano.models.create_model()`` to construct an algorithm by name

This keeps imports stable while letting the project grow its model zoo.

Registry (Factory API)
----------------------

.. automodule:: pyimgano.models.registry
   :members:
   :undoc-members:
   :show-inheritance:

Base Classes
------------

.. autoclass:: pyimgano.models.baseml.BaseVisionDetector
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyimgano.models.baseCv.BaseVisionDeepDetector
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

List available models
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.models import list_models

   print(list_models()[:20])
   print(list_models(tags=["classical"])[:20])
   print(list_models(tags=["pixel_map"])[:20])

Create and run a classical detector (feature vectors)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pyimgano.models import create_model

   class IdentityExtractor:
       def extract(self, X):
           return np.asarray(X)

   X_train = np.random.randn(1000, 64)
   X_test = np.random.randn(100, 64)

   det = create_model(
       "vision_iforest",
       feature_extractor=IdentityExtractor(),
       contamination=0.1,
       n_estimators=200,
   )
   det.fit(X_train)
   scores = det.decision_function(X_test)
   labels = det.predict(X_test)

Create and run an industrial pixel-map model (images)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.models import create_model

   det = create_model("vision_patchcore", device="cuda", pretrained=True)
   det.fit(train_paths)  # list[str] image paths
   scores = det.decision_function(test_paths)

Notes
-----

- Use ``pyimgano-benchmark --list-models`` to discover model names from the CLI.
- Optional backends are loaded lazily and raise install hints when constructed.

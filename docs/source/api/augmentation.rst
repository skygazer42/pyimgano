Augmentation API Reference
==========================

This page provides detailed API documentation for data augmentation operations in PyImgAno.

Pipeline Classes
----------------

Compose
~~~~~~~

.. autoclass:: pyimgano.preprocessing.Compose
   :members:
   :undoc-members:
   :show-inheritance:

OneOf
~~~~~

.. autoclass:: pyimgano.preprocessing.OneOf
   :members:
   :undoc-members:
   :show-inheritance:

RandomApply
~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomApply
   :members:
   :undoc-members:
   :show-inheritance:

AugmentationPipeline
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.AugmentationPipeline
   :members:
   :undoc-members:
   :show-inheritance:

Geometric Transforms
--------------------

RandomRotate
~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomRotate
   :members:
   :undoc-members:
   :show-inheritance:

RandomFlip
~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomFlip
   :members:
   :undoc-members:
   :show-inheritance:

RandomScale
~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomScale
   :members:
   :undoc-members:
   :show-inheritance:

RandomTranslate
~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomTranslate
   :members:
   :undoc-members:
   :show-inheritance:

RandomShear
~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomShear
   :members:
   :undoc-members:
   :show-inheritance:

RandomPerspective
~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomPerspective
   :members:
   :undoc-members:
   :show-inheritance:

ElasticTransform
~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.ElasticTransform
   :members:
   :undoc-members:
   :show-inheritance:

Color Transforms
----------------

ColorJitter
~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.ColorJitter
   :members:
   :undoc-members:
   :show-inheritance:

Noise Transforms
----------------

GaussianNoise
~~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.GaussianNoise
   :members:
   :undoc-members:
   :show-inheritance:

SaltPepperNoise
~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.SaltPepperNoise
   :members:
   :undoc-members:
   :show-inheritance:

Blur Transforms
---------------

MotionBlur
~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.MotionBlur
   :members:
   :undoc-members:
   :show-inheritance:

DefocusBlur
~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.DefocusBlur
   :members:
   :undoc-members:
   :show-inheritance:

Weather Effects
---------------

RandomRain
~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomRain
   :members:
   :undoc-members:
   :show-inheritance:

RandomFog
~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomFog
   :members:
   :undoc-members:
   :show-inheritance:

RandomSnow
~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomSnow
   :members:
   :undoc-members:
   :show-inheritance:

RandomShadow
~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomShadow
   :members:
   :undoc-members:
   :show-inheritance:

Occlusion Transforms
--------------------

RandomCutout
~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.RandomCutout
   :members:
   :undoc-members:
   :show-inheritance:

GridMask
~~~~~~~~

.. autoclass:: pyimgano.preprocessing.GridMask
   :members:
   :undoc-members:
   :show-inheritance:

Preset Pipelines
----------------

.. autofunction:: pyimgano.preprocessing.get_light_augmentation

.. autofunction:: pyimgano.preprocessing.get_medium_augmentation

.. autofunction:: pyimgano.preprocessing.get_heavy_augmentation

.. autofunction:: pyimgano.preprocessing.get_weather_augmentation

.. autofunction:: pyimgano.preprocessing.get_anomaly_augmentation

Examples
--------

Simple Augmentation
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import RandomRotate, RandomFlip
   import cv2

   # Load image
   image = cv2.imread('sample.jpg')

   # Single augmentations
   rotate = RandomRotate(angle_range=(-30, 30), p=1.0)
   flip = RandomFlip(mode='horizontal', p=1.0)

   # Apply
   rotated = rotate(image)
   flipped = flip(image)

Augmentation Pipeline
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import (
       Compose, OneOf, RandomApply,
       RandomRotate, RandomFlip, ColorJitter, GaussianNoise
   )

   # Create complex pipeline
   augmentation = Compose([
       RandomFlip(mode='horizontal', p=0.5),
       RandomRotate(angle_range=(-15, 15), p=0.5),
       OneOf([
           GaussianNoise(std=0.01, p=1.0),
           ColorJitter(brightness=0.2, p=1.0),
       ], p=0.3),
       RandomApply([
           RandomRotate(angle_range=(-5, 5), p=1.0)
       ], p=0.2),
   ])

   # Apply to training data
   augmented_images = [augmentation(img) for img in training_images]

Preset Pipelines
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import (
       get_light_augmentation,
       get_medium_augmentation,
       get_heavy_augmentation,
   )

   # Light augmentation (minimal changes)
   light_aug = get_light_augmentation()
   augmented = light_aug(image)

   # Medium augmentation (balanced)
   medium_aug = get_medium_augmentation()
   augmented = medium_aug(image)

   # Heavy augmentation (aggressive)
   heavy_aug = get_heavy_augmentation()
   augmented = heavy_aug(image)

Pipeline with Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import (
       AugmentationPipeline,
       get_medium_augmentation
   )

   # Create pipeline with tracking
   pipeline = AugmentationPipeline(
       get_medium_augmentation(),
       track_stats=True
   )

   # Apply to batch
   augmented_batch = [pipeline(img) for img in training_images]

   # Get statistics
   stats = pipeline.get_stats()
   print(f"Total applications: {stats['total_applications']}")
   print(f"Average time: {stats['avg_time_ms']:.2f}ms")
   print(f"Operation counts: {stats['operation_counts']}")

   # Reset statistics
   pipeline.reset_stats()

Training with Augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import (
       get_medium_augmentation,
       AdvancedImageEnhancer
   )
   from pyimgano.models import create_model
   import numpy as np
   import cv2

   # Setup
   augmentation = get_medium_augmentation()
   enhancer = AdvancedImageEnhancer()

   def preprocess(img):
       """Preprocess with feature extraction."""
       lbp = enhancer.compute_lbp(img)
       return lbp.flatten()

   # Load normal training images
   normal_images = [cv2.imread(f'normal_{i}.jpg') for i in range(100)]

   # Create augmented dataset
   augmented_images = []
   for img in normal_images:
       # Original
       augmented_images.append(img)
       # 3 augmented versions
       for _ in range(3):
           augmented_images.append(augmentation(img))

   print(f"Dataset size: {len(normal_images)} -> {len(augmented_images)}")

   # Preprocess all
   X_train = np.array([preprocess(img) for img in augmented_images])

   # Train detector
   class IdentityExtractor:
       def extract(self, X):
           return np.asarray(X)

   detector = create_model(
       "vision_deep_svdd",
       feature_extractor=IdentityExtractor(),
       contamination=0.1,
       lr=1e-3,
       n_features=X_train.shape[1],
       hidden_neurons=[256, 64, 256],
       use_autoencoder=True,
       epochs=50,
       batch_size=32,
       verbose=0,
   )
   detector.fit(X_train)

   # Test
   test_image = cv2.imread('test.jpg')
   test_features = preprocess(test_image)
   score = detector.predict_proba([test_features])[0]

   print(f"Anomaly score: {score:.4f}")

Weather Effects
~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import (
       RandomRain, RandomFog, RandomSnow, RandomShadow
   )

   # Weather augmentations
   rain = RandomRain(rain_type='drizzle', p=1.0)
   fog = RandomFog(fog_coef=0.3, p=1.0)
   snow = RandomSnow(snow_coef=0.3, p=1.0)
   shadow = RandomShadow(shadow_roi=(0.3, 0.7, 0.3, 0.7), p=1.0)

   # Apply
   rainy = rain(image)
   foggy = fog(image)
   snowy = snow(image)
   shadowy = shadow(image)

   # Or use preset
   from pyimgano.preprocessing import get_weather_augmentation

   weather_aug = get_weather_augmentation()
   augmented = weather_aug(image)

Anomaly Simulation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import get_anomaly_augmentation

   # Anomaly-specific augmentation
   # Simulates defects, artifacts, and irregularities
   anomaly_aug = get_anomaly_augmentation()

   # Apply to normal images to create synthetic anomalies
   normal_image = cv2.imread('normal.jpg')
   synthetic_anomaly = anomaly_aug(normal_image)

   # Use for testing or semi-supervised learning
   test_images = [synthetic_anomaly for _ in range(50)]

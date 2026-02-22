Preprocessing API Reference
===========================

This page provides detailed API documentation for image preprocessing operations in PyImgAno.

Main Classes
------------

ImageEnhancer
~~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.ImageEnhancer
   :members:
   :undoc-members:
   :show-inheritance:

AdvancedImageEnhancer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.AdvancedImageEnhancer
   :members:
   :undoc-members:
   :show-inheritance:

PreprocessingPipeline
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyimgano.preprocessing.PreprocessingPipeline
   :members:
   :undoc-members:
   :show-inheritance:

Basic Operations
----------------

Edge Detection
~~~~~~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.edge_detection

Supported methods:

* ``canny``: Canny edge detector
* ``sobel``: Sobel operator
* ``laplacian``: Laplacian operator
* ``scharr``: Scharr operator
* ``prewitt``: Prewitt operator

Morphological Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.morphological_operation

Supported operations:

* ``erode``: Erosion
* ``dilate``: Dilation
* ``open``: Opening (erosion then dilation)
* ``close``: Closing (dilation then erosion)
* ``gradient``: Morphological gradient
* ``tophat``: Top hat transform
* ``blackhat``: Black hat transform

Filtering
~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.apply_filter

Supported filters:

* ``gaussian``: Gaussian blur
* ``bilateral``: Bilateral filter (edge-preserving)
* ``median``: Median filter
* ``box``: Box filter (average)
* ``nlm``: Non-local means filter

Normalization
~~~~~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.normalize_image

Supported methods:

* ``minmax``: Min-max scaling to [0, 1]
* ``zscore``: Z-score normalization
* ``robust``: Robust scaling using median and IQR

Advanced Operations
-------------------

Frequency Domain
~~~~~~~~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.apply_fft

.. autofunction:: pyimgano.preprocessing.apply_ifft

.. autofunction:: pyimgano.preprocessing.frequency_filter

Texture Analysis
~~~~~~~~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.apply_gabor_filter

.. autofunction:: pyimgano.preprocessing.compute_lbp

.. autofunction:: pyimgano.preprocessing.compute_glcm_features

Color Space Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.convert_color_space

.. autofunction:: pyimgano.preprocessing.equalize_color_histogram

Supported color spaces:

* ``bgr``: OpenCV default (Blue-Green-Red)
* ``rgb``: Red-Green-Blue
* ``gray``: Grayscale
* ``hsv``: Hue-Saturation-Value
* ``lab``: CIE L*a*b*
* ``ycrcb``: Y'CrCb (luminance-chrominance)
* ``hls``: Hue-Lightness-Saturation

Enhancement
~~~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.gamma_correction

.. autofunction:: pyimgano.preprocessing.contrast_stretching

.. autofunction:: pyimgano.preprocessing.retinex_ssr

.. autofunction:: pyimgano.preprocessing.retinex_msr

Denoising
~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.non_local_means_denoising

.. autofunction:: pyimgano.preprocessing.anisotropic_diffusion

Feature Extraction
~~~~~~~~~~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.extract_hog_features

.. autofunction:: pyimgano.preprocessing.detect_corners

Segmentation
~~~~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.apply_threshold

.. autofunction:: pyimgano.preprocessing.watershed_segmentation

Supported threshold methods:

* ``otsu``: Otsu's method
* ``adaptive_mean``: Adaptive threshold with mean
* ``adaptive_gaussian``: Adaptive threshold with Gaussian
* ``triangle``: Triangle method
* ``yen``: Yen's method
* ``isodata``: Isodata algorithm

Image Pyramids
~~~~~~~~~~~~~~

.. autofunction:: pyimgano.preprocessing.gaussian_pyramid

.. autofunction:: pyimgano.preprocessing.laplacian_pyramid

Enumerations
------------

.. autoclass:: pyimgano.preprocessing.ColorSpace
   :members:
   :undoc-members:

.. autoclass:: pyimgano.preprocessing.ThresholdMethod
   :members:
   :undoc-members:

.. autoclass:: pyimgano.preprocessing.CornerDetector
   :members:
   :undoc-members:

.. autoclass:: pyimgano.preprocessing.MorphologicalAdvanced
   :members:
   :undoc-members:

Examples
--------

Basic Preprocessing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import ImageEnhancer
   import cv2

   # Create enhancer
   enhancer = ImageEnhancer()

   # Load image
   image = cv2.imread('sample.jpg')

   # Edge detection
   edges = enhancer.detect_edges(image, method='canny',
                                  threshold1=100, threshold2=200)

   # Gaussian blur
   blurred = enhancer.apply_filter(image, filter_type='gaussian',
                                    kernel_size=5)

   # Normalization
   normalized = enhancer.normalize(image, method='minmax')

Advanced Preprocessing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import AdvancedImageEnhancer

   # Create advanced enhancer
   enhancer = AdvancedImageEnhancer()

   # Frequency domain
   magnitude, phase = enhancer.apply_fft(image)

   # Texture analysis
   lbp = enhancer.compute_lbp(image, n_points=8, radius=1.0)
   glcm = enhancer.compute_glcm(image)

   # Color space conversion
   lab = enhancer.convert_color(image, from_space='bgr', to_space='lab')

   # Multi-scale Retinex
   enhanced = enhancer.retinex_multi(image, sigmas=[15, 80, 250])

   # HOG features
   features = enhancer.extract_hog(image, orientations=9)

Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import PreprocessingPipeline

   # Define pipeline
   pipeline = PreprocessingPipeline([
       ('edge', 'canny', {'threshold1': 50, 'threshold2': 150}),
       ('filter', 'gaussian', {'kernel_size': 5}),
       ('normalize', 'minmax', {}),
   ])

   # Apply to single image
   processed = pipeline.transform(image)

   # Apply to batch
   images = [cv2.imread(f'img_{i}.jpg') for i in range(10)]
   processed_batch = pipeline.transform_batch(images)

Complete Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import AdvancedImageEnhancer
   from pyimgano.models import create_model
   import numpy as np

   # 1. Preprocessing
   enhancer = AdvancedImageEnhancer()

   def preprocess(img):
       # Convert to LAB
       lab = enhancer.convert_color(img, 'bgr', 'lab')

       # Enhance with Retinex
       enhanced = enhancer.retinex_multi(lab)

       # Extract texture
       lbp = enhancer.compute_lbp(enhanced)

       return lbp.flatten()

   # 2. Preprocess training data
   training_images = [cv2.imread(f'train_{i}.jpg') for i in range(100)]
   X_train = np.array([preprocess(img) for img in training_images])

   # 3. Train detector
   class IdentityExtractor:
       def extract(self, X):
           return np.asarray(X)

   detector = create_model(
       "vision_iforest",
       feature_extractor=IdentityExtractor(),
       contamination=0.1,
       n_estimators=100,
   )
   detector.fit(X_train)

   # 4. Test
   test_image = cv2.imread('test.jpg')
   test_features = preprocess(test_image)
   score = detector.decision_function([test_features])[0]

   print(f"Anomaly score: {score:.4f}")

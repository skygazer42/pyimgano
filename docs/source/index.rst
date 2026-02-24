PyImgAno Documentation
======================

**PyImgAno** is an enterprise-grade visual anomaly detection toolkit with a large model registry,
industrial-ready CLIs, and a strong focus on deployable workflows (artifacts, provenance, JSONL output).

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://github.com/skygazer42/pyimgano/blob/main/LICENSE
   :alt: License

Overview
--------

PyImgAno provides a comprehensive toolkit for visual anomaly detection in industrial inspection, manufacturing quality control, and other computer vision applications.

**Key Features:**

* 🎯 **120+ Registry Model Entries**: Statistical, classical ML, and deep learning methods (native + optional backend wrappers)
* 🖼️ **80+ Preprocessing Operations**: Edge detection, filtering, texture analysis, and more
* 🔄 **Augmentation & Robustness**: Corruptions/augmentations for evaluation and drift testing
* 📊 **Benchmarks + Workbench**: Benchmark runs and recipe-driven workbench runs with JSON artifacts and per-image JSONL
* 🏭 **Industrial Inference (JSONL)**: Deploy-style inference via ``pyimgano-infer`` including anomaly maps and defects export
* 📚 **Well Documented**: Extensive guides, examples, and API reference

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install pyimgano

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from pyimgano.models import create_model

   class IdentityExtractor:
       def extract(self, X):
           return np.asarray(X)

   # Generate sample data
   X_train = np.random.randn(1000, 100) * 0.5  # Normal samples
   X_test = np.random.randn(100, 100) * 2.0    # Test samples

   # Create and train detector
   detector = create_model(
       "vision_iforest",
       feature_extractor=IdentityExtractor(),
       contamination=0.1,
       n_estimators=100,
   )
   detector.fit(X_train)

   # Predict anomaly scores
   scores = detector.decision_function(X_test)
   predictions = detector.predict(X_test)  # 0=normal, 1=anomaly

With Image Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyimgano.preprocessing import AdvancedImageEnhancer
   import cv2
   import numpy as np
   from pyimgano.models import create_model

   # Load and preprocess images
   enhancer = AdvancedImageEnhancer()
   image = cv2.imread('sample.jpg')

   # Extract features
   lbp = enhancer.compute_lbp(image)
   features = lbp.flatten()

   class IdentityExtractor:
       def extract(self, X):
           return np.asarray(X)

   # Train detector
   detector = create_model(
       "vision_auto_encoder",
       feature_extractor=IdentityExtractor(),
       contamination=0.1,
       epoch_num=50,
       lr=1e-3,
       batch_size=32,
       verbose=0,
   )
   detector.fit(training_features)

   # Detect anomalies
   score = detector.decision_function([features])[0]

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   tutorials
   examples
   comparison

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/detectors
   api/preprocessing
   api/augmentation

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   benchmarks
   contributing
   changelog
   license

Algorithms
----------

Statistical Methods
~~~~~~~~~~~~~~~~~~~

* **IQR Detector**: Interquartile range-based outlier detection
* **MAD Detector**: Median Absolute Deviation
* **Z-Score Detector**: Standard score-based detection
* **Histogram-based**: Distribution-based anomaly detection

Distance-based Methods
~~~~~~~~~~~~~~~~~~~~~~

* **KNN**: k-Nearest Neighbors
* **LOF**: Local Outlier Factor
* **COF**: Connectivity-based Outlier Factor
* **LOCI**: Local Correlation Integral

Density-based Methods
~~~~~~~~~~~~~~~~~~~~~

* **ECOD**: Empirical Cumulative Distribution
* **COPOD**: Copula-based Outlier Detection
* **GMM**: Gaussian Mixture Model
* **KDE**: Kernel Density Estimation

Deep Learning Methods
~~~~~~~~~~~~~~~~~~~~~

* **Autoencoder**: Reconstruction-based anomaly detection
* **VAE**: Variational Autoencoder
* **Deep SVDD**: Deep Support Vector Data Description
* **DAGMM**: Deep Autoencoding Gaussian Mixture Model
* **MemAE**: Memory-augmented Autoencoder

And many more! See :doc:`api/detectors` for the complete list.

Preprocessing Operations
------------------------

Basic Operations
~~~~~~~~~~~~~~~~

* **Edge Detection**: Canny, Sobel, Laplacian, Scharr, Prewitt
* **Filters**: Gaussian, Bilateral, Median, Box, Non-local means
* **Morphology**: Erosion, Dilation, Opening, Closing
* **Normalization**: Min-max, Z-score, Robust scaling

Advanced Operations
~~~~~~~~~~~~~~~~~~~

* **Frequency Domain**: FFT, IFFT, frequency filters
* **Texture Analysis**: Gabor filters, LBP, GLCM
* **Enhancement**: Gamma correction, Retinex, contrast stretching
* **Denoising**: Non-local means, anisotropic diffusion
* **Feature Extraction**: HOG, corner detection
* **Segmentation**: Thresholding, Watershed

See :doc:`api/preprocessing` for the complete list.

Use Cases
---------

PyImgAno is designed for:

* 🏭 **Manufacturing Defect Detection**: PCB inspection, surface defects
* 🧵 **Fabric Quality Control**: Textile defect detection
* 🔬 **Medical Imaging**: X-ray, CT scan anomaly detection
* 🛰️ **Satellite Imagery**: Land use change, disaster detection
* 🔒 **Security**: Surveillance anomaly detection
* 🏗️ **Infrastructure**: Crack detection, structure inspection

Performance
-----------

PyImgAno is optimized for both speed and accuracy:

* **Fast Training**: Statistical methods train in < 1 second
* **Fast Inference**: Most algorithms < 10ms per image
* **GPU Accelerated**: Deep learning methods support CUDA
* **Memory Efficient**: Suitable for large-scale deployments

See :doc:`benchmarks` for detailed performance analysis.

Comparison with PyOD
--------------------

PyImgAno is designed for **visual anomaly detection**, while PyOD focuses on **general outlier detection**.

+------------------------+---------------+-------------+
| Feature                | PyImgAno      | PyOD        |
+========================+===============+=============+
| Algorithms             | 37+           | 40+         |
+------------------------+---------------+-------------+
| Image Preprocessing    | 80+ ops       | Minimal     |
+------------------------+---------------+-------------+
| Data Augmentation      | 30+ techniques| None        |
+------------------------+---------------+-------------+
| Focus                  | Images        | Tabular data|
+------------------------+---------------+-------------+

See :doc:`comparison` for a detailed comparison.

Community
---------

* 📖 **Documentation**: https://github.com/skygazer42/pyimgano
* 🐛 **Issue Tracker**: https://github.com/skygazer42/pyimgano/issues
* 💬 **Discussions**: https://github.com/skygazer42/pyimgano/discussions
* ⭐ **Star on GitHub**: https://github.com/skygazer42/pyimgano

Contributing
------------

We welcome contributions! See :doc:`contributing` for guidelines.

License
-------

PyImgAno is released under the MIT License. See :doc:`license` for details.

Citation
--------

If you use PyImgAno in your research, please cite:

.. code-block:: bibtex

   @software{pyimgano2024,
     title={PyImgAno: Enterprise-Grade Visual Anomaly Detection Toolkit},
     author={PyImgAno Contributors},
     year={2024},
     url={https://github.com/jhlu2019/pyimgano}
   }

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

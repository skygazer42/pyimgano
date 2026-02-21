"""
WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation.

Paper: https://arxiv.org/abs/2303.14814
Conference: CVPR 2023

WinCLIP leverages CLIP's visual-language understanding for zero-shot and few-shot
anomaly detection without requiring anomalous samples during training.

Key Features:
- Zero-shot capability using text prompts
- Few-shot learning with minimal normal samples
- Strong localization through window-based attention
- No fine-tuning required
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray as NDArray

from .baseCv import BaseVisionDeepDetector
from .registry import register_model

try:
    import clip
except ImportError:
    clip = None
    warnings.warn(
        "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git",
        UserWarning,
    )


class WindowAttention:
    """Sliding window attention mechanism for fine-grained localization."""

    def __init__(
        self,
        window_size: int = 64,
        stride: int = 32,
    ):
        """Initialize window attention.

        Args:
            window_size: Size of sliding window.
            stride: Stride of sliding window.
        """
        self.window_size = window_size
        self.stride = stride

    def extract_windows(
        self, image: NDArray
    ) -> Tuple[List[NDArray], List[Tuple[int, int]]]:
        """Extract sliding windows from image.

        Args:
            image: Input image (H, W, C).

        Returns:
            List of windows and their positions.
        """
        h, w = image.shape[:2]
        windows = []
        positions = []

        y_positions = range(0, h - self.window_size + 1, self.stride)
        x_positions = range(0, w - self.window_size + 1, self.stride)

        for y in y_positions:
            for x in x_positions:
                window = image[y : y + self.window_size, x : x + self.window_size]
                windows.append(window)
                positions.append((y, x))

        return windows, positions

    def aggregate_scores(
        self,
        scores: List[float],
        positions: List[Tuple[int, int]],
        image_shape: Tuple[int, int],
    ) -> NDArray:
        """Aggregate window scores into anomaly map.

        Args:
            scores: Anomaly scores for each window.
            positions: Window positions.
            image_shape: Original image shape (H, W).

        Returns:
            Anomaly map (H, W).
        """
        h, w = image_shape
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        for score, (y, x) in zip(scores, positions):
            anomaly_map[y : y + self.window_size, x : x + self.window_size] += score
            count_map[y : y + self.window_size, x : x + self.window_size] += 1

        # Average overlapping regions
        count_map = np.maximum(count_map, 1)  # Avoid division by zero
        anomaly_map = anomaly_map / count_map

        return anomaly_map


@register_model("winclip")
class WinCLIPDetector(BaseVisionDeepDetector):
    """WinCLIP zero-shot anomaly detector.

    Uses CLIP's visual-language understanding with sliding window attention
    for zero-shot and few-shot anomaly detection.

    Args:
        clip_model: CLIP model name ("RN50", "RN101", "ViT-B/32", "ViT-L/14").
        window_size: Size of sliding window for localization.
        window_stride: Stride of sliding window.
        text_prompts: Custom text prompts for zero-shot.
                     If None, uses default prompts.
        k_shot: Number of few-shot examples (0 for zero-shot).
        scales: Multi-scale inference scales.
        device: Device to use ("cuda" or "cpu").

    References:
        Jeong et al. "WinCLIP: Zero-/Few-Shot Anomaly Classification
        and Segmentation." CVPR 2023.
    """

    def __init__(
        self,
        clip_model: str = "ViT-B/32",
        window_size: int = 224,
        window_stride: int = 112,
        text_prompts: Optional[List[str]] = None,
        k_shot: int = 0,
        scales: List[float] = [1.0],
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if clip is None:
            raise ImportError(
                "CLIP is required for WinCLIP. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            )

        self.clip_model_name = clip_model
        self.window_size = window_size
        self.window_stride = window_stride
        self.k_shot = k_shot
        self.scales = scales

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Default text prompts for anomaly detection
        if text_prompts is None:
            self.text_prompts = {
                "normal": [
                    "a photo of a normal {}",
                    "a high-quality photo of a {}",
                    "a photo of a perfect {}",
                ],
                "anomaly": [
                    "a photo of a damaged {}",
                    "a photo of a defective {}",
                    "a photo of an anomalous {}",
                ],
            }
        else:
            self.text_prompts = text_prompts

        # Load CLIP model
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        self.model.eval()

        # Window attention
        self.window_attention = WindowAttention(window_size, window_stride)

        # Few-shot features
        self.few_shot_features = None

    def set_class_name(self, class_name: str):
        """Set the object class name for text prompts.

        Args:
            class_name: Name of the object class (e.g., "screw", "tile").
        """
        self.class_name = class_name

        # Format text prompts
        self.formatted_prompts = {
            "normal": [p.format(class_name) for p in self.text_prompts["normal"]],
            "anomaly": [p.format(class_name) for p in self.text_prompts["anomaly"]],
        }

        # Encode text features
        self._encode_text_features()

    def _encode_text_features(self):
        """Encode text prompts into CLIP features."""
        with torch.no_grad():
            # Encode normal prompts
            normal_tokens = clip.tokenize(self.formatted_prompts["normal"]).to(
                self.device
            )
            normal_features = self.model.encode_text(normal_tokens)
            normal_features = F.normalize(normal_features, dim=-1)

            # Encode anomaly prompts
            anomaly_tokens = clip.tokenize(self.formatted_prompts["anomaly"]).to(
                self.device
            )
            anomaly_features = self.model.encode_text(anomaly_tokens)
            anomaly_features = F.normalize(anomaly_features, dim=-1)

        # Store averaged features
        self.text_features_normal = normal_features.mean(dim=0, keepdim=True)
        self.text_features_anomaly = anomaly_features.mean(dim=0, keepdim=True)

    def fit(self, X: NDArray, y: Optional[NDArray] = None, **kwargs):
        """Fit the detector (optional for few-shot learning).

        Args:
            X: Training images (N, H, W, C) - normal samples only.
            y: Not used.
        """
        if self.k_shot > 0:
            # Few-shot learning: store features from normal samples
            if len(X) > self.k_shot:
                # Sample k_shot examples
                indices = np.random.choice(len(X), self.k_shot, replace=False)
                X = X[indices]

            # Extract features
            self.few_shot_features = []
            with torch.no_grad():
                for img in X:
                    img_tensor = self._preprocess_image(img).unsqueeze(0).to(self.device)
                    features = self.model.encode_image(img_tensor)
                    features = F.normalize(features, dim=-1)
                    self.few_shot_features.append(features)

            self.few_shot_features = torch.cat(self.few_shot_features, dim=0)
        else:
            # Zero-shot: no training needed
            pass

    def predict_proba(self, X: NDArray, **kwargs) -> NDArray:
        """Predict anomaly scores.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            Anomaly scores for each sample.
        """
        scores = []

        for img in X:
            if self.window_size < min(img.shape[:2]):
                # Use window-based scoring
                score = self._score_with_windows(img)
            else:
                # Use global scoring
                score = self._score_global(img)

            scores.append(score)

        return np.array(scores)

    def predict_anomaly_map(self, X: NDArray) -> List[NDArray]:
        """Predict pixel-level anomaly maps.

        Args:
            X: Test images (N, H, W, C).

        Returns:
            List of anomaly maps.
        """
        anomaly_maps = []

        for img in X:
            # Extract windows
            windows, positions = self.window_attention.extract_windows(img)

            # Score each window
            window_scores = []
            for window in windows:
                score = self._score_global(window)
                window_scores.append(score)

            # Aggregate into map
            anomaly_map = self.window_attention.aggregate_scores(
                window_scores,
                positions,
                img.shape[:2],
            )

            anomaly_maps.append(anomaly_map)

        return anomaly_maps

    def _score_global(self, image: NDArray) -> float:
        """Score image globally using CLIP.

        Args:
            image: Input image (H, W, C).

        Returns:
            Anomaly score.
        """
        with torch.no_grad():
            img_tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
            img_features = self.model.encode_image(img_tensor)
            img_features = F.normalize(img_features, dim=-1)

            if self.k_shot > 0 and self.few_shot_features is not None:
                # Few-shot: compare with normal examples
                similarities = (img_features @ self.few_shot_features.T).squeeze(0)
                # Low similarity = high anomaly score
                score = 1.0 - similarities.mean().item()
            else:
                # Zero-shot: compare with text prompts
                sim_normal = (img_features @ self.text_features_normal.T).item()
                sim_anomaly = (img_features @ self.text_features_anomaly.T).item()

                # Score based on relative similarity
                score = sim_anomaly / (sim_normal + sim_anomaly + 1e-8)

        return score

    def _score_with_windows(self, image: NDArray) -> float:
        """Score image using window-based approach.

        Args:
            image: Input image (H, W, C).

        Returns:
            Anomaly score (max window score).
        """
        windows, _ = self.window_attention.extract_windows(image)

        window_scores = []
        for window in windows:
            score = self._score_global(window)
            window_scores.append(score)

        # Return max window score as image-level score
        return max(window_scores)

    def _preprocess_image(self, image: NDArray) -> torch.Tensor:
        """Preprocess image for CLIP.

        Args:
            image: Input image (H, W, C) in [0, 255] or [0, 1].

        Returns:
            Preprocessed tensor.
        """
        # Ensure PIL Image
        from PIL import Image

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        pil_img = Image.fromarray(image)
        return self.preprocess(pil_img)

    def multi_scale_predict(self, X: NDArray, scales: Optional[List[float]] = None) -> NDArray:
        """Multi-scale anomaly prediction.

        Args:
            X: Test images (N, H, W, C).
            scales: List of scales to use. If None, uses self.scales.

        Returns:
            Anomaly scores.
        """
        if scales is None:
            scales = self.scales

        all_scores = []

        for scale in scales:
            # Resize images
            if scale != 1.0:
                import cv2

                X_scaled = []
                for img in X:
                    h, w = img.shape[:2]
                    new_h, new_w = int(h * scale), int(w * scale)
                    img_scaled = cv2.resize(img, (new_w, new_h))
                    X_scaled.append(img_scaled)
                X_scaled = np.array(X_scaled)
            else:
                X_scaled = X

            # Predict
            scores = self.predict_proba(X_scaled)
            all_scores.append(scores)

        # Average across scales
        return np.mean(all_scores, axis=0)


@register_model(
    "vision_winclip",
    tags=("vision", "deep", "winclip", "clip"),
    metadata={
        "description": "WinCLIP - Zero-/Few-shot CLIP-based anomaly detection (CVPR 2023)",
        "paper": "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation",
        "year": 2023,
    },
)
class VisionWinCLIP(WinCLIPDetector):
    """Registry alias for `winclip` (naming consistency)."""

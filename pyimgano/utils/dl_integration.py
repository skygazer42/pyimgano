"""
Deep Learning Integration

Features:
- Pre-processing utilities (NMS, thresholding, resizing)
- Post-processing utilities (NMS, confidence filtering)
- ONNX Runtime integration
- TensorRT integration
- Model conversion and optimization
- Batch inference utilities
"""

from typing import Optional, Tuple, List, Union, Dict, Any
import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class PreProcessing:
    """Pre-processing utilities for deep learning models."""

    @staticmethod
    def normalize_imagenet(image: NDArray) -> NDArray:
        """
        Normalize image with ImageNet statistics.

        Parameters
        ----------
        image : ndarray
            Input image (H, W, C) in range [0, 255]

        Returns
        -------
        normalized : ndarray
            Normalized image
        """
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        return (image - mean) / std

    @staticmethod
    def resize_with_aspect_ratio(
        image: NDArray,
        target_size: int,
        max_size: Optional[int] = None
    ) -> Tuple[NDArray, float]:
        """
        Resize maintaining aspect ratio.

        Parameters
        ----------
        image : ndarray
            Input image
        target_size : int
            Target size for shorter side
        max_size : int, optional
            Maximum size for longer side

        Returns
        -------
        resized : ndarray
            Resized image
        scale : float
            Scale factor applied
        """
        h, w = image.shape[:2]
        size = min(h, w)
        scale = target_size / size

        if max_size is not None:
            max_dim = max(h, w)
            if max_dim * scale > max_size:
                scale = max_size / max_dim

        new_h = int(h * scale)
        new_w = int(w * scale)

        try:
            import cv2
            resized = cv2.resize(image, (new_w, new_h))
        except ImportError:
            from PIL import Image
            img = Image.fromarray(image)
            resized = np.array(img.resize((new_w, new_h)))

        return resized, scale

    @staticmethod
    def pad_to_size(
        image: NDArray,
        target_size: Tuple[int, int],
        pad_value: Union[int, float] = 0
    ) -> Tuple[NDArray, Tuple[int, int]]:
        """
        Pad image to target size.

        Parameters
        ----------
        image : ndarray
            Input image
        target_size : tuple
            Target size (height, width)
        pad_value : scalar, default=0
            Padding value

        Returns
        -------
        padded : ndarray
            Padded image
        padding : tuple
            Applied padding (top, left)
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size

        # Center padding
        pad_h = target_h - h
        pad_w = target_w - w

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if len(image.shape) == 3:
            pad_width = ((top, bottom), (left, right), (0, 0))
        else:
            pad_width = ((top, bottom), (left, right))

        padded = np.pad(image, pad_width, mode='constant', constant_values=pad_value)

        return padded, (top, left)

    @staticmethod
    def hwc_to_chw(image: NDArray) -> NDArray:
        """Convert HWC to CHW format."""
        if len(image.shape) == 3:
            return np.transpose(image, (2, 0, 1))
        return image


class PostProcessing:
    """Post-processing utilities."""

    @staticmethod
    def non_max_suppression(
        boxes: NDArray,
        scores: NDArray,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.0
    ) -> NDArray:
        """
        Non-Maximum Suppression.

        Parameters
        ----------
        boxes : ndarray
            Bounding boxes (N, 4) in format [x1, y1, x2, y2]
        scores : ndarray
            Confidence scores (N,)
        iou_threshold : float, default=0.5
            IoU threshold for suppression
        score_threshold : float, default=0.0
            Minimum score threshold

        Returns
        -------
        keep : ndarray
            Indices of boxes to keep
        """
        # Filter by score threshold
        mask = scores > score_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)

    @staticmethod
    def confidence_filtering(
        boxes: NDArray,
        scores: NDArray,
        classes: Optional[NDArray] = None,
        threshold: float = 0.5
    ) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
        """
        Filter detections by confidence.

        Parameters
        ----------
        boxes : ndarray
            Bounding boxes (N, 4)
        scores : ndarray
            Confidence scores (N,)
        classes : ndarray, optional
            Class predictions (N,)
        threshold : float, default=0.5
            Confidence threshold

        Returns
        -------
        filtered_boxes : ndarray
            Filtered boxes
        filtered_scores : ndarray
            Filtered scores
        filtered_classes : ndarray or None
            Filtered classes (if provided)
        """
        mask = scores >= threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_classes = classes[mask] if classes is not None else None

        return filtered_boxes, filtered_scores, filtered_classes

    @staticmethod
    def soft_nms(
        boxes: NDArray,
        scores: NDArray,
        sigma: float = 0.5,
        score_threshold: float = 0.001
    ) -> Tuple[NDArray, NDArray]:
        """
        Soft Non-Maximum Suppression.

        Parameters
        ----------
        boxes : ndarray
            Bounding boxes (N, 4)
        scores : ndarray
            Confidence scores (N,)
        sigma : float, default=0.5
            Gaussian sigma for score decay
        score_threshold : float, default=0.001
            Minimum score threshold

        Returns
        -------
        keep_boxes : ndarray
            Kept boxes
        keep_scores : ndarray
            Updated scores
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        scores = scores.copy()

        keep_boxes = []
        keep_scores = []

        while scores.max() > score_threshold:
            i = scores.argmax()
            keep_boxes.append(boxes[i])
            keep_scores.append(scores[i])

            # Compute IoU
            xx1 = np.maximum(x1[i], x1)
            yy1 = np.maximum(y1[i], y1)
            xx2 = np.minimum(x2[i], x2)
            yy2 = np.minimum(y2[i], y2)

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas - inter)

            # Gaussian decay
            weight = np.exp(-(iou ** 2) / sigma)
            scores = scores * weight
            scores[i] = 0  # Remove current box

        return np.array(keep_boxes), np.array(keep_scores)


class ONNXWrapper:
    """ONNX Runtime wrapper for inference."""

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None
    ):
        """
        Initialize ONNX model.

        Parameters
        ----------
        model_path : str
            Path to ONNX model file
        providers : list, optional
            Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required")

        if providers is None:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def get_input_shape(self) -> Tuple:
        """Get model input shape."""
        return self.session.get_inputs()[0].shape

    def get_output_shape(self) -> Tuple:
        """Get model output shape."""
        return self.session.get_outputs()[0].shape

    def infer(self, inputs: Union[NDArray, Dict[str, NDArray]]) -> Union[NDArray, List[NDArray]]:
        """
        Run inference.

        Parameters
        ----------
        inputs : ndarray or dict
            Input data. If ndarray, uses first input name.
            If dict, maps input names to arrays.

        Returns
        -------
        outputs : ndarray or list
            Model outputs
        """
        if isinstance(inputs, dict):
            input_feed = inputs
        else:
            input_feed = {self.input_names[0]: inputs}

        outputs = self.session.run(self.output_names, input_feed)

        return outputs[0] if len(outputs) == 1 else outputs


class TorchWrapper:
    """PyTorch model wrapper for inference."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize PyTorch model wrapper.

        Parameters
        ----------
        model : nn.Module
            PyTorch model
        device : str, default='cuda'
            Device for inference
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required")

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()

    def infer(self, inputs: NDArray) -> NDArray:
        """
        Run inference.

        Parameters
        ----------
        inputs : ndarray
            Input data

        Returns
        -------
        outputs : ndarray
            Model outputs
        """
        with torch.no_grad():
            inputs_tensor = torch.from_numpy(inputs).to(self.device)
            outputs_tensor = self.model(inputs_tensor)

            if isinstance(outputs_tensor, tuple):
                outputs = [t.cpu().numpy() for t in outputs_tensor]
            else:
                outputs = outputs_tensor.cpu().numpy()

        return outputs

    def export_to_onnx(
        self,
        output_path: str,
        input_shape: Tuple[int, ...],
        opset_version: int = 11
    ):
        """
        Export model to ONNX format.

        Parameters
        ----------
        output_path : str
            Output ONNX file path
        input_shape : tuple
            Input shape (with batch dimension)
        opset_version : int, default=11
            ONNX opset version
        """
        dummy_input = torch.randn(*input_shape).to(self.device)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )


class BatchInference:
    """Batch inference utilities."""

    @staticmethod
    def process_batches(
        images: List[NDArray],
        model_fn: callable,
        batch_size: int = 32,
        preprocess_fn: Optional[callable] = None,
        postprocess_fn: Optional[callable] = None
    ) -> List[Any]:
        """
        Process images in batches.

        Parameters
        ----------
        images : list
            List of images
        model_fn : callable
            Model inference function
        batch_size : int, default=32
            Batch size
        preprocess_fn : callable, optional
            Pre-processing function
        postprocess_fn : callable, optional
            Post-processing function

        Returns
        -------
        results : list
            Inference results
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Pre-process
            if preprocess_fn:
                batch = [preprocess_fn(img) for img in batch]

            # Stack into batch
            batch_array = np.stack(batch)

            # Inference
            outputs = model_fn(batch_array)

            # Post-process
            if postprocess_fn:
                if isinstance(outputs, np.ndarray):
                    for j in range(len(outputs)):
                        results.append(postprocess_fn(outputs[j]))
                else:
                    results.append(postprocess_fn(outputs))
            else:
                if isinstance(outputs, np.ndarray):
                    results.extend([outputs[j] for j in range(len(outputs))])
                else:
                    results.append(outputs)

        return results


# Convenience functions
def nms(
    boxes: NDArray,
    scores: NDArray,
    iou_threshold: float = 0.5
) -> NDArray:
    """
    Non-Maximum Suppression.

    Parameters
    ----------
    boxes : ndarray
        Bounding boxes (N, 4)
    scores : ndarray
        Confidence scores (N,)
    iou_threshold : float, default=0.5
        IoU threshold

    Returns
    -------
    keep : ndarray
        Indices to keep
    """
    return PostProcessing.non_max_suppression(boxes, scores, iou_threshold)


def load_onnx_model(
    model_path: str,
    use_gpu: bool = True
) -> ONNXWrapper:
    """
    Load ONNX model.

    Parameters
    ----------
    model_path : str
        Path to ONNX model
    use_gpu : bool, default=True
        Use GPU if available

    Returns
    -------
    model : ONNXWrapper
        Loaded model wrapper
    """
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    return ONNXWrapper(model_path, providers=providers)


def batch_inference(
    images: List[NDArray],
    model: Union[ONNXWrapper, TorchWrapper],
    batch_size: int = 32
) -> List[NDArray]:
    """
    Run batch inference.

    Parameters
    ----------
    images : list
        List of images
    model : ONNXWrapper or TorchWrapper
        Model wrapper
    batch_size : int, default=32
        Batch size

    Returns
    -------
    results : list
        Inference results
    """
    return BatchInference.process_batches(
        images,
        model.infer,
        batch_size=batch_size
    )

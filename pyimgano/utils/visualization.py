"""
Visualization Utilities

Features:
- Bounding box visualization
- Polygon and mask visualization
- Keypoint visualization
- Heatmap and attention visualization
- Multilingual text rendering
- Jupyter notebook integration
- Web-based visualization
"""

from typing import List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class BBoxVisualizer:
    """Bounding box visualization."""

    @staticmethod
    def draw_bbox(
        image: NDArray,
        bbox: Tuple[float, float, float, float],
        label: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.5
    ) -> NDArray:
        """
        Draw bounding box on image.

        Parameters
        ----------
        image : ndarray
            Input image (RGB or BGR)
        bbox : tuple
            Bounding box (x, y, width, height)
        label : str, optional
            Text label
        color : tuple, default=(0, 255, 0)
            Box color (RGB)
        thickness : int, default=2
            Line thickness
        font_scale : float, default=0.5
            Font scale for label

        Returns
        -------
        visualized : ndarray
            Image with bounding box
        """
        image = image.copy()
        x, y, w, h = map(int, bbox)

        if HAS_OPENCV:
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

            # Draw label
            if label:
                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                # Draw background rectangle
                cv2.rectangle(
                    image,
                    (x, y - text_h - baseline - 5),
                    (x + text_w, y),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    image,
                    label,
                    (x, y - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
        elif HAS_PIL:
            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)

            # Draw rectangle
            draw.rectangle([x, y, x + w, y + h], outline=tuple(color), width=thickness)

            # Draw label
            if label:
                try:
                    font = ImageFont.truetype("arial.ttf", int(font_scale * 20))
                except:
                    font = ImageFont.load_default()

                bbox_text = draw.textbbox((x, y), label, font=font)
                text_h = bbox_text[3] - bbox_text[1]

                draw.rectangle([x, y - text_h, x + w, y], fill=tuple(color))
                draw.text((x, y - text_h), label, fill=(255, 255, 255), font=font)

            image = np.array(img)

        return image

    @staticmethod
    def draw_bboxes(
        image: NDArray,
        bboxes: List[Tuple[float, float, float, float]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        thickness: int = 2
    ) -> NDArray:
        """
        Draw multiple bounding boxes.

        Parameters
        ----------
        image : ndarray
            Input image
        bboxes : list
            List of bounding boxes
        labels : list, optional
            List of labels
        colors : list, optional
            List of colors (one per box)
        thickness : int, default=2
            Line thickness

        Returns
        -------
        visualized : ndarray
            Image with all boxes
        """
        result = image.copy()

        for i, bbox in enumerate(bboxes):
            label = labels[i] if labels and i < len(labels) else None
            color = colors[i] if colors and i < len(colors) else (0, 255, 0)

            result = BBoxVisualizer.draw_bbox(
                result, bbox, label, color, thickness
            )

        return result


class MaskVisualizer:
    """Mask and polygon visualization."""

    @staticmethod
    def draw_mask(
        image: NDArray,
        mask: NDArray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5
    ) -> NDArray:
        """
        Draw segmentation mask overlay.

        Parameters
        ----------
        image : ndarray
            Input image (H, W, 3)
        mask : ndarray
            Binary mask (H, W)
        color : tuple, default=(0, 255, 0)
            Mask color (RGB)
        alpha : float, default=0.5
            Transparency

        Returns
        -------
        visualized : ndarray
            Image with mask overlay
        """
        overlay = image.copy()

        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color

        # Blend
        result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0) if HAS_OPENCV else \
                 (overlay * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

        return result

    @staticmethod
    def draw_polygon(
        image: NDArray,
        polygon: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        fill: bool = False,
        alpha: float = 0.3
    ) -> NDArray:
        """
        Draw polygon.

        Parameters
        ----------
        image : ndarray
            Input image
        polygon : list
            List of (x, y) points
        color : tuple, default=(0, 255, 0)
            Polygon color
        thickness : int, default=2
            Line thickness
        fill : bool, default=False
            Fill polygon
        alpha : float, default=0.3
            Fill transparency

        Returns
        -------
        visualized : ndarray
            Image with polygon
        """
        result = image.copy()
        points = np.array(polygon, dtype=np.int32)

        if HAS_OPENCV:
            if fill:
                overlay = result.copy()
                cv2.fillPoly(overlay, [points], color)
                result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
            else:
                cv2.polylines(result, [points], True, color, thickness)
        elif HAS_PIL:
            img = Image.fromarray(result)
            draw = ImageDraw.Draw(img)

            if fill:
                draw.polygon([tuple(p) for p in points], fill=tuple(color))
            else:
                draw.polygon([tuple(p) for p in points], outline=tuple(color), width=thickness)

            result = np.array(img)

        return result


class KeypointVisualizer:
    """Keypoint visualization."""

    @staticmethod
    def draw_keypoints(
        image: NDArray,
        keypoints: List[Tuple[int, int]],
        radius: int = 5,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = -1
    ) -> NDArray:
        """
        Draw keypoints.

        Parameters
        ----------
        image : ndarray
            Input image
        keypoints : list
            List of (x, y) keypoint coordinates
        radius : int, default=5
            Keypoint radius
        color : tuple, default=(0, 255, 0)
            Keypoint color
        thickness : int, default=-1
            Circle thickness (-1 for filled)

        Returns
        -------
        visualized : ndarray
            Image with keypoints
        """
        result = image.copy()

        for x, y in keypoints:
            if HAS_OPENCV:
                cv2.circle(result, (int(x), int(y)), radius, color, thickness)
            elif HAS_PIL:
                img = Image.fromarray(result)
                draw = ImageDraw.Draw(img)
                draw.ellipse(
                    [x - radius, y - radius, x + radius, y + radius],
                    fill=tuple(color) if thickness == -1 else None,
                    outline=tuple(color),
                    width=max(1, thickness)
                )
                result = np.array(img)

        return result

    @staticmethod
    def draw_skeleton(
        image: NDArray,
        keypoints: List[Tuple[int, int]],
        skeleton: List[Tuple[int, int]],
        keypoint_radius: int = 5,
        line_thickness: int = 2,
        keypoint_color: Tuple[int, int, int] = (0, 255, 0),
        line_color: Tuple[int, int, int] = (255, 0, 0)
    ) -> NDArray:
        """
        Draw skeleton with connections.

        Parameters
        ----------
        image : ndarray
            Input image
        keypoints : list
            List of (x, y) keypoint coordinates
        skeleton : list
            List of (start_idx, end_idx) connections
        keypoint_radius : int, default=5
            Keypoint radius
        line_thickness : int, default=2
            Connection line thickness
        keypoint_color : tuple, default=(0, 255, 0)
            Keypoint color
        line_color : tuple, default=(255, 0, 0)
            Connection line color

        Returns
        -------
        visualized : ndarray
            Image with skeleton
        """
        result = image.copy()

        # Draw connections
        for start_idx, end_idx in skeleton:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                pt1 = tuple(map(int, keypoints[start_idx]))
                pt2 = tuple(map(int, keypoints[end_idx]))

                if HAS_OPENCV:
                    cv2.line(result, pt1, pt2, line_color, line_thickness)
                elif HAS_PIL:
                    img = Image.fromarray(result)
                    draw = ImageDraw.Draw(img)
                    draw.line([pt1, pt2], fill=tuple(line_color), width=line_thickness)
                    result = np.array(img)

        # Draw keypoints
        result = KeypointVisualizer.draw_keypoints(
            result, keypoints, keypoint_radius, keypoint_color
        )

        return result


class HeatmapVisualizer:
    """Heatmap and attention visualization."""

    @staticmethod
    def draw_heatmap(
        heatmap: NDArray,
        colormap: str = 'jet',
        normalize: bool = True
    ) -> NDArray:
        """
        Convert heatmap to colored image.

        Parameters
        ----------
        heatmap : ndarray
            Heatmap array (H, W)
        colormap : str, default='jet'
            Colormap name
        normalize : bool, default=True
            Normalize to [0, 1]

        Returns
        -------
        colored : ndarray
            Colored heatmap (H, W, 3)
        """
        if normalize:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Convert to uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)

        if HAS_OPENCV:
            # OpenCV colormaps
            colormap_map = {
                'jet': cv2.COLORMAP_JET,
                'hot': cv2.COLORMAP_HOT,
                'cool': cv2.COLORMAP_COOL,
                'viridis': cv2.COLORMAP_VIRIDIS,
                'plasma': cv2.COLORMAP_PLASMA,
            }
            cmap = colormap_map.get(colormap, cv2.COLORMAP_JET)
            colored = cv2.applyColorMap(heatmap_uint8, cmap)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        elif HAS_MATPLOTLIB:
            import matplotlib.cm as cm
            cmap_fn = plt.get_cmap(colormap)
            colored = (cmap_fn(heatmap)[:, :, :3] * 255).astype(np.uint8)
        else:
            # Fallback: grayscale
            colored = np.stack([heatmap_uint8] * 3, axis=-1)

        return colored

    @staticmethod
    def overlay_heatmap(
        image: NDArray,
        heatmap: NDArray,
        alpha: float = 0.5,
        colormap: str = 'jet'
    ) -> NDArray:
        """
        Overlay heatmap on image.

        Parameters
        ----------
        image : ndarray
            Input image (H, W, 3)
        heatmap : ndarray
            Heatmap (H, W)
        alpha : float, default=0.5
            Heatmap transparency
        colormap : str, default='jet'
            Colormap name

        Returns
        -------
        overlaid : ndarray
            Image with heatmap overlay
        """
        # Resize heatmap to match image size
        if heatmap.shape[:2] != image.shape[:2]:
            if HAS_OPENCV:
                heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            else:
                from scipy.ndimage import zoom
                zoom_factors = (image.shape[0] / heatmap.shape[0],
                               image.shape[1] / heatmap.shape[1])
                heatmap = zoom(heatmap, zoom_factors, order=1)

        # Convert heatmap to colored
        colored_heatmap = HeatmapVisualizer.draw_heatmap(heatmap, colormap)

        # Blend
        if HAS_OPENCV:
            overlaid = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)
        else:
            overlaid = (image * (1 - alpha) + colored_heatmap * alpha).astype(np.uint8)

        return overlaid


class TextRenderer:
    """Text rendering with multilingual support."""

    @staticmethod
    def draw_text(
        image: NDArray,
        text: str,
        position: Tuple[int, int],
        font_path: Optional[str] = None,
        font_size: int = 20,
        color: Tuple[int, int, int] = (255, 255, 255),
        background_color: Optional[Tuple[int, int, int]] = None
    ) -> NDArray:
        """
        Draw text with custom font.

        Parameters
        ----------
        image : ndarray
            Input image
        text : str
            Text to draw
        position : tuple
            Text position (x, y)
        font_path : str, optional
            Path to TTF font file
        font_size : int, default=20
            Font size
        color : tuple, default=(255, 255, 255)
            Text color
        background_color : tuple, optional
            Background color

        Returns
        -------
        result : ndarray
            Image with text
        """
        if not HAS_PIL:
            raise ImportError("PIL is required for text rendering")

        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)

        # Load font
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

        # Draw background
        if background_color:
            bbox = draw.textbbox(position, text, font=font)
            draw.rectangle(bbox, fill=tuple(background_color))

        # Draw text
        draw.text(position, text, fill=tuple(color), font=font)

        return np.array(img)


# Convenience functions
def show_image(
    image: NDArray,
    title: str = "Image",
    figsize: Tuple[int, int] = (10, 10)
):
    """
    Display image using matplotlib.

    Parameters
    ----------
    image : ndarray
        Image to display
    title : str, default="Image"
        Figure title
    figsize : tuple, default=(10, 10)
        Figure size
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for show_image")

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()


def visualize_detections(
    image: NDArray,
    boxes: List[Tuple[float, float, float, float]],
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None
) -> NDArray:
    """
    Visualize object detections.

    Parameters
    ----------
    image : ndarray
        Input image
    boxes : list
        List of bounding boxes
    labels : list, optional
        Class labels
    scores : list, optional
        Confidence scores
    class_colors : dict, optional
        Mapping from class names to colors

    Returns
    -------
    visualized : ndarray
        Annotated image
    """
    result = image.copy()

    for i, box in enumerate(boxes):
        label = labels[i] if labels and i < len(labels) else None
        score = scores[i] if scores and i < len(scores) else None

        # Create label text
        if label and score:
            text = f"{label}: {score:.2f}"
        elif label:
            text = label
        else:
            text = None

        # Get color
        if class_colors and label and label in class_colors:
            color = class_colors[label]
        else:
            color = (0, 255, 0)

        result = BBoxVisualizer.draw_bbox(result, box, text, color)

    return result

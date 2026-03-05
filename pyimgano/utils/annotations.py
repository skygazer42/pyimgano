"""
Annotation Format Conversion

Features:
- COCO format support
- YOLO format support
- Pascal VOC format support
- Format conversion utilities
- Annotation validation
- Visualization of annotations
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundingBox:
    """Bounding box representation."""
    x: float
    y: float
    width: float
    height: float
    class_id: int
    class_name: str = ""
    confidence: float = 1.0

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert to (x, y, w, h) format."""
        return (self.x, self.y, self.width, self.height)

    def to_cxcywh(self) -> Tuple[float, float, float, float]:
        """Convert to (cx, cy, w, h) format."""
        cx = self.x + self.width / 2
        cy = self.y + self.height / 2
        return (cx, cy, self.width, self.height)


class COCOFormat:
    """COCO format utilities."""

    @staticmethod
    def load(annotation_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Load COCO annotation file.

        Parameters
        ----------
        annotation_file : str or Path
            Path to COCO JSON file

        Returns
        -------
        annotations : dict
            COCO annotations
        """
        with open(annotation_file, 'r') as f:
            return json.load(f)

    @staticmethod
    def save(annotations: Dict[str, Any], output_file: Union[str, Path]):
        """
        Save COCO annotations.

        Parameters
        ----------
        annotations : dict
            COCO annotations
        output_file : str or Path
            Output JSON file path
        """
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)

    @staticmethod
    def get_image_annotations(
        coco_data: Dict[str, Any],
        image_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get annotations for specific image.

        Parameters
        ----------
        coco_data : dict
            COCO dataset
        image_id : int
            Image ID

        Returns
        -------
        annotations : list
            List of annotations for the image
        """
        return [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    @staticmethod
    def create_annotation(
        image_id: int,
        category_id: int,
        bbox: Tuple[float, float, float, float],
        segmentation: Optional[List] = None,
        area: Optional[float] = None,
        iscrowd: int = 0
    ) -> Dict[str, Any]:
        """
        Create COCO annotation.

        Parameters
        ----------
        image_id : int
            Image ID
        category_id : int
            Category ID
        bbox : tuple
            Bounding box (x, y, width, height)
        segmentation : list, optional
            Segmentation polygons
        area : float, optional
            Annotation area
        iscrowd : int, default=0
            Is crowd annotation

        Returns
        -------
        annotation : dict
            COCO annotation dictionary
        """
        x, y, w, h = bbox

        if area is None:
            area = w * h

        ann = {
            'image_id': image_id,
            'category_id': category_id,
            'bbox': [x, y, w, h],
            'area': area,
            'iscrowd': iscrowd
        }

        if segmentation:
            ann['segmentation'] = segmentation

        return ann


class YOLOFormat:
    """YOLO format utilities."""

    @staticmethod
    def load(
        label_file: Union[str, Path],
        class_names: Optional[List[str]] = None
    ) -> List[BoundingBox]:
        """
        Load YOLO format labels.

        Parameters
        ----------
        label_file : str or Path
            Path to YOLO .txt file
        class_names : list, optional
            List of class names

        Returns
        -------
        boxes : list
            List of BoundingBox objects
        """
        boxes = []

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    confidence = float(parts[5]) if len(parts) > 5 else 1.0

                    # Convert from normalized center format to absolute corner format
                    # Note: This assumes image size is known elsewhere
                    class_name = class_names[class_id] if class_names and class_id < len(class_names) else ""

                    box = BoundingBox(
                        x=cx,  # Will need image size to convert properly
                        y=cy,
                        width=w,
                        height=h,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence
                    )
                    boxes.append(box)

        return boxes

    @staticmethod
    def save(
        boxes: List[BoundingBox],
        output_file: Union[str, Path],
        image_size: Tuple[int, int]
    ):
        """
        Save YOLO format labels.

        Parameters
        ----------
        boxes : list
            List of BoundingBox objects
        output_file : str or Path
            Output .txt file path
        image_size : tuple
            Image size (width, height) for normalization
        """
        img_w, img_h = image_size

        with open(output_file, 'w') as f:
            for box in boxes:
                # Convert to normalized center format
                cx = (box.x + box.width / 2) / img_w
                cy = (box.y + box.height / 2) / img_h
                w = box.width / img_w
                h = box.height / img_h

                line = f"{box.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                if box.confidence < 1.0:
                    line += f" {box.confidence:.6f}"
                line += "\n"

                f.write(line)

    @staticmethod
    def denormalize_bbox(
        normalized_box: Tuple[float, float, float, float],
        image_size: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        """
        Denormalize YOLO bounding box.

        Parameters
        ----------
        normalized_box : tuple
            Normalized box (cx, cy, w, h) in [0, 1]
        image_size : tuple
            Image size (width, height)

        Returns
        -------
        box : tuple
            Absolute box (x, y, w, h)
        """
        cx_norm, cy_norm, w_norm, h_norm = normalized_box
        img_w, img_h = image_size

        w = w_norm * img_w
        h = h_norm * img_h
        x = (cx_norm * img_w) - (w / 2)
        y = (cy_norm * img_h) - (h / 2)

        return (x, y, w, h)


class VOCFormat:
    """Pascal VOC format utilities."""

    @staticmethod
    def load(xml_file: Union[str, Path]) -> Tuple[Dict[str, Any], List[BoundingBox]]:
        """
        Load Pascal VOC XML annotation.

        Parameters
        ----------
        xml_file : str or Path
            Path to XML file

        Returns
        -------
        metadata : dict
            Image metadata
        boxes : list
            List of BoundingBox objects
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract metadata
        size = root.find('size')
        metadata = {
            'filename': root.find('filename').text if root.find('filename') is not None else '',
            'width': int(size.find('width').text) if size is not None else 0,
            'height': int(size.find('height').text) if size is not None else 0,
            'depth': int(size.find('depth').text) if size is not None and size.find('depth') is not None else 3,
        }

        # Extract boxes
        boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')

            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            box = BoundingBox(
                x=xmin,
                y=ymin,
                width=xmax - xmin,
                height=ymax - ymin,
                class_id=-1,  # VOC doesn't have numeric class IDs
                class_name=name
            )
            boxes.append(box)

        return metadata, boxes

    @staticmethod
    def save(
        boxes: List[BoundingBox],
        image_info: Dict[str, Any],
        output_file: Union[str, Path]
    ):
        """
        Save Pascal VOC XML annotation.

        Parameters
        ----------
        boxes : list
            List of BoundingBox objects
        image_info : dict
            Image metadata (filename, width, height, depth)
        output_file : str or Path
            Output XML file path
        """
        root = ET.Element('annotation')

        # Filename
        ET.SubElement(root, 'filename').text = image_info.get('filename', '')

        # Size
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(image_info.get('width', 0))
        ET.SubElement(size, 'height').text = str(image_info.get('height', 0))
        ET.SubElement(size, 'depth').text = str(image_info.get('depth', 3))

        # Objects
        for box in boxes:
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = box.class_name

            bndbox = ET.SubElement(obj, 'bndbox')
            x1, y1, x2, y2 = box.to_xyxy()
            ET.SubElement(bndbox, 'xmin').text = str(int(x1))
            ET.SubElement(bndbox, 'ymin').text = str(int(y1))
            ET.SubElement(bndbox, 'xmax').text = str(int(x2))
            ET.SubElement(bndbox, 'ymax').text = str(int(y2))

        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_file, encoding='utf-8', xml_declaration=True)


class FormatConverter:
    """Convert between annotation formats."""

    @staticmethod
    def coco_to_yolo(
        coco_file: Union[str, Path],
        output_dir: Union[str, Path],
        class_mapping: Optional[Dict[int, int]] = None
    ):
        """
        Convert COCO to YOLO format.

        Parameters
        ----------
        coco_file : str or Path
            Input COCO JSON file
        output_dir : str or Path
            Output directory for YOLO .txt files
        class_mapping : dict, optional
            Map COCO category IDs to YOLO class IDs
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        coco_data = COCOFormat.load(coco_file)

        # Create image ID to filename mapping
        image_dict = {img['id']: img for img in coco_data['images']}

        # Group annotations by image
        for image_id, image_info in image_dict.items():
            annotations = COCOFormat.get_image_annotations(coco_data, image_id)

            if not annotations:
                continue

            boxes = []
            for ann in annotations:
                class_id = ann['category_id']
                if class_mapping:
                    class_id = class_mapping.get(class_id, class_id)

                x, y, w, h = ann['bbox']
                box = BoundingBox(x=x, y=y, width=w, height=h, class_id=class_id)
                boxes.append(box)

            # Save YOLO format
            image_filename = Path(image_info['file_name']).stem
            output_file = output_dir / f"{image_filename}.txt"

            YOLOFormat.save(
                boxes,
                output_file,
                (image_info['width'], image_info['height'])
            )

    @staticmethod
    def yolo_to_coco(
        yolo_dir: Union[str, Path],
        image_dir: Union[str, Path],
        output_file: Union[str, Path],
        class_names: List[str]
    ):
        """
        Convert YOLO to COCO format.

        Parameters
        ----------
        yolo_dir : str or Path
            Directory containing YOLO .txt files
        image_dir : str or Path
            Directory containing images
        output_file : str or Path
            Output COCO JSON file
        class_names : list
            List of class names
        """
        yolo_dir = Path(yolo_dir)
        image_dir = Path(image_dir)

        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }

        # Create categories
        for i, name in enumerate(class_names):
            coco_data['categories'].append({
                'id': i,
                'name': name,
                'supercategory': 'object'
            })

        # Process each label file
        ann_id = 0
        for img_id, label_file in enumerate(sorted(yolo_dir.glob('*.txt'))):
            # Find corresponding image
            image_stem = label_file.stem
            image_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = image_dir / f"{image_stem}{ext}"
                if candidate.exists():
                    image_file = candidate
                    break

            if not image_file:
                continue

            # Get image size
            try:
                from PIL import Image
                with Image.open(image_file) as img:
                    width, height = img.size
            except:
                import cv2
                img = cv2.imread(str(image_file))
                height, width = img.shape[:2]

            # Add image info
            coco_data['images'].append({
                'id': img_id,
                'file_name': image_file.name,
                'width': width,
                'height': height
            })

            # Load and convert boxes
            boxes = YOLOFormat.load(label_file, class_names)

            for box in boxes:
                # Denormalize
                abs_box = YOLOFormat.denormalize_bbox(
                    box.to_cxcywh(),
                    (width, height)
                )

                ann = COCOFormat.create_annotation(
                    image_id=img_id,
                    category_id=box.class_id,
                    bbox=abs_box
                )
                ann['id'] = ann_id
                coco_data['annotations'].append(ann)
                ann_id += 1

        # Save
        COCOFormat.save(coco_data, output_file)

    @staticmethod
    def voc_to_yolo(
        voc_dir: Union[str, Path],
        output_dir: Union[str, Path],
        class_names: List[str]
    ):
        """
        Convert Pascal VOC to YOLO format.

        Parameters
        ----------
        voc_dir : str or Path
            Directory containing VOC XML files
        output_dir : str or Path
            Output directory for YOLO .txt files
        class_names : list
            List of class names for mapping
        """
        voc_dir = Path(voc_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create class name to ID mapping
        class_to_id = {name: i for i, name in enumerate(class_names)}

        for xml_file in voc_dir.glob('*.xml'):
            metadata, boxes = VOCFormat.load(xml_file)

            # Convert class names to IDs
            yolo_boxes = []
            for box in boxes:
                if box.class_name in class_to_id:
                    box.class_id = class_to_id[box.class_name]
                    yolo_boxes.append(box)

            # Save YOLO format
            output_file = output_dir / f"{xml_file.stem}.txt"
            YOLOFormat.save(
                yolo_boxes,
                output_file,
                (metadata['width'], metadata['height'])
            )


# Convenience functions
def validate_annotations(
    boxes: List[BoundingBox],
    image_size: Tuple[int, int]
) -> List[str]:
    """
    Validate bounding box annotations.

    Parameters
    ----------
    boxes : list
        List of bounding boxes
    image_size : tuple
        Image size (width, height)

    Returns
    -------
    errors : list
        List of validation error messages
    """
    errors = []
    img_w, img_h = image_size

    for i, box in enumerate(boxes):
        # Check bounds
        if box.x < 0 or box.y < 0:
            errors.append(f"Box {i}: Negative coordinates")

        if box.width <= 0 or box.height <= 0:
            errors.append(f"Box {i}: Non-positive dimensions")

        x2 = box.x + box.width
        y2 = box.y + box.height

        if x2 > img_w or y2 > img_h:
            errors.append(f"Box {i}: Exceeds image bounds")

        # Check class ID
        if box.class_id < 0:
            errors.append(f"Box {i}: Invalid class ID")

    return errors

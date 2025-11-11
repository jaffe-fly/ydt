"""
YOLO format handling utilities

Supports both OBB (Oriented Bounding Box) and regular bounding box formats.

OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 values, normalized 0-1)
BBox format: class_id x_center y_center width height (5 values, normalized 0-1)
"""

import numpy as np
from typing import List, Tuple, Union, Literal
from dataclasses import dataclass
from pathlib import Path

FormatType = Literal["obb", "bbox"]


@dataclass
class BBoxFormat:
    """Regular bounding box format"""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_list(self) -> List[Union[int, float]]:
        """Convert to list format"""
        return [self.class_id, self.x_center, self.y_center, self.width, self.height]

    def to_string(self) -> str:
        """Convert to YOLO format string"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


@dataclass
class OBBFormat:
    """Oriented bounding box format (4 corner points)"""

    class_id: int
    points: np.ndarray  # shape: (4, 2) - 4 points with (x, y) coordinates

    def to_list(self) -> List[Union[int, float]]:
        """Convert to list format"""
        flat_points = self.points.flatten().tolist()
        return [self.class_id] + flat_points

    def to_string(self) -> str:
        """Convert to YOLO OBB format string"""
        points_str = " ".join([f"{coord:.6f}" for coord in self.points.flatten()])
        return f"{self.class_id} {points_str}"


def parse_label_line(
    line: str, format_type: FormatType = "bbox"
) -> Union[BBoxFormat, OBBFormat, None]:
    """
    Parse a label line into appropriate format

    Args:
        line: Label line string
        format_type: Expected format type ('bbox' or 'obb')

    Returns:
        Parsed format object or None if invalid

    Examples:
        >>> parse_label_line("0 0.5 0.5 0.3 0.4", "bbox")
        BBoxFormat(class_id=0, x_center=0.5, y_center=0.5, width=0.3, height=0.4)

        >>> parse_label_line("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9", "obb")
        OBBFormat(class_id=0, points=array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]))
    """
    parts = line.strip().split()

    if not parts:
        return None

    try:
        class_id = int(parts[0])
        values = [float(x) for x in parts[1:]]

        if format_type == "bbox":
            if len(values) != 4:
                return None
            return BBoxFormat(class_id, values[0], values[1], values[2], values[3])

        elif format_type == "obb":
            if len(values) != 8:
                return None
            # Reshape to (4, 2) for 4 points
            points = np.array(values).reshape(4, 2)
            return OBBFormat(class_id, points)

    except (ValueError, IndexError):
        return None

    return None


def detect_format(label_file: Union[str, Path] = None, label_line: str = None) -> FormatType:
    """
    Auto-detect format type from label file or label line

    Args:
        label_file: Path to label file (deprecated, use label_line instead)
        label_line: Single label line string to analyze

    Returns:
        Detected format type ('bbox' or 'obb')

    Note:
        If label_line is provided, it will be used instead of reading from file.
        label_file parameter is kept for backward compatibility.
    """
    # Prefer label_line if provided
    if label_line is not None:
        parts = label_line.strip().split()
        if len(parts) == 5:
            return "bbox"
        elif len(parts) == 9:
            return "obb"
        else:
            # Default to bbox if unclear
            return "bbox"

    # Backward compatibility: read from file
    if label_file is None:
        raise ValueError("Either label_file or label_line must be provided")

    label_path = Path(label_file)

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {str(label_path)}")

    # Read first non-empty line
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) == 5:
                    return "bbox"
                elif len(parts) == 9:
                    return "obb"
                else:
                    # Default to bbox if unclear
                    return "bbox"

    # Empty file - default to bbox
    return "bbox"


def convert_bbox_to_obb(bbox: BBoxFormat) -> OBBFormat:
    """
    Convert regular bounding box to OBB format

    Args:
        bbox: BBox format object

    Returns:
        OBB format with 4 corner points

    Note:
        Creates axis-aligned rectangle from bbox
    """
    x_center, y_center = bbox.x_center, bbox.y_center
    width, height = bbox.width, bbox.height

    # Calculate 4 corners
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2

    # Create points in clockwise order: top-left, top-right, bottom-right, bottom-left
    points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    return OBBFormat(bbox.class_id, points)


def convert_obb_to_bbox(obb: OBBFormat) -> BBoxFormat:
    """
    Convert OBB format to regular bounding box

    Args:
        obb: OBB format object

    Returns:
        BBox format (axis-aligned bounding box)

    Note:
        Creates minimum axis-aligned bounding box that contains all points
    """
    points = obb.points

    # Find min/max coordinates
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()

    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return BBoxFormat(obb.class_id, x_center, y_center, width, height)


def normalize_coordinates(
    coords: Union[List[float], np.ndarray], image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Normalize absolute coordinates to 0-1 range

    Args:
        coords: Absolute coordinates (can be flat list or array)
        image_size: Image dimensions (width, height)

    Returns:
        Normalized coordinates as numpy array
    """
    coords = np.array(coords)
    width, height = image_size

    # Handle both flat and structured coordinates
    if len(coords.shape) == 1:
        # Flat array: [x1, y1, x2, y2, ...]
        normalized = coords.copy()
        normalized[0::2] /= width  # x coordinates
        normalized[1::2] /= height  # y coordinates
    else:
        # Structured array: [[x1, y1], [x2, y2], ...]
        normalized = coords.copy()
        normalized[:, 0] /= width
        normalized[:, 1] /= height

    return normalized


def denormalize_coordinates(
    coords: Union[List[float], np.ndarray], image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Convert normalized coordinates (0-1) to absolute pixel coordinates

    Args:
        coords: Normalized coordinates
        image_size: Image dimensions (width, height)

    Returns:
        Absolute coordinates as numpy array
    """
    coords = np.array(coords)
    width, height = image_size

    # Handle both flat and structured coordinates
    if len(coords.shape) == 1:
        denormalized = coords.copy()
        denormalized[0::2] *= width
        denormalized[1::2] *= height
    else:
        denormalized = coords.copy()
        denormalized[:, 0] *= width
        denormalized[:, 1] *= height

    return denormalized


def validate_coordinates(coords: np.ndarray) -> bool:
    """
    Validate that normalized coordinates are in valid range [0, 1]

    Args:
        coords: Normalized coordinates

    Returns:
        True if all coordinates are valid
    """
    return np.all((coords >= 0) & (coords <= 1))


def clip_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Clip coordinates to valid range [0, 1]

    Args:
        coords: Normalized coordinates

    Returns:
        Clipped coordinates
    """
    return np.clip(coords, 0, 1)

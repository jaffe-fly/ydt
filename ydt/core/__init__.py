"""
Core utilities module

Provides essential utilities for YOLO dataset processing:
- Format handling (OBB and bbox)
- Logging configuration
- Common utility functions
"""

from .formats import (
    BBoxFormat,
    FormatType,
    OBBFormat,
    convert_bbox_to_obb,
    convert_obb_to_bbox,
    denormalize_coordinates,
    detect_format,
    normalize_coordinates,
    parse_label_line,
)
from .logger import get_logger, setup_logger
from .utils import (
    count_class_distribution,
    generate_class_mapping,
    scan_empty_labels,
    update_label_classes,
)

__all__ = [
    # Format handling
    "BBoxFormat",
    "OBBFormat",
    "FormatType",
    "parse_label_line",
    "convert_bbox_to_obb",
    "convert_obb_to_bbox",
    "normalize_coordinates",
    "denormalize_coordinates",
    "detect_format",
    # Logging
    "setup_logger",
    "get_logger",
    # Utilities
    "scan_empty_labels",
    "update_label_classes",
    "generate_class_mapping",
    "count_class_distribution",
]

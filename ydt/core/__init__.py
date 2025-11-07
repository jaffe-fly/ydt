"""
Core utilities module

Provides essential utilities for YOLO dataset processing:
- Format handling (OBB and bbox)
- Logging configuration
- Common utility functions
"""

from .formats import (
    BBoxFormat,
    OBBFormat,
    FormatType,
    parse_label_line,
    convert_bbox_to_obb,
    convert_obb_to_bbox,
    normalize_coordinates,
    denormalize_coordinates,
    detect_format,
)
from .logger import setup_logger, get_logger
from .utils import (
    scan_empty_labels,
    update_label_classes,
    generate_class_mapping,
    count_class_distribution,
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

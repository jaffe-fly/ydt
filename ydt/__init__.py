"""
YDT - YOLO Dataset Tools

A comprehensive toolkit for YOLO format dataset processing, supporting both OBB
(Oriented Bounding Box) and regular bounding box formats.

Modules:
    - image: Image processing operations (augmentation, slicing, resizing, concat, etc.)
    - dataset: Dataset manipulation (splitting, merging, synthesis, auto-labeling)
    - visual: Visualization utilities
    - core: Core utilities and format handlers
    - auto_label: Automatic dataset labeling using YOLO models
    - cli: Command-line interface
"""

__version__ = "0.2.6"

# Import main classes and functions for easy access
from .image import (
    augment_dataset,
    slice_dataset,
    extract_frames,
    rotate_image_with_labels,
    concat_images_horizontally,
    concat_images_vertically,
    process_images_multi_method,
)
from .dataset import split_dataset, merge_datasets, DatasetSynthesizer
from .auto_label import auto_label_dataset
from .visual import (
    visualize_dataset,
    visualize_letterbox,
    visualize_hsv_augmentation,
)
from .core import FormatType, detect_format, OBBFormat, BBoxFormat

__all__ = [
    "__version__",
    # Image processing
    "augment_dataset",
    "slice_dataset",
    "extract_frames",
    "rotate_image_with_labels",
    "concat_images_horizontally",
    "concat_images_vertically",
    "process_images_multi_method",
    # Dataset operations
    "split_dataset",
    "merge_datasets",
    "DatasetSynthesizer",
    "auto_label_dataset",
    # Visualization
    "visualize_dataset",
    "visualize_letterbox",
    "visualize_hsv_augmentation",
    # Core
    "FormatType",
    "detect_format",
    "OBBFormat",
    "BBoxFormat",
]

"""
Image processing module

Provides image manipulation operations:
- Augmentation (rotation, HSV, etc.)
- Slicing and tiling
- Resizing and cropping
- Coordinate-based cropping
- Object cropping (from model or dataset)
- Concatenation
- Video frame extraction for dataset creation
"""

from .augment import augment_dataset, rotate_image_with_labels
from .concat import concat_images_horizontally, concat_images_vertically
from .crop import crop_from_dataset, crop_with_model
from .resize import (
    crop_directory_by_coords,
    crop_image_by_coords,
    process_images_multi_method,
    resize_dataset,
    resize_directory,
)
from .slice import slice_dataset
from .video import extract_frames, extract_frames_parallel

__all__ = [
    "rotate_image_with_labels",
    "augment_dataset",
    "slice_dataset",
    "extract_frames",
    "extract_frames_parallel",
    "crop_image_by_coords",
    "crop_directory_by_coords",
    "process_images_multi_method",
    "resize_dataset",
    "resize_directory",
    "concat_images_horizontally",
    "concat_images_vertically",
    "crop_with_model",
    "crop_from_dataset",
]

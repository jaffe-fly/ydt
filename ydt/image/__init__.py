"""
Image processing module

Provides image manipulation operations:
- Augmentation (rotation, HSV, etc.)
- Slicing and tiling
- Resizing and cropping
- Coordinate-based cropping
- Concatenation
- Video frame extraction for dataset creation
"""

from .augment import augment_dataset, rotate_image_with_labels
from .concat import concat_images_horizontally, concat_images_vertically
from .resize import crop_directory_by_coords, crop_image_by_coords, process_images_multi_method
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
    "concat_images_horizontally",
    "concat_images_vertically",
]

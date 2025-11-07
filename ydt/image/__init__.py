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

from .augment import rotate_image_with_labels, augment_dataset
from .slice import slice_dataset
from .video import extract_frames, extract_frames_parallel
from .resize import crop_image_by_coords, crop_directory_by_coords, process_images_multi_method
from .concat import concat_images_horizontally, concat_images_vertically

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

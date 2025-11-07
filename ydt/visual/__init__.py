"""
Visualization module

Provides visualization utilities:
- Dataset annotation visualization with interactive navigation
- Letterbox effect preview for YOLO preprocessing
- HSV and other augmentation visualizations
- Side-by-side comparison tools
"""

from .dataset import draw_obb, visualize_dataset
from .letter import (
    extract_roi_with_padding,
    resize_window_to_screen,
    visualize_letterbox,
    visualize_regions_letterbox,
)
from .augment import (
    show_side_by_side,
    visualize_albumentations,
    visualize_hsv_augmentation,
    visualize_multiple_augmentations,
)

__all__ = [
    # Dataset visualization
    "draw_obb",
    "visualize_dataset",
    # Letterbox visualization
    "extract_roi_with_padding",
    "resize_window_to_screen",
    "visualize_letterbox",
    "visualize_regions_letterbox",
    # Augmentation visualization
    "show_side_by_side",
    "visualize_albumentations",
    "visualize_hsv_augmentation",
    "visualize_multiple_augmentations",
]

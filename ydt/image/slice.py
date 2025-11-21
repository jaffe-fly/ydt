"""
Image slicing operations for dataset preparation.

This module provides functions for slicing large images into smaller tiles,
particularly useful for object detection tasks with oriented bounding boxes (OBB).
"""

import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from sahi.slicing import slice_image

from ydt.core import IMAGE_EXTENSIONS
from ydt.core.formats import detect_format
from ydt.core.logger import get_logger

logger = get_logger(__name__)


def convert_label_coordinates(
    label_line: str,
    original_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
    new_size: tuple[int, int],
    format_type: str | None = None,
) -> str | None:
    """
    Convert label coordinates to fit cropped and scaled images.

    Args:
        label_line: Original label line
        original_size: Original image size (width, height)
        crop_box: Crop region (x1, y1, x2, y2)
        new_size: New image size (width, height)
        format_type: Label format type ('obb' or 'bbox'). If None, auto-detect

    Returns:
        Converted label line, or None if annotation points are not in crop region

    Raises:
        ValueError: If label format is invalid
    """
    try:
        parts = label_line.strip().split()

        # Auto-detect format if not specified
        if format_type is None:
            format_type = detect_format(label_line=label_line)

        if format_type == "obb":
            if len(parts) != 9:  # class_id + 4 points (x,y)
                return None

            class_id = parts[0]
            points = np.array([float(x) for x in parts[1:]])

            # Convert normalized coordinates to actual coordinates
            points[::2] *= original_size[0]  # x coordinates
            points[1::2] *= original_size[1]  # y coordinates

            # Adjust coordinates to crop region
            points[::2] -= crop_box[0]  # x coordinates minus x1
            points[1::2] -= crop_box[1]  # y coordinates minus y1

            # Check if all points are within crop region
            crop_width = crop_box[2] - crop_box[0]
            crop_height = crop_box[3] - crop_box[1]

            if not all(
                (0 <= points[::2])
                & (points[::2] <= crop_width)
                & (0 <= points[1::2])
                & (points[1::2] <= crop_height)
            ):
                return None

            # Normalize coordinates to new image size
            points[::2] /= crop_width
            points[1::2] /= crop_height

            # Ensure all coordinates are in [0,1] range
            if not all((0 <= points) & (points <= 1)):
                return None

            # Convert back to string format
            return " ".join([class_id] + [f"{x:.6f}" for x in points]) + "\n"

        elif format_type == "bbox":
            # Handle bbox format (class_id, x_center, y_center, width, height)
            if len(parts) != 5:
                return None

            class_id = parts[0]
            x_center, y_center, width, height = [float(x) for x in parts[1:]]

            # Convert normalized to actual coordinates
            abs_x_center = x_center * original_size[0]
            abs_y_center = y_center * original_size[1]
            abs_width = width * original_size[0]
            abs_height = height * original_size[1]

            # Calculate bbox corners
            x1 = abs_x_center - abs_width / 2
            y1 = abs_y_center - abs_height / 2
            x2 = abs_x_center + abs_width / 2
            y2 = abs_y_center + abs_height / 2

            # Adjust to crop region
            x1 -= crop_box[0]
            y1 -= crop_box[1]
            x2 -= crop_box[0]
            y2 -= crop_box[1]

            crop_width = crop_box[2] - crop_box[0]
            crop_height = crop_box[3] - crop_box[1]

            # Check if bbox is within crop region
            if x1 >= crop_width or y1 >= crop_height or x2 <= 0 or y2 <= 0:
                return None

            # Clip to crop region
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(crop_width, x2)
            y2 = min(crop_height, y2)

            # Recalculate center and dimensions
            new_x_center = (x1 + x2) / 2 / crop_width
            new_y_center = (y1 + y2) / 2 / crop_height
            new_width = (x2 - x1) / crop_width
            new_height = (y2 - y1) / crop_height

            return f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n"

        else:
            logger.warning(f"Unknown format type: {format_type}")
            return None

    except Exception as e:
        logger.error(f"Error processing label: {e}")
        return None


def slice_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    horizontal_count: int = 3,
    vertical_count: int | None = None,
    overlap_ratio_horizontal: float = 0.1,
    overlap_ratio_vertical: float = 0.0,
    crop_x1: int | None = None,
    crop_x2: int | None = None,
    crop_y1: int | None = None,
    crop_y2: int | None = None,
    test_mode: bool = False,
    test_count: int = 2,
    format_type: str | None = None,
) -> dict[str, int]:
    """
    Slice images using SAHI and process corresponding label files.

    Args:
        input_dir: Input image file or directory containing images
        output_dir: Output directory
        horizontal_count: Number of horizontal slices
        vertical_count: Number of vertical slices (optional, if None only horizontal slicing)
        overlap_ratio_horizontal: Overlap ratio between horizontal slices
        overlap_ratio_vertical: Overlap ratio between vertical slices
        crop_x1: Optional x-axis crop start coordinate
        crop_x2: Optional x-axis crop end coordinate
        crop_y1: Optional y-axis crop start coordinate
        crop_y2: Optional y-axis crop end coordinate
        test_mode: Whether to run in test mode (process limited images)
        test_count: Number of images to process in test mode
        format_type: Label format type ('obb' or 'bbox'). If None, auto-detect

    Returns:
        Dictionary with statistics (processed_files, total_slices, etc.)

    Raises:
        FileNotFoundError: If input path doesn't exist
        ValueError: If input file is not a supported image format
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input path not found: {input_dir}")

    if test_mode:
        logger.info(f"Test mode: processing only {test_count} images")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files (including subdirectories)
    image_files = []
    has_subfolders = False

    # Check if input is a single file or directory
    if input_dir.is_file():
        # Single file mode
        if input_dir.suffix not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {input_dir.suffix}. Supported formats: {IMAGE_EXTENSIONS}"
            )

        logger.info(f"Processing single image file: {input_dir.name}")
        image_files = [input_dir]
        search_dir = input_dir.parent
        has_subfolders = False
    else:
        # Directory mode
        # First check if there's an images subdirectory
        images_dir = input_dir / "images"

        if images_dir.exists():
            logger.info(f"Found images subdirectory, reading from {images_dir}")
            search_dir = images_dir
            has_subfolders = True
        else:
            logger.info(f"No images subdirectory found, reading from {input_dir}")
            search_dir = input_dir

        # Recursively search for all image files
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(list(search_dir.rglob(f"*{ext}")))

        # In test mode, only process specified number of images
        if test_mode:
            image_files = image_files[:test_count]

    image_files = set(image_files)
    total_files = len(image_files)
    logger.info(f"Found {total_files} image file(s)")

    processed_files = 0
    total_slices = 0

    # If has subfolders, collect all sliced image paths
    train_images: set[str] = set()
    val_images: set[str] = set()

    for idx, img_path in enumerate(image_files, 1):
        try:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                logger.error(f"Failed to read image: {img_path}")
                continue

            logger.info(f"Processing image: {img_path.name}")
            original_size = (image.shape[1], image.shape[0])  # (width, height)

            # If crop region specified, crop first
            if all(x is not None for x in [crop_x1, crop_x2, crop_y1, crop_y2]):
                image = image[crop_y1:crop_y2, crop_x1:crop_x2]
                logger.info(f"Crop region: x={crop_x1} to x={crop_x2}, y={crop_y1} to y={crop_y2}")
                logger.info(f"Cropped size: {image.shape}")
                crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
            else:
                crop_box = (0, 0, original_size[0], original_size[1])

            # Calculate slice parameters
            crop_width = crop_box[2] - crop_box[0]
            crop_height = crop_box[3] - crop_box[1]

            # Determine slicing mode
            if vertical_count is None:
                # Horizontal slicing only
                logger.info(f"Horizontal slicing: {horizontal_count} slices")
                # Formula: slice_size = total_size / (count - overlap_ratio * (count - 1))
                # Use ceil to ensure complete coverage and exact slice count
                slice_width = math.ceil(
                    crop_width
                    / (horizontal_count - overlap_ratio_horizontal * (horizontal_count - 1))
                )
                slice_height = crop_height

                # Use SAHI for slicing
                slice_result = slice_image(
                    image=image,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=0.0,
                    overlap_width_ratio=overlap_ratio_horizontal,
                )
            else:
                # Grid slicing (both horizontal and vertical)
                logger.info(
                    f"Grid slicing: {horizontal_count} × {vertical_count} = {horizontal_count * vertical_count} slices"
                )

                # Calculate slice dimensions to achieve desired grid count
                # Formula: slice_size = total_size / (count - overlap_ratio * (count - 1))
                # Use ceil to ensure complete coverage and exact slice count
                slice_width = math.ceil(
                    crop_width
                    / (horizontal_count - overlap_ratio_horizontal * (horizontal_count - 1))
                )
                slice_height = math.ceil(
                    crop_height / (vertical_count - overlap_ratio_vertical * (vertical_count - 1))
                )

                # Use SAHI for grid slicing
                slice_result = slice_image(
                    image=image,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_ratio_vertical,
                    overlap_width_ratio=overlap_ratio_horizontal,
                )

            logger.info(f"Image {img_path.name} will be sliced into {len(slice_result)} slices")

            # Get relative path from search directory
            rel_path = img_path.relative_to(search_dir)

            # Process each slice
            for slice_number, slice_data in enumerate(slice_result):
                slice_img = slice_data["image"]

                # Maintain directory structure
                if has_subfolders:
                    # If original has images subdirectory, create same structure in output
                    rel_output_dir = output_dir / "images" / rel_path.parent
                else:
                    # Otherwise use relative path directly
                    rel_output_dir = output_dir / rel_path.parent
                rel_output_dir.mkdir(parents=True, exist_ok=True)

                # Calculate grid position for naming
                if vertical_count is not None:
                    row = slice_number // horizontal_count
                    col = slice_number % horizontal_count
                    slice_name = f"{img_path.stem}_slice_{row:02d}_{col:02d}{img_path.suffix}"
                else:
                    slice_name = f"{img_path.stem}_slice_{slice_number:03d}{img_path.suffix}"
                slice_image_path = rel_output_dir / slice_name

                # Save slice image
                cv2.imwrite(str(slice_image_path), slice_img)

                # If has subfolder structure, process label files
                if has_subfolders:
                    # Check if original label file exists
                    if images_dir.exists():
                        # If has images directory, label should be in labels directory
                        label_path = input_dir / "labels" / rel_path.parent / f"{img_path.stem}.txt"
                    else:
                        # Otherwise look for labels in same level
                        label_path = img_path.parent / "labels" / f"{img_path.stem}.txt"

                    # Create label output directory (maintain same structure)
                    labels_dir = output_dir / "labels" / rel_path.parent
                    labels_dir.mkdir(parents=True, exist_ok=True)

                    # Calculate current slice crop box
                    slice_box = slice_data["starting_pixel"]
                    x1 = slice_box[0]
                    y1 = slice_box[1]
                    x2 = x1 + slice_img.shape[1]
                    y2 = y1 + slice_img.shape[0]

                    # Create new label file
                    if vertical_count is not None:
                        row = slice_number // horizontal_count
                        col = slice_number % horizontal_count
                        label_save_path = (
                            labels_dir / f"{img_path.stem}_slice_{row:02d}_{col:02d}.txt"
                        )
                    else:
                        label_save_path = (
                            labels_dir / f"{img_path.stem}_slice_{slice_number:03d}.txt"
                        )

                    if label_path.exists():
                        # If original label file exists, convert coordinates
                        with open(label_path) as f:
                            original_labels = f.readlines()

                        # Convert label coordinates
                        new_labels = []
                        for label in original_labels:
                            new_label = convert_label_coordinates(
                                label,
                                original_size,
                                (
                                    x1 + crop_box[0],
                                    y1 + crop_box[1],
                                    x2 + crop_box[0],
                                    y2 + crop_box[1],
                                ),
                                (slice_img.shape[1], slice_img.shape[0]),
                                format_type=format_type,
                            )
                            if new_label:
                                new_labels.append(new_label)

                        # Save converted labels
                        with open(label_save_path, "w") as f:
                            f.writelines(new_labels)
                    else:
                        # If no original label file, create empty file
                        with open(label_save_path, "w") as f:
                            pass

                    # Collect image paths (relative to output directory)
                    rel_img_path = os.path.relpath(slice_image_path, output_dir)
                    rel_img_path = rel_img_path.replace("\\", "/")  # Use forward slashes

                    # Determine if train or val based on directory structure
                    if "train" in str(rel_path):
                        train_images.add(rel_img_path)
                    elif "val" in str(rel_path):
                        val_images.add(rel_img_path)
                    else:
                        train_images.add(rel_img_path)  # Default to train

            total_slices += len(slice_result)
            processed_files += 1

            if idx % 10 == 0:
                logger.info(
                    f"Processed: {idx}/{total_files} files, generated slices: {total_slices}"
                )

        except Exception as e:
            logger.error(f"Error processing file {img_path}: {e}")
            continue

    # If has subfolder structure, create data.yaml and path files
    if has_subfolders:
        # If original data.yaml exists, copy it
        src_yaml = input_dir / "data.yaml"
        dst_yaml = output_dir / "data.yaml"
        if src_yaml.exists():
            shutil.copy2(src_yaml, dst_yaml)
            logger.info("Copied data.yaml to output directory")

        # Save train and val image paths
        if train_images:
            train_list = sorted(train_images)
            with open(output_dir / "train.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(train_list))
            logger.info(f"Generated train.txt with {len(train_images)} image paths")

        if val_images:
            val_list = sorted(val_images)
            with open(output_dir / "val.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(val_list))
            logger.info(f"Generated val.txt with {len(val_images)} image paths")

    logger.info("All processing complete!")
    logger.info(f"Processed files: {processed_files}")
    logger.info(f"Generated slices: {total_slices}")

    return {
        "processed_files": processed_files,
        "total_slices": total_slices,
        "train_images": len(train_images),
        "val_images": len(val_images),
    }

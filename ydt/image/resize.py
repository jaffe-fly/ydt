"""
Image resizing and cropping operations.

This module provides functions for resizing and cropping images with various methods.
"""

from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

from ydt.core.logger import get_logger

logger = get_logger(__name__)


def resize_image(image: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
    """
    Resize image by scale factor while maintaining aspect ratio.

    Args:
        image: Input image
        scale_factor: Scale factor (0-1), e.g. 0.5 means half the original size

    Returns:
        Resized image

    Raises:
        ValueError: If scale_factor is not in valid range
    """
    if not 0 < scale_factor <= 1.0:
        raise ValueError(f"Scale factor must be between 0 and 1, got {scale_factor}")

    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"Resized dimensions ({new_w}x{new_h}) must be positive")

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop_image(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Crop the center part of the image to reach target size.

    Args:
        image: Input image
        target_width: Desired width
        target_height: Desired height

    Returns:
        Cropped image

    Raises:
        ValueError: If target dimensions are invalid
    """
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Target dimensions must be positive, got {target_width}x{target_height}")

    h, w = image.shape[:2]

    # Calculate crop dimensions
    start_x = (w - target_width) // 2
    start_y = (h - target_height) // 2

    # Ensure non-negative starting points
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    # Perform the crop
    cropped = image[start_y : start_y + target_height, start_x : start_x + target_width]

    # If the image is smaller than target size in any dimension,
    # resize it to match the target size
    if cropped.shape[0] != target_height or cropped.shape[1] != target_width:
        cropped = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)

    return cropped


def process_single_image_multi_method(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    target_sizes: List[int],
    use_crop: bool = False,
) -> int:
    """
    Process a single image using either scaling or cropping method.

    Args:
        input_file: Path to input image file
        output_dir: Path to save resized images
        target_sizes: List of target widths (heights will be proportionally calculated for scaling)
        use_crop: If True, use center cropping; if False, use scaling

    Returns:
        Number of images successfully processed

    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Read image
    img = cv2.imread(str(input_path))
    if img is None:
        logger.error(f"Failed to read image: {input_path}")
        return 0

    # Get filename without extension
    filename = input_path.stem
    extension = input_path.suffix

    # Get original dimensions
    orig_h, orig_w = img.shape[:2]
    aspect_ratio = orig_h / orig_w

    processed_count = 0

    # Process each target size
    for target_w in target_sizes:
        target_h = int(target_w * aspect_ratio)

        if use_crop:
            # Use center cropping method
            processed_img = center_crop_image(img, target_w, target_h)
            method = "crop"
        else:
            # Use scaling method
            scale_factor = target_w / orig_w
            processed_img = resize_image(img, scale_factor)
            method = "scale"

        # Save processed image
        output_file = output_path / f"{filename}_{method}_{target_w}x{target_h}{extension}"
        cv2.imwrite(str(output_file), processed_img)
        processed_count += 1

        logger.info(
            f"Saved image with size {target_w}x{target_h} using {method} method: {output_file}"
        )

    return processed_count


def process_images_multi_method(
    input_path: Union[str, Path], output_dir: Union[str, Path], target_sizes: List[int]
) -> Tuple[int, int]:
    """
    Process images using both scaling and cropping methods.

    Args:
        input_path: Path to input image file or directory
        output_dir: Path to save processed images
        target_sizes: List of target widths

    Returns:
        Tuple of (total_processed, total_failed)

    Raises:
        FileNotFoundError: If input path doesn't exist
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    total_processed = 0
    total_failed = 0

    # If input is a file, process single image
    if input_path.is_file():
        # Process with scaling
        count = process_single_image_multi_method(
            input_path, output_dir, target_sizes, use_crop=False
        )
        total_processed += count
        # Process with cropping
        count = process_single_image_multi_method(
            input_path, output_dir, target_sizes, use_crop=True
        )
        total_processed += count
        return total_processed, total_failed

    # If input is a directory, process all images
    img_extensions = (".jpg", ".jpeg", ".png")

    processed_count = 0
    for img_file in input_path.glob("*"):
        if img_file.suffix.lower() in img_extensions:
            try:
                # Process with scaling
                count = process_single_image_multi_method(
                    img_file, output_dir, target_sizes, use_crop=False
                )
                total_processed += count
                # Process with cropping
                count = process_single_image_multi_method(
                    img_file, output_dir, target_sizes, use_crop=True
                )
                total_processed += count

                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} images...")
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                total_failed += 1

    logger.info(
        f"Finished processing {processed_count} images using both scaling and cropping methods"
    )
    return total_processed, total_failed


def crop_image_by_coords(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Crop image based on specified coordinates.

    Args:
        image: Input image as numpy array
        x1: Left coordinate (top-left corner x)
        y1: Top coordinate (top-left corner y)
        x2: Right coordinate (bottom-right corner x)
        y2: Bottom coordinate (bottom-right corner y)

    Returns:
        Cropped image as numpy array

    Raises:
        ValueError: If coordinates are invalid or out of image bounds
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None")

    # Get image dimensions
    height, width = image.shape[:2]

    # Validate coordinates
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            f"Crop coordinates ({x1},{y1},{x2},{y2}) are outside image bounds ({width}x{height})"
        )

    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"Invalid crop region: x1 ({x1}) must be < x2 ({x2}) and y1 ({y1}) must be < y2 ({y2})"
        )

    # Crop the image
    cropped = image[y1:y2, x1:x2]

    logger.debug(f"Cropped image from {width}x{height} to {(x2 - x1)}x{(y2 - y1)}")
    return cropped


def crop_directory_by_coords(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    recursive: bool = True,
) -> Tuple[int, int]:
    """
    Crop all images in a directory based on specified coordinates.

    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for cropped images
        x1: Left coordinate (top-left corner x)
        y1: Top coordinate (top-left corner y)
        x2: Right coordinate (bottom-right corner x)
        y2: Bottom coordinate (bottom-right corner y)
        recursive: Whether to search subdirectories recursively

    Returns:
        Tuple of (success_count, failure_count)

    Raises:
        FileNotFoundError: If input directory doesn't exist
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Supported image formats
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".webp",
        ".JPG",
        ".JPEG",
        ".PNG",
        ".BMP",
        ".TIFF",
        ".WEBP",
    )

    # Get all image files
    if recursive:
        image_files = [f for f in input_path.rglob("*") if f.suffix in image_extensions]
    else:
        image_files = [f for f in input_path.glob("*") if f.suffix in image_extensions]

    total_images = len(image_files)

    if total_images == 0:
        logger.warning(f"No image files found in {input_path}")
        return 0, 0

    logger.info(f"Found {total_images} images to process")
    logger.info(f"Crop region: ({x1}, {y1}) -> ({x2}, {y2})")

    success_count = 0
    failure_count = 0

    # Process each image
    for i, image_file in enumerate(image_files, 1):
        # Calculate relative path to maintain directory structure
        rel_path = image_file.relative_to(input_path)
        output_file = output_path / rel_path

        logger.info(f"Processing [{i}/{total_images}] {rel_path}")

        try:
            # Read image
            img = cv2.imread(str(image_file))
            if img is None:
                logger.error(f"Failed to read image: {image_file}")
                failure_count += 1
                continue

            # Validate crop coordinates for this specific image
            height, width = img.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                logger.warning(
                    f"Skip {image_file.name}: crop region outside image bounds ({width}x{height})"
                )
                failure_count += 1
                continue

            # Crop image
            cropped = crop_image_by_coords(img, x1, y1, x2, y2)

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save cropped image
            cv2.imwrite(str(output_file), cropped)
            success_count += 1

            logger.debug(f"Saved cropped image: {output_file}")

        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            failure_count += 1

    logger.info(f"Processing complete. Success: {success_count}, Failed: {failure_count}")
    return success_count, failure_count

"""
Image resizing and cropping operations.

This module provides functions for resizing and cropping images with various methods.
"""

from pathlib import Path

import cv2
import numpy as np

from ydt.core import IMAGE_EXTENSIONS
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
    input_file: str | Path,
    output_dir: str | Path,
    target_sizes: list[int],
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
    input_path: str | Path, output_dir: str | Path, target_sizes: list[int]
) -> tuple[int, int]:
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

    processed_count = 0
    for img_file in input_path.glob("*"):
        if img_file.suffix.lower() in IMAGE_EXTENSIONS:
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
    input_dir: str | Path,
    output_dir: str | Path,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    recursive: bool = True,
) -> tuple[int, int]:
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

    # Get all image files
    if recursive:
        image_files = [f for f in input_path.rglob("*") if f.suffix in IMAGE_EXTENSIONS]
    else:
        image_files = [f for f in input_path.glob("*") if f.suffix in IMAGE_EXTENSIONS]

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


def resize_directory(
    input_path: str | Path,
    output_dir: str | Path,
    target_size: int,
    resize_mode: str = "longest",
    interpolation: str = "linear",
    resize_all: bool = False,
    recursive: bool = True,
) -> tuple[int, int]:
    """
    Resize all images in a directory or a single image file.

    Args:
        input_path: Input directory containing images or single image file
        output_dir: Output directory for resized images
        target_size: Target size in pixels
        resize_mode: Which edge to resize - 'longest', 'shortest', 'width', or 'height' (default: 'longest')
        interpolation: Interpolation method ('linear' or 'lanczos4')
        resize_all: If True, resize all images (including those larger than target).
                    If False, only upscale images smaller than target size
        recursive: Whether to search subdirectories recursively (only for directory input)

    Returns:
        Tuple of (success_count, failure_count)

    Raises:
        FileNotFoundError: If input path doesn't exist
        ValueError: If resize_mode is invalid
    """
    if resize_mode not in ["longest", "shortest", "width", "height"]:
        raise ValueError(
            f"Invalid resize_mode '{resize_mode}'. Must be 'longest', 'shortest', 'width', or 'height'"
        )

    # Map interpolation method to OpenCV constant
    interp_map = {
        "linear": cv2.INTER_LINEAR,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    interp_flag = interp_map.get(interpolation, cv2.INTER_LINEAR)

    input_path = Path(input_path)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Handle single file input
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            logger.error(f"Input file is not a supported image format: {input_path}")
            return 0, 1

        logger.info(f"Processing single image: {input_path.name}")
        logger.info(
            f"Resize mode: {resize_mode}, target size: {target_size}, interpolation: {interpolation}"
        )

        try:
            # Read image
            img = cv2.imread(str(input_path))
            if img is None:
                logger.error(f"Failed to read image: {input_path}")
                return 0, 1

            # Get original dimensions
            orig_h, orig_w = img.shape[:2]

            # Determine which edge to compare based on resize_mode
            if resize_mode == "longest":
                current_edge = max(orig_w, orig_h)
            elif resize_mode == "shortest":
                current_edge = min(orig_w, orig_h)
            elif resize_mode == "width":
                current_edge = orig_w
            else:  # height
                current_edge = orig_h

            # Decide whether to resize
            should_resize = False
            if current_edge < target_size:
                # Always upscale if smaller than target
                should_resize = True
            elif resize_all and current_edge > target_size:
                # Only downscale if resize_all is True
                should_resize = True

            if not should_resize:
                # Copy without resizing
                output_file = output_path / input_path.name
                cv2.imwrite(str(output_file), img)
                logger.info(f"Copied without resizing: {orig_w}x{orig_h}")
                logger.info(f"Saved: {output_file}")
                return 1, 0

            # Calculate new dimensions based on resize_mode
            if resize_mode == "longest":
                if orig_w >= orig_h:
                    scale_factor = target_size / orig_w
                else:
                    scale_factor = target_size / orig_h
            elif resize_mode == "shortest":
                if orig_w <= orig_h:
                    scale_factor = target_size / orig_w
                else:
                    scale_factor = target_size / orig_h
            elif resize_mode == "width":
                scale_factor = target_size / orig_w
            else:  # height
                scale_factor = target_size / orig_h

            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp_flag)

            # Save resized image with same name
            output_file = output_path / input_path.name
            cv2.imwrite(str(output_file), resized)

            logger.info(f"Resized from {orig_w}x{orig_h} to {new_w}x{new_h}")
            logger.info(f"Saved: {output_file}")
            return 1, 0

        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return 0, 1

    # Handle directory input
    # Get all image files
    if recursive:
        image_files = [f for f in input_path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        image_files = [f for f in input_path.glob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]

    total_images = len(image_files)

    if total_images == 0:
        logger.warning(f"No image files found in {input_path}")
        return 0, 0

    logger.info(f"Found {total_images} images to process")
    logger.info(
        f"Resize mode: {resize_mode}, target size: {target_size}, interpolation: {interpolation}"
    )
    logger.info(f"Resize all: {resize_all}")

    success_count = 0
    failure_count = 0

    # Process each image
    for i, image_file in enumerate(image_files, 1):
        # Calculate relative path to maintain directory structure
        rel_path = image_file.relative_to(input_path)
        output_file = output_path / rel_path

        if i % 100 == 0:
            logger.info(f"Processing [{i}/{total_images}]...")

        try:
            # Read image
            img = cv2.imread(str(image_file))
            if img is None:
                logger.error(f"Failed to read image: {image_file}")
                failure_count += 1
                continue

            # Get original dimensions
            orig_h, orig_w = img.shape[:2]

            # Determine which edge to compare based on resize_mode
            if resize_mode == "longest":
                current_edge = max(orig_w, orig_h)
            elif resize_mode == "shortest":
                current_edge = min(orig_w, orig_h)
            elif resize_mode == "width":
                current_edge = orig_w
            else:  # height
                current_edge = orig_h

            # Decide whether to resize
            should_resize = False
            if current_edge < target_size:
                # Always upscale if smaller than target
                should_resize = True
            elif resize_all and current_edge > target_size:
                # Only downscale if resize_all is True
                should_resize = True

            if not should_resize:
                # Copy without resizing
                output_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_file), img)
                success_count += 1
                logger.debug(f"Copied without resizing: {output_file}")
                continue

            # Calculate new dimensions based on resize_mode
            if resize_mode == "longest":
                if orig_w >= orig_h:
                    scale_factor = target_size / orig_w
                else:
                    scale_factor = target_size / orig_h
            elif resize_mode == "shortest":
                if orig_w <= orig_h:
                    scale_factor = target_size / orig_w
                else:
                    scale_factor = target_size / orig_h
            elif resize_mode == "width":
                scale_factor = target_size / orig_w
            else:  # height
                scale_factor = target_size / orig_h

            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
            resized = cv2.resize(img, (new_w, new_h), interpolation=interp_flag)

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save resized image
            cv2.imwrite(str(output_file), resized)
            success_count += 1

            logger.debug(f"Saved resized image: {output_file}")

        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            failure_count += 1

    logger.info(f"Processing complete. Success: {success_count}, Failed: {failure_count}")
    return success_count, failure_count


def resize_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    target_size: int = 640,
    resize_mode: str = "longest",
    interpolation: str = "linear",
    resize_all: bool = False,
    split: str = "both",
) -> dict[str, int]:
    """
    Resize YOLO dataset images and adjust corresponding label coordinates.

    Args:
        dataset_dir: Input YOLO dataset directory containing data.yaml
        output_dir: Output directory for resized dataset
        target_size: Target size in pixels
        resize_mode: Which edge to resize - 'longest', 'shortest', 'width', or 'height' (default: 'longest')
        interpolation: Interpolation method ('linear' or 'lanczos4')
        resize_all: If True, resize all images (including those larger than target).
                    If False, only upscale images smaller than target size
        split: Which split to process - 'train', 'val', or 'both' (default: 'both')

    Returns:
        Dictionary with statistics (resized_count, copied_count, failed_count)

    Raises:
        FileNotFoundError: If dataset directory or data.yaml not found
        ValueError: If resize_mode or split is invalid

    Examples:
        >>> # Resize only small images in both splits
        >>> stats = resize_dataset("./dataset", "./resized", target_size=640)

        >>> # Resize all images (including large ones) in train split
        >>> stats = resize_dataset(
        ...     "./dataset",
        ...     "./resized",
        ...     target_size=640,
        ...     resize_all=True,
        ...     split="train"
        ... )
    """
    import yaml

    if resize_mode not in ["longest", "shortest", "width", "height"]:
        raise ValueError(
            f"Invalid resize_mode '{resize_mode}'. Must be 'longest', 'shortest', 'width', or 'height'"
        )

    valid_splits = ["train", "val", "both"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")

    # Map interpolation method to OpenCV constant
    interp_map = {
        "linear": cv2.INTER_LINEAR,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    interp_flag = interp_map.get(interpolation, cv2.INTER_LINEAR)

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Read data.yaml
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    # Determine which splits to process
    splits_to_process = []
    if split == "both":
        splits_to_process = ["train", "val"]
    else:
        splits_to_process = [split]

    # Create output directories
    for s in splits_to_process:
        (output_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    logger.info(f"Resizing dataset from {dataset_dir}")
    logger.info(f"Processing splits: {splits_to_process}")
    logger.info(
        f"Resize mode: {resize_mode}, target size: {target_size}, interpolation: {interpolation}"
    )
    logger.info(f"Resize all: {resize_all}")

    # Process each split
    total_resized = 0
    total_copied = 0
    total_failed = 0

    for current_split in splits_to_process:
        src_img_dir = dataset_dir / "images" / current_split
        src_label_dir = dataset_dir / "labels" / current_split
        dst_img_dir = output_dir / "images" / current_split
        dst_label_dir = output_dir / "labels" / current_split

        if not src_img_dir.exists():
            logger.warning(f"Image directory not found: {src_img_dir}")
            continue

        # Get all image files
        image_files = [f for f in src_img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

        if len(image_files) == 0:
            logger.warning(f"No images found in {src_img_dir}")
            continue

        logger.info(f"\nProcessing {current_split} split: {len(image_files)} images")

        split_resized = 0
        split_copied = 0
        split_failed = 0

        for i, img_file in enumerate(image_files, 1):
            if i % 100 == 0:
                logger.info(f"  [{i}/{len(image_files)}]...")

            try:
                # Read image
                img = cv2.imread(str(img_file))
                if img is None:
                    logger.error(f"Failed to read image: {img_file}")
                    split_failed += 1
                    continue

                # Get original dimensions
                orig_h, orig_w = img.shape[:2]

                # Determine which edge to compare based on resize_mode
                if resize_mode == "longest":
                    current_edge = max(orig_w, orig_h)
                elif resize_mode == "shortest":
                    current_edge = min(orig_w, orig_h)
                elif resize_mode == "width":
                    current_edge = orig_w
                else:  # height
                    current_edge = orig_h

                # Decide whether to resize
                should_resize = False
                if current_edge < target_size:
                    # Always upscale if smaller than target
                    should_resize = True
                elif resize_all and current_edge > target_size:
                    # Only downscale if resize_all is True
                    should_resize = True

                # Calculate scale factor if needed
                if should_resize:
                    # Calculate new dimensions based on resize_mode
                    if resize_mode == "longest":
                        if orig_w >= orig_h:
                            scale_factor = target_size / orig_w
                        else:
                            scale_factor = target_size / orig_h
                    elif resize_mode == "shortest":
                        if orig_w <= orig_h:
                            scale_factor = target_size / orig_w
                        else:
                            scale_factor = target_size / orig_h
                    elif resize_mode == "width":
                        scale_factor = target_size / orig_w
                    else:  # height
                        scale_factor = target_size / orig_h

                    new_w = int(orig_w * scale_factor)
                    new_h = int(orig_h * scale_factor)
                    resized_img = cv2.resize(img, (new_w, new_h), interpolation=interp_flag)
                    split_resized += 1
                else:
                    # No resize needed
                    resized_img = img
                    new_w, new_h = orig_w, orig_h
                    split_copied += 1
                    scale_factor = 1.0

                # Save resized image
                dst_img_file = dst_img_dir / img_file.name
                cv2.imwrite(str(dst_img_file), resized_img)

                # Process corresponding label file
                label_file = src_label_dir / f"{img_file.stem}.txt"
                dst_label_file = dst_label_dir / f"{img_file.stem}.txt"

                if label_file.exists():
                    # Read and adjust label coordinates
                    with open(label_file, encoding="utf-8") as f:
                        lines = f.readlines()

                    # Labels remain the same (normalized coordinates don't change)
                    # Just copy the label file
                    with open(dst_label_file, "w", encoding="utf-8") as f:
                        f.writelines(lines)

            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                split_failed += 1

        logger.info(
            f"  {current_split} complete: {split_resized} resized, {split_copied} copied, {split_failed} failed"
        )
        total_resized += split_resized
        total_copied += split_copied
        total_failed += split_failed

    # Copy data.yaml to output directory
    dst_yaml = output_dir / "data.yaml"
    with open(dst_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    logger.info("\nDataset resize complete!")
    logger.info(f"Total: {total_resized} resized, {total_copied} copied, {total_failed} failed")

    return {
        "resized_count": total_resized,
        "copied_count": total_copied,
        "failed_count": total_failed,
    }

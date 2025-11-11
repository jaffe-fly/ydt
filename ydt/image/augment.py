"""
Image augmentation for YOLO datasets.

Provides rotation-based augmentation with automatic label coordinate transformation
for both OBB and bbox formats.
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ydt.core.formats import FormatType, detect_format
from ydt.core.logger import get_logger

logger = get_logger(__name__)


def preprocess_image_with_labels(
    image: np.ndarray,
    label_lines: list[str],
    padding: int = 20,
    format_type: FormatType | None = None,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """
    Preprocess image by cropping to annotation bounding box with padding.

    For landscape images, crops horizontally to包含 all annotations.
    For portrait images, crops vertically. Ensures crop region is at least
    as large as the short side of the image.

    Args:
        image: Input image
        label_lines: List of label lines
        padding: Padding around annotations in pixels
        format_type: Label format ('obb' or 'bbox'). If None, auto-detect.

    Returns:
        Tuple of (cropped_image, updated_labels, crop_info)
            crop_info contains 'x', 'y', 'width', 'height'

    Examples:
        >>> img = cv2.imread("image.jpg")
        >>> labels = ["0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"]
        >>> cropped, new_labels, info = preprocess_image_with_labels(img, labels)
        >>> print(info)
        {'x': 50, 'y': 0, 'width': 1820, 'height': 1080}
    """
    height, width = image.shape[:2]
    short_side = min(width, height)

    logger.debug(f"Image size: {width}x{height}, short side: {short_side}")

    # Collect all annotation points
    all_points = []
    for line in label_lines:
        if not line.strip():
            continue

        parts = line.strip().split()
        if not parts:
            continue

        # Auto-detect format from line content
        if format_type is None:
            if len(parts) == 5:
                line_format = "bbox"
            elif len(parts) == 9:
                line_format = "obb"
            else:
                line_format = "bbox"  # default
        else:
            line_format = format_type

        if line_format == "obb" and len(parts) == 9:
            # OBB format: class_id + 4 points (x,y)
            coords = np.array([float(x) for x in parts[1:]])
            points = coords.reshape(-1, 2)
        elif line_format == "bbox" and len(parts) == 5:
            # bbox format: class_id + center_x + center_y + width + height
            _, cx, cy, w, h = [float(x) for x in parts]
            # Convert to 4 corner points
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        else:
            logger.warning(f"Invalid label format: {line.strip()}")
            continue

        # Convert to absolute coordinates
        points[:, 0] *= width
        points[:, 1] *= height
        all_points.extend(points)

    if not all_points:
        logger.info("No annotations found, returning original image")
        return image, label_lines, {"x": 0, "y": 0, "width": width, "height": height}

    # Calculate annotation bounding box
    all_points = np.array(all_points)
    min_x = max(0, int(np.min(all_points[:, 0])) - padding)
    max_x = min(width, int(np.max(all_points[:, 0])) + padding)
    min_y = max(0, int(np.min(all_points[:, 1])) - padding)
    max_y = min(height, int(np.max(all_points[:, 1])) + padding)

    logger.debug(f"Annotation bbox (with padding): x:{min_x}-{max_x}, y:{min_y}-{max_y}")

    # Process image based on aspect ratio
    if width >= height:  # Landscape
        crop_width = max_x - min_x

        # Expand to at least short_side if needed
        if crop_width < short_side:
            extra_width = short_side - crop_width
            extra_each_side = extra_width // 2

            new_min_x = max(0, min_x - extra_each_side)
            new_max_x = min(width, max_x + extra_each_side)

            # Adjust if hitting boundaries
            if new_min_x == 0:
                new_max_x = min(width, short_side)
            elif new_max_x == width:
                new_min_x = max(0, width - short_side)

            min_x, max_x = new_min_x, new_max_x

        cropped_img = image[:, min_x:max_x]
        crop_info = {"x": min_x, "y": 0, "width": max_x - min_x, "height": height}

    else:  # Portrait
        crop_height = max_y - min_y

        # Expand to at least short_side if needed
        if crop_height < short_side:
            extra_height = short_side - crop_height
            extra_each_side = extra_height // 2

            new_min_y = max(0, min_y - extra_each_side)
            new_max_y = min(height, max_y + extra_each_side)

            # Adjust if hitting boundaries
            if new_min_y == 0:
                new_max_y = min(height, short_side)
            elif new_max_y == height:
                new_min_y = max(0, height - short_side)

            min_y, max_y = new_min_y, new_max_y

        cropped_img = image[min_y:max_y, :]
        crop_info = {"x": 0, "y": min_y, "width": width, "height": max_y - min_y}

    logger.debug(
        f"Crop region: x:{crop_info['x']}-{crop_info['x'] + crop_info['width']}, "
        f"y:{crop_info['y']}-{crop_info['y'] + crop_info['height']}"
    )

    # Update labels
    new_lines = []
    for line in label_lines:
        if not line.strip():
            continue

        parts = line.strip().split()
        if not parts:
            continue

        class_id = parts[0]

        # Auto-detect format from line content
        if format_type is None:
            line_format = detect_format(label_line=line)
        else:
            line_format = format_type

        if line_format == "obb" and len(parts) == 9:
            coords = np.array([float(x) for x in parts[1:]])
            points = coords.reshape(-1, 2)

            # Convert to absolute coordinates
            points[:, 0] *= width
            points[:, 1] *= height

            # Adjust to crop region
            points[:, 0] -= crop_info["x"]
            points[:, 1] -= crop_info["y"]

            # Convert back to relative coordinates
            points[:, 0] /= crop_info["width"]
            points[:, 1] /= crop_info["height"]

            # Format as string
            new_line = class_id
            for x, y in points:
                new_line += f" {x:.6f} {y:.6f}"
            new_lines.append(new_line)

        elif line_format == "bbox" and len(parts) == 5:
            cx, cy, w, h = [float(x) for x in parts[1:]]

            # Convert to absolute coordinates
            cx *= width
            cy *= height
            w *= width
            h *= height

            # Adjust to crop region
            cx -= crop_info["x"]
            cy -= crop_info["y"]

            # Convert back to relative coordinates
            cx /= crop_info["width"]
            cy /= crop_info["height"]
            w /= crop_info["width"]
            h /= crop_info["height"]

            new_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return cropped_img, new_lines, crop_info


def rotate_image_with_labels(
    image: np.ndarray,
    label_lines: list[str],
    angle: float,
    format_type: FormatType | None = None,
    min_area_ratio: float = 0.33,
) -> tuple[np.ndarray | None, list[str] | None]:
    """
    Rotate image and transform label coordinates.

    Args:
        image: Input image
        label_lines: List of label lines
        angle: Rotation angle in degrees (positive = counter-clockwise)
        format_type: Label format ('obb' or 'bbox'). If None, auto-detect.
        min_area_ratio: Minimum area ratio after rotation (filters out severely distorted annotations)

    Returns:
        Tuple of (rotated_image, updated_labels), or (None, None) if no valid labels remain

    Examples:
        >>> img = cv2.imread("image.jpg")
        >>> labels = ["0 0.5 0.5 0.7 0.5 0.7 0.7 0.5 0.7"]
        >>> rotated_img, rotated_labels = rotate_image_with_labels(img, labels, 45.0)
    """
    height, width = image.shape[:2]

    # Calculate rotated image dimensions
    angle_rad = np.deg2rad(angle)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    new_width = int(width * cos_a + height * sin_a)
    new_height = int(height * cos_a + width * sin_a)

    # Calculate rotation matrix
    center_x = width / 2
    center_y = height / 2
    new_center_x = new_width / 2
    new_center_y = new_height / 2

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotation_matrix[0, 2] += new_center_x - center_x
    rotation_matrix[1, 2] += new_center_y - center_y

    # Rotate image
    rotated_img = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    # Transform labels
    new_lines = []

    for line in label_lines:
        if not line.strip():
            continue

        parts = line.strip().split()
        if not parts:
            continue

        class_id = int(parts[0])

        # Auto-detect format from line content
        if format_type is None:
            line_format = detect_format(label_line=line)
        else:
            line_format = format_type

        # Convert to points for rotation
        if line_format == "obb" and len(parts) == 9:
            coords = np.array([float(x) for x in parts[1:]])
            points = coords.reshape(-1, 2)
        elif line_format == "bbox" and len(parts) == 5:
            cx, cy, w, h = [float(x) for x in parts[1:]]
            # Convert bbox to 4 corner points
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        else:
            logger.warning(f"Invalid label format: {line.strip()}")
            continue

        # Convert to absolute coordinates
        points[:, 0] *= width
        points[:, 1] *= height

        # Calculate original area
        original_area = cv2.contourArea(points.astype(np.float32))

        # Apply rotation
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        rotated_points = np.dot(points_homogeneous, rotation_matrix.T)

        # Calculate rotated area
        rotated_area = cv2.contourArea(rotated_points.astype(np.float32))

        # Filter out severely distorted annotations
        if rotated_area < original_area * min_area_ratio:
            logger.debug(
                f"Skipping annotation: area ratio {rotated_area / original_area:.2f} < {min_area_ratio}"
            )
            continue

        # Get minimum area rectangle for OBB
        rect = cv2.minAreaRect(rotated_points.astype(np.float32))
        box = cv2.boxPoints(rect)

        # Convert to relative coordinates
        box[:, 0] /= new_width
        box[:, 1] /= new_height

        # Clip to [0, 1] range
        box = np.clip(box, 0, 1)

        # Format as string (always OBB after rotation)
        new_line = f"{class_id}"
        for x, y in box:
            new_line += f" {x:.6f} {y:.6f}"
        new_lines.append(new_line)

    if not new_lines:
        logger.warning(f"No valid labels after rotation by {angle} degrees")
        return None, None

    return rotated_img, new_lines


def augment_dataset(
    dataset_path: str | Path,
    output_path: str | Path,
    angles: list[float] | None = None,
    class_specific_angles: dict[set[int], list[float]] | None = None,
    format_type: FormatType | None = None,
    preprocess: bool = True,
    test_mode: bool = False,
    test_count: int = 20,
) -> dict[str, int]:
    """
    Batch augment dataset with rotation.

    Args:
        dataset_path: Input dataset directory or single image file
        output_path: Output directory for augmented dataset
        angles: List of rotation angles (degrees). If None and class_specific_angles is None,
                uses [45, 90, 135] as default.
        class_specific_angles: Dictionary mapping class ID sets to rotation angles.
                               Example: {{0, 5, 10}: [45, 90], {7}: [45, 90, 135, 180, 225, 270, 315]}
        format_type: Label format ('obb' or 'bbox'). If None, auto-detect.
        preprocess: If True, preprocess images by cropping to annotation bbox
        test_mode: If True, process only test_count images
        test_count: Number of images to process in test mode

    Returns:
        Dictionary with statistics (processed, rotations, skipped)

    Raises:
        FileNotFoundError: If dataset_path doesn't exist

    Examples:
        >>> # Basic augmentation with default angles
        >>> stats = augment_dataset("./dataset", "./dataset_augmented")

        >>> # Custom angles for specific classes
        >>> class_angles = {
        ...     {0, 1, 2}: [45, 90, 135],  # Regular classes
        ...     {7}: [45, 90, 135, 180, 225, 270, 315]  # Class 7 needs more angles
        ... }
        >>> stats = augment_dataset(
        ...     "./dataset",
        ...     "./output",
        ...     class_specific_angles=class_angles
        ... )
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if test_mode:
        logger.info(f"Test mode: processing only {test_count} images")

    stats = {
        "processed": 0,
        "rotations": 0,
        "skipped": 0,
    }

    # Check if input is a single file or directory
    if dataset_path.is_file():
        # Single file mode
        supported_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]
        if dataset_path.suffix not in supported_extensions:
            raise ValueError(f"Unsupported image format: {dataset_path.suffix}")

        logger.info(f"Processing single image file: {dataset_path.name}")

        # Create output directories
        output_images = output_path
        output_labels = output_path
        output_images.mkdir(parents=True, exist_ok=True)

        # Find corresponding label file
        label_path = dataset_path.with_suffix(".txt")
        if not label_path.exists():
            # Try looking in sibling labels directory
            labels_dir = dataset_path.parent / "labels"
            label_path = labels_dir / f"{dataset_path.stem}.txt"

        image_files = [dataset_path]
        label_files = {dataset_path: label_path if label_path.exists() else None}

        # Process single image
        for img_path in image_files:
            label_path = label_files[img_path]
            if label_path is None or not label_path.exists():
                logger.warning(f"Label file not found for: {img_path.name}")
                stats["skipped"] += 1
                continue

            # Read image and labels
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Cannot read image: {img_path}")
                stats["skipped"] += 1
                continue

            with open(label_path, encoding="utf-8") as f:
                label_lines = f.readlines()

            # Preprocess if requested
            if preprocess:
                img, label_lines, crop_info = preprocess_image_with_labels(
                    img, label_lines, format_type=format_type
                )

            # Save preprocessed original
            output_img_path = output_images / img_path.name
            output_label_path = output_labels / f"{img_path.stem}.txt"

            cv2.imwrite(str(output_img_path), img)
            with open(output_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            # Determine rotation angles
            img_angles = None

            if class_specific_angles:
                # Check which classes are in this image
                image_classes = set()
                for line in label_lines:
                    if line.strip():
                        parts = line.strip().split()
                        if parts:
                            image_classes.add(int(parts[0]))

                # Find matching class set
                for class_set, angle_list in class_specific_angles.items():
                    if image_classes & class_set:  # If any overlap
                        img_angles = angle_list
                        logger.debug(f"Using class-specific angles: {img_angles}")
                        break

            if img_angles is None:
                img_angles = angles if angles is not None else [45, 90, 135]

            # Perform rotations
            rotation_success = 0
            for angle in img_angles:
                rotated_img, rotated_labels = rotate_image_with_labels(
                    img, label_lines, angle, format_type=format_type
                )

                if rotated_img is not None:
                    # Save rotated image and labels
                    rot_img_name = f"rot_{int(angle)}_{img_path.name}"
                    rot_label_name = f"rot_{int(angle)}_{img_path.stem}.txt"

                    rot_img_path = output_images / rot_img_name
                    rot_label_path = output_labels / rot_label_name

                    cv2.imwrite(str(rot_img_path), rotated_img)
                    with open(rot_label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(rotated_labels))

                    rotation_success += 1
                    stats["rotations"] += 1

            logger.debug(f"Successful rotations: {rotation_success}/{len(img_angles)}")
            stats["processed"] += 1

        logger.info("Augmentation complete!")
        logger.info(
            f"Processed: {stats['processed']}, Rotations: {stats['rotations']}, "
            f"Skipped: {stats['skipped']}"
        )
        return stats

    # Directory mode
    # Determine if using subdirectory structure
    use_subdir = (dataset_path / "images" / "train").exists()
    subdirs = ["train", "val"] if use_subdir else [""]

    for subdir in subdirs:
        if subdir:
            logger.info(f"Processing {subdir} split...")

        # Setup directories
        if use_subdir:
            image_dir = dataset_path / "images" / subdir
            label_dir = dataset_path / "labels" / subdir
            output_images = output_path / "images" / subdir
            output_labels = output_path / "labels" / subdir
        else:
            image_dir = dataset_path / "images"
            label_dir = dataset_path / "labels"
            output_images = output_path / "images"
            output_labels = output_path / "labels"

        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]:
            image_files.extend(list(image_dir.glob(f"*{ext}")))

        image_files = sorted(image_files)
        if test_mode:
            image_files = image_files[:test_count]

        logger.info(f"Found {len(image_files)} images")

        for img_path in tqdm(
            image_files, desc=f"Processing {subdir}" if subdir else "Processing images"
        ):
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                logger.warning(f"Label file not found: {str(label_path)}")
                stats["skipped"] += 1
                continue

            # Read image and labels
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Cannot read image: {img_path}")
                stats["skipped"] += 1
                continue

            with open(label_path, encoding="utf-8") as f:
                label_lines = f.readlines()

            # Preprocess if requested
            if preprocess:
                img, label_lines, crop_info = preprocess_image_with_labels(
                    img, label_lines, format_type=format_type
                )

            # Save preprocessed original
            output_img_path = output_images / img_path.name
            output_label_path = output_labels / label_path.name

            cv2.imwrite(str(output_img_path), img)
            with open(output_label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            # Determine rotation angles
            img_angles = None

            if class_specific_angles:
                # Check which classes are in this image
                image_classes = set()
                for line in label_lines:
                    if line.strip():
                        parts = line.strip().split()
                        if parts:
                            image_classes.add(int(parts[0]))

                # Find matching class set
                for class_set, angle_list in class_specific_angles.items():
                    if image_classes & class_set:  # If any overlap
                        img_angles = angle_list
                        logger.debug(f"Using class-specific angles: {img_angles}")
                        break

            if img_angles is None:
                img_angles = angles if angles is not None else [45, 90, 135]

            # Perform rotations
            rotation_success = 0
            for angle in img_angles:
                rotated_img, rotated_labels = rotate_image_with_labels(
                    img, label_lines, angle, format_type=format_type
                )

                if rotated_img is not None:
                    # Save rotated image and labels
                    rot_img_name = f"rot_{int(angle)}_{img_path.name}"
                    rot_label_name = f"rot_{int(angle)}_{label_path.name}"

                    rot_img_path = output_images / rot_img_name
                    rot_label_path = output_labels / rot_label_name

                    cv2.imwrite(str(rot_img_path), rotated_img)
                    with open(rot_label_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(rotated_labels))

                    rotation_success += 1
                    stats["rotations"] += 1

            logger.debug(f"Successful rotations: {rotation_success}/{len(img_angles)}")
            stats["processed"] += 1

            if stats["processed"] % 10 == 0:
                logger.info(
                    f"Progress: {stats['processed']} images processed, "
                    f"{stats['rotations']} rotations generated"
                )

        logger.info(f"{subdir if subdir else 'Main'} directory complete")

    logger.info("Augmentation complete!")
    logger.info(
        f"Processed: {stats['processed']}, Rotations: {stats['rotations']}, "
        f"Skipped: {stats['skipped']}"
    )

    return stats

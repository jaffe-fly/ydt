"""
Image cropping utilities for extracting objects from images.

Provides functionality to crop detected objects from images in two modes:
1. Model inference mode: Use YOLO model to detect and crop objects
2. Dataset mode: Crop objects based on existing label files

Features:
- Support both OBB (oriented bounding box) and regular bbox formats
- Transparent background (PNG with alpha channel)
- Size filtering (min/max dimensions)
- Class-based organization of cropped objects

Examples:
    >>> # Model inference mode
    >>> crop_with_model(
    ...     source="./images",
    ...     model_path="weights/best.pt",
    ...     output_dir="./cropped",
    ...     obb=True
    ... )

    >>> # Dataset mode
    >>> crop_from_dataset(
    ...     dataset_path="./dataset",
    ...     output_dir="./cropped",
    ...     split="train",
    ...     format_type="obb"
    ... )
"""

from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from ydt.core import IMAGE_EXTENSIONS, get_logger
from ydt.core.formats import detect_format

logger = get_logger(__name__)


def get_rotated_rect_points(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    """
    Calculate four corner points of rotated rectangle.

    Args:
        cx: Center x coordinate
        cy: Center y coordinate
        w: Width
        h: Height
        angle: Rotation angle in radians

    Returns:
        4x2 array of corner points
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Four corners relative to center
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])

    # Rotate
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = corners @ rotation_matrix.T

    # Translate to actual position
    rotated_corners[:, 0] += cx
    rotated_corners[:, 1] += cy

    return rotated_corners


def crop_rotated_bbox(
    image: np.ndarray,
    obb_coords: np.ndarray,
    padding: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Crop rotated bounding box region from image with transparent background.

    Args:
        image: Input image
        obb_coords: OBB coordinates [cx, cy, w, h, angle]
        padding: Padding pixels around crop

    Returns:
        Tuple of (cropped_image with alpha channel, mask), or (None, None) if invalid
    """
    cx, cy, w, h, angle = obb_coords

    # Get rotated rectangle corner points
    corners = get_rotated_rect_points(cx, cy, w, h, angle)

    # Calculate minimum bounding rectangle
    x_min = max(0, int(corners[:, 0].min()) - padding)
    y_min = max(0, int(corners[:, 1].min()) - padding)
    x_max = min(image.shape[1], int(corners[:, 0].max()) + padding)
    y_max = min(image.shape[0], int(corners[:, 1].max()) + padding)

    # Crop dimensions
    crop_width = x_max - x_min
    crop_height = y_max - y_min

    if crop_width <= 0 or crop_height <= 0:
        return None, None

    # Adjust corner coordinates to crop region
    corners_adjusted = corners.copy()
    corners_adjusted[:, 0] -= x_min
    corners_adjusted[:, 1] -= y_min

    # Create mask
    mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_adjusted.astype(np.int32)], 255)

    # Crop image
    cropped = image[y_min:y_max, x_min:x_max].copy()

    # Create image with alpha channel (transparent background)
    cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    cropped_rgba[:, :, 3] = mask
    return cropped_rgba, mask


def crop_bbox(
    image: np.ndarray,
    bbox_coords: list[float],
    padding: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Crop regular bounding box region from image with transparent background.

    Args:
        image: Input image
        bbox_coords: Bounding box coordinates [x1, y1, x2, y2]
        padding: Padding pixels around crop

    Returns:
        Tuple of (cropped_image with alpha channel, mask), or (None, None) if invalid
    """
    x1, y1, x2, y2 = bbox_coords

    # Add padding and ensure within image bounds
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(image.shape[1], int(x2) + padding)
    y2 = min(image.shape[0], int(y2) + padding)

    crop_width = x2 - x1
    crop_height = y2 - y1

    if crop_width <= 0 or crop_height <= 0:
        return None, None

    # Crop image
    cropped = image[y1:y2, x1:x2].copy()

    # Create rectangular mask
    mask = np.ones((crop_height, crop_width), dtype=np.uint8) * 255

    # Create image with alpha channel (transparent background)
    cropped_rgba = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    cropped_rgba[:, :, 3] = mask
    return cropped_rgba, mask


def _check_size_filter(cropped: np.ndarray, min_size: int | None, max_size: int | None) -> bool:
    """
    Check if cropped image meets size requirements.

    Args:
        cropped: Cropped image
        min_size: Minimum dimension (width or height)
        max_size: Maximum dimension (width or height)

    Returns:
        True if image meets requirements, False otherwise
    """
    h, w = cropped.shape[:2]

    if min_size is not None and (h < min_size or w < min_size):
        return False

    if max_size is not None and (h > max_size or w > max_size):
        return False

    return True


def crop_with_model(
    source: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    classes: list[int] | None = None,
    conf: float = 0.5,
    iou: float = 0.5,
    obb: bool = True,
    padding: int = 0,
    min_size: int | None = None,
    max_size: int | None = None,
    device: int = 0,
) -> dict:
    """
    Crop objects from images using YOLO model inference with transparent background.

    Args:
        source: Input image path or directory
        model_path: YOLO model weights path
        output_dir: Output directory
        classes: List of class IDs to crop (None = all classes)
        conf: Confidence threshold
        iou: NMS IOU threshold
        obb: Whether to use OBB (oriented bounding box) mode
        padding: Padding pixels around crop (default: 0)
        min_size: Minimum object dimension in pixels
        max_size: Maximum object dimension in pixels
        device: Device ID for inference

    Returns:
        Dictionary with statistics:
        - total_images: Number of images processed
        - total_cropped: Number of objects cropped
        - stats_by_class: Dictionary of {class_name: count}
        - filtered_by_size: Number of objects filtered by size

    Examples:
        >>> result = crop_with_model(
        ...     source="./images",
        ...     model_path="weights/best.pt",
        ...     output_dir="./cropped",
        ...     obb=True,
        ...     min_size=50,
        ...     max_size=1000
        ... )
        >>> print(f"Cropped {result['total_cropped']} objects")
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for model inference. Install it with: pip install ultralytics"
        )

    # Load model
    logger.info(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path, task="obb" if obb else "detect")
        class_names = model.names
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect image files
    source_path = Path(source)
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = [f for f in source_path.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        raise ValueError(f"Invalid source path: {source}")

    logger.info(f"Found {len(image_files)} images")
    logger.info(f"Output directory: {output_path}")
    logger.info("Background: Transparent (PNG with alpha channel)")
    if min_size:
        logger.info(f"Min size filter: {min_size}px")
    if max_size:
        logger.info(f"Max size filter: {max_size}px")

    total_cropped = 0
    filtered_by_size = 0
    stats_by_class = {}
    failed_images = []

    # Process each image
    for img_path in tqdm(image_files, desc="Cropping objects"):
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Failed to read {img_path}")
            failed_images.append(str(img_path))
            continue

        # Inference
        try:
            results = model(
                image,
                conf=conf,
                iou=iou,
                device=device,
                classes=classes,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"Inference failed for {img_path}: {e}")
            failed_images.append(str(img_path))
            continue

        result = results[0]
        boxes = result.obb if obb else result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        # Process each detected object
        img_name = img_path.stem
        for obj_idx, (box, cls_id, conf_score) in enumerate(
            zip[tuple](boxes.data, boxes.cls, boxes.conf)
        ):
            cls_id = int(cls_id)
            cls_name = class_names[cls_id]
            conf_score = float(conf_score)

            # Crop object
            if obb:
                # OBB format: [cx, cy, w, h, angle]
                obb_coords = box[:5].cpu().numpy()
                cropped, mask = crop_rotated_bbox(image, obb_coords, padding)
            else:
                # BBOX format: [x1, y1, x2, y2]
                bbox_coords = box[:4].cpu().numpy().tolist()
                cropped, mask = crop_bbox(image, bbox_coords, padding)

            if cropped is None:
                continue

            # Size filter
            if not _check_size_filter(cropped, min_size, max_size):
                filtered_by_size += 1
                continue

            # Generate filename
            filename = f"{img_name}_obj{obj_idx}_conf{conf_score:.2f}.png"

            # Save cropped object
            obj_save_path = output_path / "objects" / cls_name / filename
            obj_save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(obj_save_path), cropped)

            total_cropped += 1
            stats_by_class[cls_name] = stats_by_class.get(cls_name, 0) + 1

    # Log statistics
    logger.info("=" * 70)
    logger.info("Cropping completed successfully")
    logger.info(f"Total images processed: {len(image_files)}")
    logger.info(f"Total objects cropped: {total_cropped}")
    if filtered_by_size > 0:
        logger.info(f"Objects filtered by size: {filtered_by_size}")
    if failed_images:
        logger.warning(f"Failed to process {len(failed_images)} images")

    logger.info("\nStatistics by class:")
    for cls_name, count in sorted(stats_by_class.items()):
        logger.info(f"  {cls_name}: {count} objects")

    return {
        "total_images": len(image_files),
        "total_cropped": total_cropped,
        "stats_by_class": stats_by_class,
        "filtered_by_size": filtered_by_size,
        "failed_images": failed_images,
    }


def crop_from_dataset(
    dataset_path: str | Path,
    output_dir: str | Path,
    split: str = "train",
    classes: list[int] | None = None,
    padding: int = 0,
    min_size: int | None = None,
    max_size: int | None = None,
    format_type: str | None = None,
) -> dict:
    """
    Crop objects from dataset using existing label files with transparent background.

    Args:
        dataset_path: Path to YOLO dataset directory
        output_dir: Output directory
        split: Which split to process ("train", "val", or "both")
        classes: List of class IDs to crop (None = all classes)
        min_size: Minimum object dimension in pixels
        max_size: Maximum object dimension in pixels
        format_type: Label format ("obb" or "bbox", auto-detect if None)

    Returns:
        Dictionary with statistics:
        - total_images: Number of images processed
        - total_cropped: Number of objects cropped
        - stats_by_class: Dictionary of {class_name: count}
        - filtered_by_size: Number of objects filtered by size

    Examples:
        >>> result = crop_from_dataset(
        ...     dataset_path="./dataset",
        ...     output_dir="./cropped",
        ...     split="train",
        ...     classes=[0, 1],
        ...     min_size=50
        ... )
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Read data.yaml
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Get class names
    names = data.get("names", {})
    if isinstance(names, list):
        class_names = dict(enumerate(names))
    else:
        class_names = names

    # Determine splits to process
    splits_to_process = []
    if split == "both":
        splits_to_process = ["train", "val"]
    else:
        splits_to_process = [split]

    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Processing splits: {', '.join(splits_to_process)}")
    logger.info("Background: Transparent (PNG with alpha channel)")
    if min_size:
        logger.info(f"Min size filter: {min_size}px")
    if max_size:
        logger.info(f"Max size filter: {max_size}px")

    total_cropped = 0
    filtered_by_size = 0
    stats_by_class = {}
    total_images = 0
    failed_images = []

    for current_split in splits_to_process:
        img_dir = dataset_path / "images" / current_split
        label_dir = dataset_path / "labels" / current_split

        if not img_dir.exists() or not label_dir.exists():
            logger.warning(f"Split '{current_split}' not found, skipping")
            continue

        # Get all images
        image_files = [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        logger.info(f"Processing {len(image_files)} images from {current_split} split")

        # Auto-detect format if not specified
        detected_format = format_type
        if detected_format is None:
            # Try to detect from first label file
            for img_file in image_files:
                label_file = label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    try:
                        detected_format = detect_format(str(label_file))
                        logger.info(f"Auto-detected format: {detected_format}")
                        break
                    except Exception:
                        continue

            if detected_format is None:
                detected_format = "bbox"
                logger.warning(f"Could not auto-detect format, assuming: {detected_format}")

        is_obb = detected_format == "obb"

        # Process each image
        for img_file in tqdm(image_files, desc=f"Cropping {current_split}"):
            label_file = label_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue

            image = cv2.imread(str(img_file))
            if image is None:
                logger.warning(f"Failed to read {img_file}")
                failed_images.append(str(img_file))
                continue

            total_images += 1
            img_h, img_w = image.shape[:2]

            # Read labels
            try:
                with open(label_file, encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                logger.warning(f"Failed to read label {label_file}: {e}")
                continue

            # Process each object in the image
            for obj_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                cls_id = int(float(parts[0]))

                # Filter by class
                if classes is not None and cls_id not in classes:
                    continue

                cls_name = class_names.get(cls_id, f"class_{cls_id}")

                # Parse coordinates and crop
                if is_obb and len(parts) == 9:
                    # OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
                    coords = [float(x) for x in parts[1:]]
                    # Convert to pixel coordinates
                    points = np.array(coords).reshape(-1, 2)
                    points[:, 0] *= img_w
                    points[:, 1] *= img_h

                    # Calculate center, width, height, angle
                    rect = cv2.minAreaRect(points.astype(np.float32))
                    cx, cy = rect[0]
                    w, h = rect[1]
                    angle = np.deg2rad(rect[2])

                    obb_coords = np.array([cx, cy, w, h, angle])
                    cropped, mask = crop_rotated_bbox(image, obb_coords, padding)

                elif len(parts) == 5:
                    # BBox format: class_id x_center y_center width height (normalized)
                    _, cx, cy, w, h = [float(x) for x in parts]

                    # Convert to pixel coordinates
                    cx *= img_w
                    cy *= img_h
                    w *= img_w
                    h *= img_h

                    # Convert to x1, y1, x2, y2
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    bbox_coords = [x1, y1, x2, y2]
                    cropped, mask = crop_bbox(image, bbox_coords, padding)
                else:
                    continue

                if cropped is None:
                    continue

                # Size filter
                if not _check_size_filter(cropped, min_size, max_size):
                    filtered_by_size += 1
                    continue

                # Generate filename
                filename = f"{img_file.stem}_obj{obj_idx}.png"

                # Save cropped object
                obj_save_path = output_path / "objects" / cls_name / filename
                obj_save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(obj_save_path), cropped)

                total_cropped += 1
                stats_by_class[cls_name] = stats_by_class.get(cls_name, 0) + 1

    # Log statistics
    logger.info("=" * 70)
    logger.info("Cropping completed successfully")
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Total objects cropped: {total_cropped}")
    if filtered_by_size > 0:
        logger.info(f"Objects filtered by size: {filtered_by_size}")
    if failed_images:
        logger.warning(f"Failed to process {len(failed_images)} images")

    logger.info("\nStatistics by class:")
    for cls_name, count in sorted(stats_by_class.items()):
        logger.info(f"  {cls_name}: {count} objects")

    return {
        "total_images": total_images,
        "total_cropped": total_cropped,
        "stats_by_class": stats_by_class,
        "filtered_by_size": filtered_by_size,
        "failed_images": failed_images,
    }

"""
Dataset visualization tools for YOLO OBB format.

Provides interactive visualization of datasets with keyboard navigation
and label filtering capabilities.
"""

from pathlib import Path

import cv2
import numpy as np
import yaml

from ydt.core import IMAGE_EXTENSIONS
from ydt.core.logger import get_logger

logger = get_logger(__name__)


def draw_obb(
    points: list[float],
    img: np.ndarray,
    color: tuple[int, int, int] | None = None,
    label: str | None = None,
    line_thickness: int | None = None,
) -> None:
    """
    Draw oriented bounding box on image.

    Args:
        points: Flat list of 8 coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
        img: Image array (modified in-place)
        color: BGR color tuple. If None, uses random color.
        label: Optional label text to draw
        line_thickness: Line thickness. If None, auto-calculated from ydt.image size.

    Examples:
        >>> img = cv2.imread("image.jpg")
        >>> points = [100, 100, 200, 100, 200, 200, 100, 200]
        >>> draw_obb(points, img, color=(0, 255, 0), label="card")
    """
    # Calculate line thickness
    if line_thickness is None:
        tl = max(round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1, 1)
    else:
        tl = line_thickness

    # Generate random color if not provided
    if color is None:
        color = tuple(int(x) for x in np.random.randint(0, 255, 3))

    # Convert points to integer coordinates
    pts = np.array(
        [[int(points[i]), int(points[i + 1])] for i in range(0, len(points), 2)],
        np.int32,
    )
    pts = pts.reshape((-1, 1, 2))

    # Draw polygon
    cv2.polylines(img, [pts], True, color, thickness=tl, lineType=cv2.LINE_AA)

    # Draw label if provided
    if label:
        tf = max(tl - 1, 1)  # Font thickness

        # Calculate center point for text placement
        center_x = int(np.mean(pts[:, 0, 0]))
        center_y = int(np.mean(pts[:, 0, 1]))

        # Get text size
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = center_x + t_size[0], center_y - t_size[1] - 3

        # Draw background rectangle and text
        cv2.rectangle(img, (center_x, center_y), c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (center_x, center_y - 2),
            0,
            tl / 3,
            (225, 255, 255),
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def visualize_dataset(
    dataset_path: str | Path,
    filter_labels: list[str] | None = None,
    scan_train: bool = True,
    scan_val: bool = True,
    single_image_path: str | Path | None = None,
    window_name: str = "Dataset Visualization",
    wait_key: int = 0,
) -> int:
    """
    Interactively visualize YOLO OBB dataset.

    Keyboard controls:
    - 'n' or Space: Next image
    - 'p': Previous image
    - ESC or 'q': Quit

    Args:
        dataset_path: Root directory of dataset
        filter_labels: Optional list of label names to display (shows all if None)
        scan_train: If True, include training set
        scan_val: If True, include validation set
        single_image_path: If provided, only visualize this single image
        window_name: OpenCV window name
        wait_key: Milliseconds to wait for key (0 = wait indefinitely)

    Returns:
        Number of images visualized

    Raises:
        FileNotFoundError: If dataset path or YAML file doesn't exist
        ValueError: If neither scan_train nor scan_val is True

    Examples:
        >>> # Visualize entire dataset
        >>> visualize_dataset("./dataset", scan_train=True, scan_val=True)

        >>> # Visualize only specific labels
        >>> visualize_dataset(
        ...     "./dataset",
        ...     filter_labels=["card_A", "card_K"],
        ...     scan_val=True
        ... )

        >>> # Visualize single image
        >>> visualize_dataset(
        ...     "./dataset",
        ...     single_image_path="./dataset/images/train/img_001.jpg"
        ... )
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Read data.yaml
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Get class names
    names = data.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))

    logger.info(f"Loaded {len(names)} class names")

    # Collect image files
    if single_image_path:
        single_image_path = Path(single_image_path)
        if not single_image_path.exists():
            raise FileNotFoundError(f"Image not found: {single_image_path}")

        # Determine split (train or val)
        split = None
        if "train" in str(single_image_path):
            split = "train"
        elif "val" in str(single_image_path):
            split = "val"
        else:
            logger.warning("Cannot determine if image is train or val, assuming train")
            split = "train"

        image_files = [(split, single_image_path.name)]
        logger.info(f"Visualizing single image: {single_image_path}")

    else:
        if not scan_train and not scan_val:
            raise ValueError("At least one of scan_train or scan_val must be True")

        image_files = []

        def has_filtered_labels(label_path: Path) -> bool:
            """Check if label file contains any of the filtered labels"""
            if not filter_labels or not label_path.exists():
                return True

            try:
                with open(label_path, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 9:  # OBB format
                            cls = int(float(parts[0]))
                            class_name = names.get(cls, f"class_{cls}")
                            if class_name in filter_labels:
                                return True
            except Exception as e:
                logger.warning(f"Error reading {label_path}: {e}")

            return False

        # Scan train set
        if scan_train:
            train_img_dir = dataset_path / "images" / "train"
            train_label_dir = dataset_path / "labels" / "train"

            if train_img_dir.exists():
                train_images = [
                    f.name for f in train_img_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS
                ]

                # Filter by labels
                if filter_labels:
                    train_images = [
                        img
                        for img in train_images
                        if has_filtered_labels(train_label_dir / (Path(img).stem + ".txt"))
                    ]

                image_files.extend([("train", img) for img in train_images])
                logger.info(f"Found {len(train_images)} training images")

        # Scan val set
        if scan_val:
            val_img_dir = dataset_path / "images" / "val"
            val_label_dir = dataset_path / "labels" / "val"

            if val_img_dir.exists():
                val_images = [f.name for f in val_img_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS]

                # Filter by labels
                if filter_labels:
                    val_images = [
                        img
                        for img in val_images
                        if has_filtered_labels(val_label_dir / (Path(img).stem + ".txt"))
                    ]

                image_files.extend([("val", img) for img in val_images])
                logger.info(f"Found {len(val_images)} validation images")

    if not image_files:
        logger.warning("No images found")
        return 0

    logger.info(f"Total images: {len(image_files)}")

    # Interactive visualization loop
    current_idx = 0
    visualized_count = 0

    while current_idx < len(image_files):
        split, img_file = image_files[current_idx]

        # Build paths
        if single_image_path:
            img_path = single_image_path
            label_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        else:
            img_path = dataset_path / "images" / split / img_file
            label_path = dataset_path / "labels" / split / (Path(img_file).stem + ".txt")

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error(f"Cannot read image: {img_path}")
            current_idx += 1
            continue

        height, width = img.shape[:2]

        # Draw split indicator
        cv2.putText(
            img,
            f"Dataset: {split}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Draw labels
        if label_path.exists():
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) == 9:  # OBB format
                        cls = int(float(parts[0]))
                        class_name = names.get(cls, f"class_{cls}")

                        # Skip if filtering and not in filter list
                        if filter_labels and class_name not in filter_labels:
                            continue

                        # Convert normalized coordinates to pixels
                        points = []
                        for i in range(1, 9, 2):
                            x = float(parts[i]) * width
                            y = float(parts[i + 1]) * height
                            points.extend([x, y])

                        # Draw OBB
                        draw_obb(points, img, label=class_name)

                    elif len(parts) == 5:  # bbox format
                        cls = int(float(parts[0]))
                        class_name = names.get(cls, f"class_{cls}")

                        if filter_labels and class_name not in filter_labels:
                            continue

                        # Convert bbox to OBB for drawing
                        cx, cy, w, h = [float(x) for x in parts[1:]]
                        x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
                        x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
                        points = [x1, y1, x2, y1, x2, y2, x1, y2]

                        draw_obb(points, img, label=class_name)

                    else:
                        logger.warning(f"Unexpected label format: {len(parts)} values")

        logger.info(f"[{current_idx + 1}/{len(image_files)}] {split}/{img_path.name} {img.shape}")

        # Show image
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)

        visualized_count += 1

        # Wait for key press
        key = cv2.waitKey(wait_key) & 0xFF

        if key == 27 or key == ord("q"):  # ESC or 'q'
            logger.info("User quit")
            break
        elif key == ord("n") or key == ord(" "):  # Next
            if single_image_path:
                break
            current_idx += 1
            if current_idx >= len(image_files):
                logger.info("Reached last image")
                current_idx = len(image_files) - 1
        elif key == ord("p"):  # Previous
            if single_image_path:
                break
            current_idx = max(0, current_idx - 1)
            if current_idx == 0:
                logger.info("Reached first image")

    cv2.destroyAllWindows()
    logger.info(f"Visualized {visualized_count} images")

    return visualized_count

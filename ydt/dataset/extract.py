"""
Dataset extraction utilities.

Extract specific classes, images, or labels from YOLO datasets.
"""

import shutil
from pathlib import Path

import yaml

from ydt.core import IMAGE_EXTENSIONS
from ydt.core.logger import get_logger

logger = get_logger(__name__)


def extract_by_class(
    input_dir: str | Path,
    output_dir: str | Path,
    class_ids: list[int],
    operation: str = "copy",
    extract_train: bool = True,
    extract_val: bool = True,
    filter_labels: bool = False,
    remap_ids: bool = False,
) -> dict[str, int]:
    """
    Extract images and labels containing specified classes from dataset.

    Args:
        input_dir: Input YOLO dataset directory
        output_dir: Output directory for extracted data
        class_ids: List of class IDs to extract (e.g., [0, 2, 5])
        operation: "copy" (default) or "move"
        extract_train: Extract from training set
        extract_val: Extract from validation set
        filter_labels: If True, only keep annotation lines for specified classes;
                      If False, keep entire label file as-is (default)
        remap_ids: If True and filter_labels=True, remap class IDs to sequential
                  starting from 0 (e.g., [2, 5] -> [0, 1])

    Returns:
        Dictionary with extraction statistics

    Raises:
        FileNotFoundError: If input directory or data.yaml not found
        ValueError: If operation is invalid or no classes specified

    Examples:
        >>> # Extract classes 0 and 2, keep all labels
        >>> stats = extract_by_class(
        ...     "./dataset",
        ...     "./extracted",
        ...     class_ids=[0, 2],
        ... )

        >>> # Extract class 1, filter labels (only keep class 1 annotations)
        >>> stats = extract_by_class(
        ...     "./dataset",
        ...     "./extracted",
        ...     class_ids=[1],
        ...     filter_labels=True,
        ... )

        >>> # Extract classes 2,5, filter and remap to 0,1
        >>> stats = extract_by_class(
        ...     "./dataset",
        ...     "./extracted",
        ...     class_ids=[2, 5],
        ...     filter_labels=True,
        ...     remap_ids=True,
        ... )
    """
    if not class_ids:
        raise ValueError("At least one class ID must be specified")

    if operation not in ["copy", "move"]:
        raise ValueError(f"Invalid operation: {operation}. Must be 'copy' or 'move'")

    if not extract_train and not extract_val:
        raise ValueError("At least one of extract_train or extract_val must be True")

    if remap_ids and not filter_labels:
        raise ValueError("remap_ids requires filter_labels=True")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Read data.yaml
    yaml_path = input_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {input_dir}")

    with open(yaml_path, encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    class_names = data_config.get("names", [])
    if not class_names:
        raise ValueError("No class names found in data.yaml")

    # Validate class IDs
    class_ids_set = set(class_ids)
    for cid in class_ids:
        if cid < 0 or cid >= len(class_names):
            raise ValueError(f"Invalid class ID {cid}. Valid range: 0-{len(class_names) - 1}")

    # Create ID mapping if remapping
    id_mapping = None
    if remap_ids:
        sorted_ids = sorted(class_ids_set)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
        logger.info(f"Class ID mapping: {id_mapping}")

    logger.info(f"Extracting class IDs: {sorted(class_ids_set)}")
    logger.info(f"Class names: {[class_names[i] for i in sorted(class_ids_set)]}")
    logger.info(f"Operation: {operation}")
    logger.info(f"Filter labels: {filter_labels}")
    if filter_labels:
        logger.info(f"Remap IDs: {remap_ids}")

    # Statistics
    stats = {"train_images": 0, "train_labels": 0, "val_images": 0, "val_labels": 0}

    # Image extensions

    # Process training set
    if extract_train:
        src_train_img = input_dir / "images" / "train"
        src_train_label = input_dir / "labels" / "train"

        if src_train_img.exists() and src_train_label.exists():
            dst_train_img = output_dir / "images" / "train"
            dst_train_label = output_dir / "labels" / "train"
            dst_train_img.mkdir(parents=True, exist_ok=True)
            dst_train_label.mkdir(parents=True, exist_ok=True)

            logger.info("Processing training set...")
            train_stats = _extract_split(
                src_train_img,
                src_train_label,
                dst_train_img,
                dst_train_label,
                class_ids_set,
                operation,
                filter_labels,
                id_mapping,
            )
            stats["train_images"] = train_stats["images"]
            stats["train_labels"] = train_stats["labels"]
        else:
            logger.warning(f"Training directories not found in {input_dir}")

    # Process validation set
    if extract_val:
        src_val_img = input_dir / "images" / "val"
        src_val_label = input_dir / "labels" / "val"

        if src_val_img.exists() and src_val_label.exists():
            dst_val_img = output_dir / "images" / "val"
            dst_val_label = output_dir / "labels" / "val"
            dst_val_img.mkdir(parents=True, exist_ok=True)
            dst_val_label.mkdir(parents=True, exist_ok=True)

            logger.info("Processing validation set...")
            val_stats = _extract_split(
                src_val_img,
                src_val_label,
                dst_val_img,
                dst_val_label,
                class_ids_set,
                operation,
                filter_labels,
                id_mapping,
            )
            stats["val_images"] = val_stats["images"]
            stats["val_labels"] = val_stats["labels"]
        else:
            logger.warning(f"Validation directories not found in {input_dir}")

    # Create data.yaml
    dst_yaml = output_dir / "data.yaml"
    if remap_ids:
        # Create new config with remapped classes
        new_names = [class_names[old_id] for old_id in sorted(class_ids_set)]
        new_config = {
            "path": ".",
            "nc": len(new_names),
            "names": new_names,
        }
    else:
        # Keep original config
        new_config = {
            "path": ".",
            "nc": data_config.get("nc"),
            "names": class_names,
        }

    if extract_train:
        new_config["train"] = "images/train"
    if extract_val:
        new_config["val"] = "images/val"

    with open(dst_yaml, "w", encoding="utf-8") as f:
        yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True)

    logger.info("Extraction complete!")
    logger.info(f"Train: {stats['train_images']} images, {stats['train_labels']} labels")
    logger.info(f"Val: {stats['val_images']} images, {stats['val_labels']} labels")
    logger.info(f"Output: {output_dir}")

    return stats


def _extract_split(
    src_img_dir: Path,
    src_label_dir: Path,
    dst_img_dir: Path,
    dst_label_dir: Path,
    class_ids: set[int],
    operation: str,
    filter_labels: bool,
    id_mapping: dict[int, int] | None,
) -> dict[str, int]:
    """
    Extract images and labels from a single split (train or val).

    Args:
        src_img_dir: Source images directory
        src_label_dir: Source labels directory
        dst_img_dir: Destination images directory
        dst_label_dir: Destination labels directory
        class_ids: Set of class IDs to extract
        operation: "copy" or "move"
        filter_labels: Whether to filter label content
        id_mapping: Class ID remapping dict (None if no remapping)

    Returns:
        Dictionary with counts of extracted images and labels
    """
    stats = {"images": 0, "labels": 0}

    # Get all label files
    label_files = list(src_label_dir.glob("*.txt"))

    for label_file in label_files:
        # Read and check if label contains target classes
        contains_class = False
        filtered_lines = []

        try:
            with open(label_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id in class_ids:
                            contains_class = True
                            if filter_labels:
                                # Apply ID remapping if needed
                                if id_mapping:
                                    new_id = id_mapping[class_id]
                                    parts[0] = str(new_id)
                                    filtered_lines.append(" ".join(parts))
                                else:
                                    filtered_lines.append(line)
                        elif not filter_labels:
                            # Keep all lines if not filtering
                            filtered_lines.append(line)
        except (ValueError, IndexError) as e:
            logger.warning(f"Invalid label format in {label_file.name}: {e}")
            continue

        if not contains_class:
            continue

        # Find corresponding image
        base_name = label_file.stem
        src_img = None
        for ext in IMAGE_EXTENSIONS:
            candidate = src_img_dir / f"{base_name}{ext}"
            if candidate.exists():
                src_img = candidate
                break

        if src_img is None:
            logger.warning(f"Image not found for label: {label_file.name}")
            continue

        # Copy or move image
        dst_img = dst_img_dir / src_img.name
        if operation == "copy":
            shutil.copy2(src_img, dst_img)
        else:  # move
            shutil.move(str(src_img), str(dst_img))
        stats["images"] += 1

        # Write label file
        dst_label = dst_label_dir / label_file.name
        if filter_labels:
            # Write filtered content
            with open(dst_label, "w", encoding="utf-8") as f:
                f.write("\n".join(filtered_lines) + "\n")
        else:
            # Copy entire file
            if operation == "copy":
                shutil.copy2(label_file, dst_label)
            else:
                shutil.move(str(label_file), str(dst_label))
        stats["labels"] += 1

    return stats


def extract_images_only(
    input_dir: str | Path,
    output_dir: str | Path,
    class_ids: list[int],
    operation: str = "copy",
    extract_train: bool = True,
    extract_val: bool = True,
) -> dict[str, int]:
    """
    Extract only images (no labels) containing specified classes from dataset.

    Args:
        input_dir: Input YOLO dataset directory
        output_dir: Output directory for extracted images
        class_ids: List of class IDs to extract
        operation: "copy" (default) or "move"
        extract_train: Extract from training set
        extract_val: Extract from validation set

    Returns:
        Dictionary with extraction statistics

    Raises:
        FileNotFoundError: If input directory or data.yaml not found
        ValueError: If operation is invalid or no classes specified

    Examples:
        >>> stats = extract_images_only(
        ...     "./dataset",
        ...     "./images_only",
        ...     class_ids=[0, 2],
        ...     operation="copy"
        ... )
    """
    if not class_ids:
        raise ValueError("At least one class ID must be specified")

    if operation not in ["copy", "move"]:
        raise ValueError(f"Invalid operation: {operation}. Must be 'copy' or 'move'")

    if not extract_train and not extract_val:
        raise ValueError("At least one of extract_train or extract_val must be True")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Read data.yaml to validate class IDs
    yaml_path = input_dir / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found in {input_dir}")

    with open(yaml_path, encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    class_names = data_config.get("names", [])
    if not class_names:
        raise ValueError("No class names found in data.yaml")

    # Validate class IDs
    class_ids_set = set(class_ids)
    for cid in class_ids:
        if cid < 0 or cid >= len(class_names):
            raise ValueError(f"Invalid class ID {cid}. Valid range: 0-{len(class_names) - 1}")

    logger.info(f"Extracting images for class IDs: {sorted(class_ids_set)}")
    logger.info(f"Class names: {[class_names[i] for i in sorted(class_ids_set)]}")
    logger.info(f"Operation: {operation}")

    stats = {"train_images": 0, "val_images": 0}

    # Process training set
    if extract_train:
        src_train_img = input_dir / "images" / "train"
        src_train_label = input_dir / "labels" / "train"

        if src_train_img.exists() and src_train_label.exists():
            dst_train_img = output_dir / "train"
            dst_train_img.mkdir(parents=True, exist_ok=True)

            logger.info("Processing training images...")
            count = _extract_images_split(
                src_train_img,
                src_train_label,
                dst_train_img,
                class_ids_set,
                operation,
            )
            stats["train_images"] = count

    # Process validation set
    if extract_val:
        src_val_img = input_dir / "images" / "val"
        src_val_label = input_dir / "labels" / "val"

        if src_val_img.exists() and src_val_label.exists():
            dst_val_img = output_dir / "val"
            dst_val_img.mkdir(parents=True, exist_ok=True)

            logger.info("Processing validation images...")
            count = _extract_images_split(
                src_val_img, src_val_label, dst_val_img, class_ids_set, operation
            )
            stats["val_images"] = count

    logger.info("Image extraction complete!")
    logger.info(f"Train: {stats['train_images']} images")
    logger.info(f"Val: {stats['val_images']} images")
    logger.info(f"Output: {output_dir}")

    return stats


def _extract_images_split(
    src_img_dir: Path,
    src_label_dir: Path,
    dst_img_dir: Path,
    class_ids: set[int],
    operation: str,
) -> int:
    """Extract images only from a single split."""
    count = 0
    label_files = list(src_label_dir.glob("*.txt"))

    for label_file in label_files:
        # Check if contains target classes
        contains_class = False
        try:
            with open(label_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id in class_ids:
                            contains_class = True
                            break
        except (ValueError, IndexError):
            continue

        if not contains_class:
            continue

        # Find and copy/move image
        base_name = label_file.stem
        for ext in IMAGE_EXTENSIONS:
            src_img = src_img_dir / f"{base_name}{ext}"
            if src_img.exists():
                dst_img = dst_img_dir / src_img.name
                if operation == "copy":
                    shutil.copy2(src_img, dst_img)
                else:
                    shutil.move(str(src_img), str(dst_img))
                count += 1
                break

    return count


def extract_labels_only(
    input_dir: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    extract_train: bool = True,
    extract_val: bool = True,
) -> dict[str, int]:
    """
    Extract labels corresponding to images in a given directory.

    Given a folder of images, find and extract the corresponding label files
    from a YOLO dataset.

    Args:
        input_dir: Input YOLO dataset directory (containing labels)
        image_dir: Directory containing images to match
        output_dir: Output directory for extracted labels
        extract_train: Search in training set labels
        extract_val: Search in validation set labels

    Returns:
        Dictionary with extraction statistics

    Raises:
        FileNotFoundError: If input directories not found
        ValueError: If no splits selected for extraction

    Examples:
        >>> stats = extract_labels_only(
        ...     "./dataset",
        ...     "./my_images",
        ...     "./extracted_labels"
        ... )
        >>> print(f"Extracted {stats['total_labels']} labels")
    """
    if not extract_train and not extract_val:
        raise ValueError("At least one of extract_train or extract_val must be True")

    input_dir = Path(input_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Get all image base names (without extension)
    image_files = [f for f in image_dir.rglob("*") if f.suffix in IMAGE_EXTENSIONS]
    image_base_names = {img.stem for img in image_files}

    logger.info(f"Found {len(image_base_names)} images in {image_dir}")

    stats = {"train_labels": 0, "val_labels": 0, "total_labels": 0}

    # Process training labels
    if extract_train:
        src_train_label = input_dir / "labels" / "train"
        if src_train_label.exists():
            dst_train_label = output_dir / "train"
            dst_train_label.mkdir(parents=True, exist_ok=True)

            logger.info("Extracting training labels...")
            count = _extract_labels_split(src_train_label, dst_train_label, image_base_names)
            stats["train_labels"] = count
            stats["total_labels"] += count

    # Process validation labels
    if extract_val:
        src_val_label = input_dir / "labels" / "val"
        if src_val_label.exists():
            dst_val_label = output_dir / "val"
            dst_val_label.mkdir(parents=True, exist_ok=True)

            logger.info("Extracting validation labels...")
            count = _extract_labels_split(src_val_label, dst_val_label, image_base_names)
            stats["val_labels"] = count
            stats["total_labels"] += count

    logger.info("Label extraction complete!")
    logger.info(f"Train: {stats['train_labels']} labels")
    logger.info(f"Val: {stats['val_labels']} labels")
    logger.info(f"Total: {stats['total_labels']} labels")
    logger.info(f"Output: {output_dir}")

    return stats


def _extract_labels_split(
    src_label_dir: Path, dst_label_dir: Path, image_base_names: set[str]
) -> int:
    """Extract labels for a single split."""
    count = 0
    label_files = list(src_label_dir.glob("*.txt"))

    for label_file in label_files:
        if label_file.stem in image_base_names:
            dst_label = dst_label_dir / label_file.name
            shutil.copy2(label_file, dst_label)
            count += 1

    return count

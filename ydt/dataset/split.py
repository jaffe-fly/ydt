"""
Dataset splitting and merging operations.

Provides functions for splitting datasets into train/val sets with
balanced class distribution, merging multiple datasets, and more.
"""

import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import yaml

from ydt.core.logger import get_logger

logger = get_logger(__name__)


def split_dataset(
    data_yaml_path: Union[str, Path],
    output_dir: Union[str, Path],
    train_ratio: float = 0.8,
    balance_rotation: bool = False,
    balance_classes: bool = True,
) -> Dict[str, int]:
    """
    Split dataset into train and validation sets with balanced distribution.

    This function ensures each class has representation in both train and val sets.
    Optionally balances by rotation angles (useful for augmented datasets).

    Args:
        data_yaml_path: Path to dataset YAML file
        output_dir: Output directory for split dataset
        train_ratio: Ratio of training data (0.0 to 1.0)
        balance_rotation: If True, balance rotation angles (looks for 'rot_' prefix)
        balance_classes: If True, ensure all classes are represented in both sets

    Returns:
        Dictionary with statistics (train_count, val_count, etc.)

    Raises:
        FileNotFoundError: If YAML file or source directories don't exist
        ValueError: If train_ratio is not in valid range

    Examples:
        >>> stats = split_dataset(
        ...     "./dataset/data.yaml",
        ...     "./dataset_split",
        ...     train_ratio=0.8
        ... )
        >>> print(f"Train: {stats['train_count']}, Val: {stats['val_count']}")
    """
    data_yaml_path = Path(data_yaml_path)
    output_dir = Path(output_dir)

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {data_yaml_path}")

    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    # Read YAML config
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    # Get source directories
    src_root_dir = data_yaml_path.parent
    src_train_dir = src_root_dir / "images" / "train"
    src_labels_dir = src_root_dir / "labels" / "train"

    if not src_train_dir.exists():
        raise FileNotFoundError(f"Source train directory not found: {src_train_dir}")

    # Create output directories
    dst_train_img_dir = output_dir / "images" / "train"
    dst_train_label_dir = output_dir / "labels" / "train"
    dst_val_img_dir = output_dir / "images" / "val"
    dst_val_label_dir = output_dir / "labels" / "val"

    for d in [dst_train_img_dir, dst_train_label_dir, dst_val_img_dir, dst_val_label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = (".jpg", ".png", ".jpeg", ".PNG", ".JPG")
    image_files = [f.name for f in src_train_dir.iterdir() if f.suffix in image_extensions]

    logger.info(f"Found {len(image_files)} images")

    # Collect class and rotation information
    class_samples: Dict[int, List[str]] = {}
    rotation_samples: Dict[int, List[str]] = {}
    image_classes: Dict[str, Set[int]] = {}
    image_rotation: Dict[str, int] = {}

    rotation_pattern = re.compile(r"rot_(\d+)")

    for img_file in image_files:
        label_file = Path(img_file).stem + ".txt"
        label_path = src_labels_dir / label_file

        # Extract rotation angle if present
        if balance_rotation:
            rotation_match = rotation_pattern.search(img_file)
            rotation_angle = int(rotation_match.group(1)) if rotation_match else 0
            image_rotation[img_file] = rotation_angle

            if rotation_angle not in rotation_samples:
                rotation_samples[rotation_angle] = []
            rotation_samples[rotation_angle].append(img_file)

        # Read labels to get classes
        if label_path.exists():
            image_classes[img_file] = set()
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        class_id = int(float(line.split()[0]))
                        image_classes[img_file].add(class_id)

                        if class_id not in class_samples:
                            class_samples[class_id] = []
                        class_samples[class_id].append(img_file)
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid label in {label_file}: {line}")
                        continue

    # Calculate validation set size
    total_val_count = int(len(image_files) * (1 - train_ratio))
    logger.info(f"Target validation set size: {total_val_count}")

    # Define scoring function for balanced split
    def get_image_score(img: str) -> Tuple:
        class_count = len(image_classes.get(img, set()))
        if balance_rotation:
            rotation_count = len(
                [x for x in rotation_samples.get(image_rotation[img], []) if x != img]
            )
            return (class_count, rotation_count)
        return (class_count,)

    # Sort images by score (prefer images with fewer classes)
    sorted_images = sorted(image_files, key=get_image_score)

    # Initialize validation set
    val_images: Set[str] = set()
    class_val_samples = {class_id: set() for class_id in class_samples.keys()}

    if balance_rotation:
        rotation_val_samples = {angle: set() for angle in rotation_samples.keys()}

    # Phase 1: Ensure each class and rotation angle has at least one sample
    if balance_classes:
        for img in sorted_images:
            if len(val_images) >= total_val_count:
                break

            img_classes = image_classes.get(img, set())
            needs_sample = False

            # Check if any class lacks validation samples
            for class_id in img_classes:
                if not class_val_samples[class_id]:
                    needs_sample = True
                    break

            # Check rotation balance if enabled
            if balance_rotation and not needs_sample:
                rotation_angle = image_rotation[img]
                if not rotation_val_samples[rotation_angle]:
                    needs_sample = True

            if needs_sample:
                val_images.add(img)
                for class_id in img_classes:
                    class_val_samples[class_id].add(img)
                if balance_rotation:
                    rotation_val_samples[image_rotation[img]].add(img)

    # Phase 2: Fill remaining validation set slots
    remaining_images = sorted_images.copy()
    random.shuffle(remaining_images)

    for img in remaining_images:
        if len(val_images) >= total_val_count:
            break
        if img not in val_images:
            val_images.add(img)
            img_classes = image_classes.get(img, set())
            for class_id in img_classes:
                class_val_samples[class_id].add(img)
            if balance_rotation:
                rotation_val_samples[image_rotation[img]].add(img)

    # Copy files to output directories
    train_count = 0
    val_count = 0

    for img_file in image_files:
        src_img = src_train_dir / img_file
        src_label = src_labels_dir / (Path(img_file).stem + ".txt")

        if img_file in val_images:
            dst_img = dst_val_img_dir / img_file
            dst_label = dst_val_label_dir / (Path(img_file).stem + ".txt")
            val_count += 1
        else:
            dst_img = dst_train_img_dir / img_file
            dst_label = dst_train_label_dir / (Path(img_file).stem + ".txt")
            train_count += 1

        shutil.copy2(src_img, dst_img)
        if src_label.exists():
            shutil.copy2(src_label, dst_label)

    # Create new YAML file
    new_yaml_path = output_dir / "data.yaml"
    data_config["train"] = str(dst_train_img_dir)
    data_config["val"] = str(dst_val_img_dir)

    with open(new_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    # Log statistics
    logger.info("Dataset split complete!")
    logger.info(f"Training set: {train_count} images")
    logger.info(f"Validation set: {val_count} images")

    # Log class distribution
    logger.info("\nClass distribution:")
    for class_id in sorted(class_val_samples.keys()):
        val_count_cls = len(class_val_samples[class_id])
        total_count = len(class_samples[class_id])
        ratio = val_count_cls / total_count if total_count > 0 else 0
        logger.info(f"  Class {class_id}: {val_count_cls}/{total_count} ({ratio:.1%})")

    # Log rotation distribution if enabled
    if balance_rotation:
        logger.info("\nRotation angle distribution:")
        for angle in sorted(rotation_val_samples.keys()):
            val_count_rot = len(rotation_val_samples[angle])
            total_count = len(rotation_samples[angle])
            ratio = val_count_rot / total_count if total_count > 0 else 0
            logger.info(f"  {angle}°: {val_count_rot}/{total_count} ({ratio:.1%})")

    return {
        "train_count": train_count,
        "val_count": val_count,
        "output_dir": str(output_dir),
    }


def merge_datasets(
    dataset_dirs: List[Union[str, Path]],
    output_dir: Union[str, Path],
    merge_train: bool = True,
    merge_val: bool = True,
    handle_duplicates: str = "rename",
) -> Dict[str, int]:
    """
    Merge multiple datasets into a single dataset.

    Args:
        dataset_dirs: List of source dataset directories
        output_dir: Output directory for merged dataset
        merge_train: If True, merge training sets
        merge_val: If True, merge validation sets
        handle_duplicates: How to handle duplicate filenames:
                          - 'rename': Rename duplicates with counter suffix
                          - 'skip': Skip duplicate files
                          - 'overwrite': Overwrite with latest

    Returns:
        Dictionary with merge statistics

    Raises:
        ValueError: If neither merge_train nor merge_val is True
        FileNotFoundError: If source directories don't exist

    Examples:
        >>> stats = merge_datasets(
        ...     ["./dataset1", "./dataset2", "./dataset3"],
        ...     "./merged_dataset",
        ...     merge_train=True,
        ...     merge_val=True
        ... )
        >>> print(f"Merged {stats['train_images']} training images")
    """
    if not merge_train and not merge_val:
        raise ValueError("At least one of merge_train or merge_val must be True")

    output_dir = Path(output_dir)
    dataset_dirs = [Path(d) for d in dataset_dirs]

    # Verify all source directories exist
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Create output directories
    if merge_train:
        dst_train_img_dir = output_dir / "images" / "train"
        dst_train_label_dir = output_dir / "labels" / "train"
        dst_train_img_dir.mkdir(parents=True, exist_ok=True)
        dst_train_label_dir.mkdir(parents=True, exist_ok=True)

    if merge_val:
        dst_val_img_dir = output_dir / "images" / "val"
        dst_val_label_dir = output_dir / "labels" / "val"
        dst_val_img_dir.mkdir(parents=True, exist_ok=True)
        dst_val_label_dir.mkdir(parents=True, exist_ok=True)

    # Track processed files and statistics
    processed_names = {"train": set(), "val": set()}
    stats = {"train_images": 0, "train_labels": 0, "val_images": 0, "val_labels": 0}

    image_extensions = (".jpg", ".png", ".jpeg", ".PNG", ".JPG")

    for dataset_dir in dataset_dirs:
        logger.info(f"Processing dataset: {dataset_dir}")

        # Process training set
        if merge_train:
            src_train_img_dir = dataset_dir / "images" / "train"
            src_train_label_dir = dataset_dir / "labels" / "train"

            if src_train_img_dir.exists():
                logger.info("Merging training set...")
                train_images = [
                    f for f in src_train_img_dir.iterdir()
                    if f.suffix in image_extensions
                ]

                for img_file in train_images:
                    base_name = img_file.stem
                    ext = img_file.suffix

                    src_img = img_file
                    src_label = src_train_label_dir / f"{base_name}.txt"

                    # Handle duplicates
                    new_base_name = base_name
                    if handle_duplicates == "rename":
                        counter = 1
                        while f"{new_base_name}{ext}" in processed_names["train"]:
                            new_base_name = f"{base_name}_{counter}"
                            counter += 1
                    elif handle_duplicates == "skip":
                        if f"{base_name}{ext}" in processed_names["train"]:
                            logger.debug(f"Skipping duplicate: {img_file.name}")
                            continue

                    # Copy files
                    dst_img = dst_train_img_dir / f"{new_base_name}{ext}"
                    dst_label = dst_train_label_dir / f"{new_base_name}.txt"

                    shutil.copy2(src_img, dst_img)
                    stats["train_images"] += 1

                    if src_label.exists():
                        shutil.copy2(src_label, dst_label)
                        stats["train_labels"] += 1
                    else:
                        dst_label.touch()  # Create empty label file

                    processed_names["train"].add(f"{new_base_name}{ext}")
            else:
                logger.warning(f"Training directory not found: {src_train_img_dir}")

        # Process validation set
        if merge_val:
            src_val_img_dir = dataset_dir / "images" / "val"
            src_val_label_dir = dataset_dir / "labels" / "val"

            if src_val_img_dir.exists():
                logger.info("Merging validation set...")
                val_images = [
                    f for f in src_val_img_dir.iterdir()
                    if f.suffix in image_extensions
                ]

                for img_file in val_images:
                    base_name = img_file.stem
                    ext = img_file.suffix

                    src_img = img_file
                    src_label = src_val_label_dir / f"{base_name}.txt"

                    # Handle duplicates
                    new_base_name = base_name
                    if handle_duplicates == "rename":
                        counter = 1
                        while f"{new_base_name}{ext}" in processed_names["val"]:
                            new_base_name = f"{base_name}_{counter}"
                            counter += 1
                    elif handle_duplicates == "skip":
                        if f"{base_name}{ext}" in processed_names["val"]:
                            logger.debug(f"Skipping duplicate: {img_file.name}")
                            continue

                    # Copy files
                    dst_img = dst_val_img_dir / f"{new_base_name}{ext}"
                    dst_label = dst_val_label_dir / f"{new_base_name}.txt"

                    shutil.copy2(src_img, dst_img)
                    stats["val_images"] += 1

                    if src_label.exists():
                        shutil.copy2(src_label, dst_label)
                        stats["val_labels"] += 1
                    else:
                        dst_label.touch()

                    processed_names["val"].add(f"{new_base_name}{ext}")
            else:
                logger.warning(f"Validation directory not found: {src_val_img_dir}")

    # Copy YAML file from first dataset
    yaml_path = dataset_dirs[0] / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)

        if merge_train:
            data_config["train"] = str(dst_train_img_dir)
        if merge_val:
            data_config["val"] = str(dst_val_img_dir)

        new_yaml_path = output_dir / "data.yaml"
        with open(new_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    logger.info("Dataset merge complete!")
    if merge_train:
        logger.info(f"Training set: {stats['train_images']} images, {stats['train_labels']} labels")
    if merge_val:
        logger.info(f"Validation set: {stats['val_images']} images, {stats['val_labels']} labels")
    logger.info(f"Output directory: {output_dir}")

    return stats

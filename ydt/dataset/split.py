"""
Dataset splitting and merging operations.

Provides functions for splitting datasets into train/val sets with
balanced class distribution, merging multiple datasets, and more.
"""

import random
import re
import shutil
from pathlib import Path

import yaml

from ydt.core import IMAGE_EXTENSIONS
from ydt.core.logger import get_logger

logger = get_logger(__name__)


def split_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    balance_rotation: bool = False,
    balance_classes: bool = True,
) -> dict[str, int]:
    """
    Split dataset into train and validation sets with balanced distribution.

    This function ensures each class has representation in both train and val sets.
    Optionally balances by rotation angles (useful for augmented datasets).

    Args:
        input_path: Path to dataset YAML file or dataset directory (containing data.yaml)
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
        >>> # Using YAML file path
        >>> stats = split_dataset(
        ...     "./dataset/data.yaml",
        ...     "./dataset_split",
        ...     train_ratio=0.8
        ... )
        >>> print(f"Train: {stats['train_count']}, Val: {stats['val_count']}")

        >>> # Using dataset directory
        >>> stats = split_dataset(
        ...     "./dataset",
        ...     "./dataset_split",
        ...     train_ratio=0.8
        ... )
    """
    output_dir = Path(output_dir)
    input_path = Path(input_path)

    # Determine data_yaml_path based on input type
    if input_path.is_file() and input_path.suffix in [".yaml", ".yml"]:
        # Input is a YAML file
        data_yaml_path = input_path
    elif input_path.is_dir():
        # Input is a directory, look for data.yaml
        data_yaml_path = input_path / "data.yaml"
        if not data_yaml_path.exists():
            data_yaml_path = input_path / "data.yml"
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"No data.yaml or data.yml found in directory: {input_path}")
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {data_yaml_path}")

    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    # Read YAML config
    with open(data_yaml_path, encoding="utf-8") as f:
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
    image_files = [f.name for f in src_train_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS]

    logger.info(f"Found {len(image_files)} images")

    # Collect class and rotation information
    #
    # NOTE:
    # - Unlabeled images (no txt file OR empty/invalid txt) do NOT participate in splitting
    #   and are always kept in train.
    # - Validation selection is based on per-class INSTANCE ratio (not image count).
    class_instance_counts: dict[int, int] = {}
    image_class_counts: dict[str, dict[int, int]] = {}
    image_classes: dict[str, set[int]] = {}
    labeled_images: list[str] = []
    unlabeled_images: list[str] = []
    rotation_samples: dict[int, list[str]] = {}
    image_rotation: dict[str, int] = {}

    rotation_pattern = re.compile(r"rot_(\d+)")

    for img_file in image_files:
        label_file = Path(img_file).stem + ".txt"
        label_path = src_labels_dir / label_file

        # Extract rotation angle if present
        if balance_rotation:
            rotation_match = rotation_pattern.search(img_file)
            rotation_angle = int(rotation_match.group(1)) if rotation_match else 0
            image_rotation[img_file] = rotation_angle

        # Read labels to get per-image per-class instance counts
        if not label_path.exists():
            unlabeled_images.append(img_file)
            continue

        per_image_counts: dict[int, int] = {}
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id = int(float(line.split()[0]))
                except (ValueError, IndexError):
                    logger.warning(f"Invalid label in {label_file}: {line}")
                    continue

                per_image_counts[class_id] = per_image_counts.get(class_id, 0) + 1
                class_instance_counts[class_id] = class_instance_counts.get(class_id, 0) + 1

        # Empty/invalid label file is treated as unlabeled background
        if not per_image_counts:
            unlabeled_images.append(img_file)
            continue

        image_class_counts[img_file] = per_image_counts
        image_classes[img_file] = set(per_image_counts.keys())
        labeled_images.append(img_file)

        if balance_rotation:
            rotation_angle = image_rotation[img_file]
            if rotation_angle not in rotation_samples:
                rotation_samples[rotation_angle] = []
            rotation_samples[rotation_angle].append(img_file)

    logger.info(
        f"Labeled images: {len(labeled_images)}; Unlabeled/empty-label images kept in train: {len(unlabeled_images)}"
    )

    val_ratio = 1 - train_ratio
    class_target_instances: dict[int, int] = {}
    for class_id, total_instances in class_instance_counts.items():
        target = int(round(total_instances * val_ratio))
        if balance_classes and total_instances > 0 and val_ratio > 0:
            target = max(1, target)
        class_target_instances[class_id] = min(target, total_instances)

    logger.info("Target validation instance counts by class:")
    for class_id in sorted(class_target_instances.keys()):
        logger.info(
            f"  Class {class_id}: {class_target_instances[class_id]}/{class_instance_counts[class_id]}"
        )

    # Define scoring function for balanced split
    def get_image_score(
        img: str, remaining: dict[int, int], val_angles: set[int]
    ) -> tuple[int, int, int, int]:
        counts = image_class_counts.get(img, {})
        gain = sum(min(cnt, remaining.get(class_id, 0)) for class_id, cnt in counts.items())
        overflow = sum(max(0, cnt - remaining.get(class_id, 0)) for class_id, cnt in counts.items())
        total_instances = sum(counts.values())
        rotation_bonus = 0
        if balance_rotation:
            rotation_bonus = 1 if image_rotation.get(img, 0) not in val_angles else 0
        # Higher gain/rotation_bonus is better; lower overflow/total_instances is better
        return (gain, rotation_bonus, -overflow, -total_instances)

    # Initialize validation set
    val_images: set[str] = set()
    val_instance_counts = dict.fromkeys(class_instance_counts.keys(), 0)
    remaining_needed = class_target_instances.copy()
    val_angles: set[int] = set()

    if val_ratio > 0 and labeled_images and any(v > 0 for v in remaining_needed.values()):
        # Greedy selection: pick images that best satisfy remaining per-class instance targets
        candidate_images = labeled_images.copy()
        random.shuffle(candidate_images)

        while any(v > 0 for v in remaining_needed.values()):
            best_img: str | None = None
            best_score: tuple[int, int, int, int] = (0, 0, 0, 0)

            for img in candidate_images:
                if img in val_images:
                    continue
                score = get_image_score(img, remaining_needed, val_angles)
                if score > best_score:
                    best_score = score
                    best_img = img

            # No further progress possible
            if best_img is None or best_score[0] <= 0:
                break

            val_images.add(best_img)
            if balance_rotation:
                val_angles.add(image_rotation.get(best_img, 0))

            for class_id, cnt in image_class_counts[best_img].items():
                if class_id in val_instance_counts:
                    val_instance_counts[class_id] += cnt
                if class_id in remaining_needed:
                    remaining_needed[class_id] = max(0, remaining_needed[class_id] - cnt)

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
    logger.info(f"Training set: {train_count} images, Validation set: {val_count} images")

    # Log class distribution (instance-based)
    logger.info("\nClass distribution (instances):")
    for class_id in sorted(class_instance_counts.keys()):
        val_inst = val_instance_counts.get(class_id, 0)
        total_inst = class_instance_counts[class_id]
        ratio = val_inst / total_inst if total_inst > 0 else 0
        logger.info(f"  Class {class_id}: {val_inst}/{total_inst} ({ratio:.1%})")

    # Log rotation distribution if enabled (image-based)
    if balance_rotation and rotation_samples:
        logger.info("\nRotation angle distribution (val images):")
        rotation_val_counts: dict[int, int] = dict.fromkeys(rotation_samples.keys(), 0)
        for img in val_images:
            angle = image_rotation.get(img, 0)
            if angle in rotation_val_counts:
                rotation_val_counts[angle] += 1
        for angle in sorted(rotation_val_counts.keys()):
            total_count = len(rotation_samples.get(angle, []))
            val_count_rot = rotation_val_counts[angle]
            ratio = val_count_rot / total_count if total_count > 0 else 0
            logger.info(f"  {angle}°: {val_count_rot}/{total_count} ({ratio:.1%})")

    return {
        "train_count": train_count,
        "val_count": val_count,
        "output_dir": str(output_dir),
    }


def merge_datasets(
    dataset_dirs: list[str | Path],
    output_dir: str | Path,
    merge_train: bool = True,
    merge_val: bool = True,
    handle_duplicates: str = "rename",
) -> dict[str, int]:
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

    for dataset_dir in dataset_dirs:
        logger.info(f"Processing dataset: {dataset_dir}")

        # Process training set
        if merge_train:
            src_train_img_dir = dataset_dir / "images" / "train"
            src_train_label_dir = dataset_dir / "labels" / "train"

            if src_train_img_dir.exists():
                logger.info("Merging training set...")
                train_images = [
                    f for f in src_train_img_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS
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
                val_images = [f for f in src_val_img_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS]

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
        with open(yaml_path, encoding="utf-8") as f:
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

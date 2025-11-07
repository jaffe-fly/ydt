"""
Common utility functions for dataset processing
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
from .logger import get_logger
from .formats import FormatType, detect_format

logger = get_logger(__name__)


def scan_empty_labels(directory: Union[str, Path]) -> List[Path]:
    """
    Scan directory for empty label files

    Args:
        directory: Directory path to scan

    Returns:
        List of empty label file paths

    Examples:
        >>> empty_files = scan_empty_labels("dataset/labels/train")
        >>> print(f"Found {len(empty_files)} empty label files")
    """
    directory = Path(directory)

    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []

    empty_files = []

    for label_file in directory.rglob("*.txt"):
        if label_file.stat().st_size == 0:
            empty_files.append(label_file)

    if empty_files:
        logger.info(f"Found {len(empty_files)} empty label files")
        for file_path in empty_files:
            logger.debug(f"  - {file_path}")
    else:
        logger.info("No empty label files found")

    return empty_files


def update_label_classes(
    labels_dir: Union[str, Path],
    class_mapping: Dict[int, int],
    format_type: Optional[FormatType] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Update class IDs in label files according to mapping

    Args:
        labels_dir: Directory containing label files
        class_mapping: Old class ID to new class ID mapping
        format_type: Label format type ('bbox' or 'obb'), auto-detected if None
        dry_run: If True, only show what would be changed without modifying files

    Returns:
        Dictionary with statistics: {'updated': count, 'errors': count, 'skipped': count}

    Examples:
        >>> mapping = {0: 2, 1: 0, 2: 1}  # Remap classes
        >>> stats = update_label_classes("labels/train", mapping)
        >>> print(f"Updated {stats['updated']} files")
    """
    labels_dir = Path(labels_dir)

    if not labels_dir.exists():
        logger.error(f"Labels directory does not exist: {labels_dir}")
        return {"updated": 0, "errors": 0, "skipped": 0}

    stats = {"updated": 0, "errors": 0, "skipped": 0}
    label_files = list(labels_dir.glob("*.txt"))

    logger.info(f"Processing {len(label_files)} label files...")
    if dry_run:
        logger.info("DRY RUN - No files will be modified")

    for label_file in label_files:
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                stats["skipped"] += 1
                continue

            # Auto-detect format if not specified
            file_format = format_type or detect_format(label_line=lines[0].strip())

            new_lines = []
            file_modified = False

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    new_lines.append(line)
                    continue

                try:
                    old_class = int(parts[0])

                    if old_class in class_mapping:
                        new_class = class_mapping[old_class]
                        parts[0] = str(new_class)
                        new_lines.append(" ".join(parts) + "\n")
                        file_modified = True
                        logger.debug(f"{label_file.name}: class {old_class} -> {new_class}")
                    else:
                        new_lines.append(line)
                        logger.warning(
                            f"{label_file.name}: Unknown class {old_class}, keeping unchanged"
                        )

                except ValueError:
                    new_lines.append(line)
                    logger.warning(f"{label_file.name}: Invalid class ID in line: {line.strip()}")

            # Write back if modified
            if file_modified and not dry_run:
                with open(label_file, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                stats["updated"] += 1
            elif file_modified and dry_run:
                logger.info(f"Would update: {label_file.name}")
                stats["updated"] += 1
            else:
                stats["skipped"] += 1

        except Exception as e:
            logger.error(f"Error processing {label_file.name}: {str(e)}")
            stats["errors"] += 1

    logger.info(
        f"Completed: {stats['updated']} updated, {stats['skipped']} skipped, {stats['errors']} errors"
    )
    return stats


def generate_class_mapping(
    old_yaml: Union[str, Path], new_yaml: Union[str, Path]
) -> Optional[Dict[int, int]]:
    """
    Generate class ID mapping from old dataset to new dataset

    Args:
        old_yaml: Path to old data.yaml
        new_yaml: Path to new data.yaml

    Returns:
        Mapping dictionary {old_id: new_id} or None if error

    Examples:
        >>> mapping = generate_class_mapping("old/data.yaml", "new/data.yaml")
        >>> if mapping:
        ...     print(f"Generated mapping for {len(mapping)} classes")
    """
    try:
        with open(old_yaml, "r", encoding="utf-8") as f:
            old_data = yaml.safe_load(f)
        with open(new_yaml, "r", encoding="utf-8") as f:
            new_data = yaml.safe_load(f)

        old_names = old_data.get("names", {})
        new_names = new_data.get("names", {})

        if not old_names or not new_names:
            logger.error("Missing 'names' field in YAML files")
            return None

        # Convert to dict if list
        if isinstance(old_names, list):
            old_names = {i: name for i, name in enumerate(old_names)}
        if isinstance(new_names, list):
            new_names = {i: name for i, name in enumerate(new_names)}

        # Create mapping
        mapping = {}
        for old_id, old_name in old_names.items():
            for new_id, new_name in new_names.items():
                if old_name == new_name:
                    mapping[int(old_id)] = int(new_id)
                    logger.debug(f"Class '{old_name}': {old_id} -> {new_id}")
                    break
            else:
                logger.warning(f"Class '{old_name}' not found in new dataset")

        logger.info(f"Generated mapping for {len(mapping)} classes")
        return mapping

    except Exception as e:
        logger.error(f"Error generating class mapping: {str(e)}")
        return None


def find_label_files(directory: Union[str, Path]) -> List[Path]:
    """
    Find all label files in directory.

    Args:
        directory: Directory path to scan

    Returns:
        List of label file paths

    Examples:
        >>> labels = find_label_files("dataset/labels/train")
        >>> print(f"Found {len(labels)} label files")
    """
    directory = Path(directory)

    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []

    label_files = list(directory.rglob("*.txt"))
    logger.info(f"Found {len(label_files)} label files in {directory}")

    return label_files


def count_class_distribution(
    labels_dir: Union[str, Path], format_type: Optional[FormatType] = None
) -> Optional[Dict[int, int]]:
    """
    Count class distribution in label directory

    Args:
        labels_dir: Directory containing label files
        format_type: Label format type, auto-detected if None

    Returns:
        Dictionary {class_id: count} or None if error

    Examples:
        >>> dist = count_class_distribution("labels/train")
        >>> for class_id, count in sorted(dist.items()):
        ...     print(f"Class {class_id}: {count} instances")
    """
    labels_dir = Path(labels_dir)

    if not labels_dir.exists():
        logger.error(f"Labels directory does not exist: {labels_dir}")
        return None

    class_counts = {}
    label_files = list(labels_dir.glob("*.txt"))

    for label_file in label_files:
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if parts:
                    try:
                        class_id = int(parts[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    except ValueError:
                        logger.warning(f"Invalid class ID in {label_file.name}: {parts[0]}")

        except Exception as e:
            logger.error(f"Error processing {label_file.name}: {str(e)}")

    # Sort by class ID
    class_counts = dict(sorted(class_counts.items()))

    # Log statistics
    total_instances = sum(class_counts.values())
    logger.info(f"\n{'=' * 50}")
    logger.info("Class Distribution Statistics")
    logger.info(f"{'=' * 50}")
    logger.info(f"{'Class ID':<15} {'Count':<10} {'Percentage':<10}")
    logger.info(f"{'-' * 50}")

    for class_id, count in class_counts.items():
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        logger.info(f"{class_id:<15} {count:<10} {percentage:>6.2f}%")

    logger.info(f"{'-' * 50}")
    logger.info(f"Total instances: {total_instances}")
    logger.info(f"Total classes: {len(class_counts)}")
    logger.info(f"{'=' * 50}\n")

    return class_counts


def read_yaml_config(yaml_path: Union[str, Path]) -> Optional[Dict]:
    """
    Read YAML configuration file

    Args:
        yaml_path: Path to YAML file

    Returns:
        Dictionary with configuration or None if error
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error reading YAML file {yaml_path}: {str(e)}")
        return None


def write_yaml_config(yaml_path: Union[str, Path], data: Dict) -> bool:
    """
    Write YAML configuration file

    Args:
        yaml_path: Path to YAML file
        data: Dictionary to write

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        logger.error(f"Error writing YAML file {yaml_path}: {str(e)}")
        return False

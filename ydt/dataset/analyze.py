"""
Dataset analysis tools.

Provides utilities for analyzing YOLO format datasets including class distribution,
instance counting, and dataset statistics.
"""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from ydt.core import IMAGE_EXTENSIONS
from ydt.core.formats import detect_format
from ydt.core.logger import get_logger

logger = get_logger(__name__)


def count_labels(
    dataset_path: str | Path,
    split: str = "train",
    show_details: bool = True,
) -> dict[int, int]:
    """
    Count instances of each class in dataset.

    Args:
        dataset_path: Path to dataset directory containing data.yaml
        split: Which split to analyze ("train", "val", or "both")
        show_details: If True, print detailed statistics

    Returns:
        Dictionary mapping class_id to instance count

    Raises:
        FileNotFoundError: If dataset path or data.yaml not found
        ValueError: If split is invalid

    Examples:
        >>> # Count instances in training set
        >>> counts = count_labels("./dataset", split="train")
        >>> print(f"Class 0 has {counts[0]} instances")

        >>> # Count instances in both train and val
        >>> counts = count_labels("./dataset", split="both")
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Validate split parameter
    valid_splits = ["train", "val", "both"]
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")

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

    # Determine which splits to analyze
    splits_to_analyze = []
    if split == "both":
        splits_to_analyze = ["train", "val"]
    else:
        splits_to_analyze = [split]

    # Count instances
    total_counts: dict[int, int] = {}
    split_counts: dict[str, dict[int, int]] = {}

    for current_split in splits_to_analyze:
        label_dir = dataset_path / "labels" / current_split

        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue

        split_counts[current_split] = {}

        # Read all label files
        label_files = list(label_dir.glob("*.txt"))
        logger.info(f"Analyzing {len(label_files)} label files in {current_split} set")

        for label_file in label_files:
            try:
                with open(label_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            continue

                        # Get class ID
                        class_id = int(float(parts[0]))

                        # Update counts
                        split_counts[current_split][class_id] = (
                            split_counts[current_split].get(class_id, 0) + 1
                        )
                        total_counts[class_id] = total_counts.get(class_id, 0) + 1

            except Exception as e:
                logger.warning(f"Error reading {label_file}: {e}")
                continue

    # Print statistics if requested
    if show_details:
        print("\n" + "=" * 70)
        print("数据集类别统计")
        print("=" * 70)

        if split == "both":
            # Show per-split statistics
            for current_split in splits_to_analyze:
                if current_split not in split_counts:
                    continue

                print(f"\n{current_split.upper()} 集合:")
                print("-" * 70)

                counts = split_counts[current_split]
                if counts:
                    for class_id in sorted(counts.keys()):
                        class_name = names.get(class_id, f"class_{class_id}")
                        count = counts[class_id]
                        print(f"  类别 {class_id:2d} ({class_name:20s}): {count:6d} 个实例")
                    print(f"  小计: {sum(counts.values())} 个实例")
                else:
                    print("  无数据")

            # Show total statistics
            print("\n总计:")
            print("-" * 70)

        if total_counts:
            for class_id in sorted(total_counts.keys()):
                class_name = names.get(class_id, f"class_{class_id}")
                count = total_counts[class_id]

                # Calculate split distribution if analyzing both
                if split == "both":
                    train_count = split_counts.get("train", {}).get(class_id, 0)
                    val_count = split_counts.get("val", {}).get(class_id, 0)
                    print(
                        f"  类别 {class_id:2d} ({class_name:20s}): {count:6d} 个实例 "
                        f"(train: {train_count:5d}, val: {val_count:5d})"
                    )
                else:
                    print(f"  类别 {class_id:2d} ({class_name:20s}): {count:6d} 个实例")

            total_instances = sum(total_counts.values())
            print("-" * 70)
            print(f"总计: {total_instances} 个实例")
        else:
            print("  无数据")

        print("=" * 70 + "\n")

    return total_counts


def analyze_dataset(
    dataset_path: str | Path,
    split: str = "train",
    show_details: bool = True,
) -> dict:
    """
    Comprehensive dataset analysis including class distribution and format detection.

    Args:
        dataset_path: Path to dataset directory
        split: Which split to analyze ("train", "val", or "both")
        show_details: If True, print detailed statistics

    Returns:
        Dictionary containing analysis results including:
        - class_counts: Number of instances per class
        - total_instances: Total number of annotations
        - num_classes: Number of unique classes
        - format: Detected annotation format (obb/bbox)
        - total_images: Total number of images
        - image_counts: Number of images per split
        - resolution_distribution: Image resolution distribution
        - class_sizes: Pixel size and ratio statistics per class

    Examples:
        >>> analysis = analyze_dataset("./dataset", split="both")
        >>> print(f"Total images: {analysis['total_images']}")
        >>> print(f"Format: {analysis['format']}")
    """
    dataset_path = Path(dataset_path)

    # Count labels
    class_counts = count_labels(dataset_path, split=split, show_details=False)

    # Read data.yaml for class names
    yaml_path = dataset_path / "data.yaml"
    class_names = {}
    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            names = data.get("names", {})
            if isinstance(names, list):
                class_names = dict(enumerate(names))
            else:
                class_names = names

    # Determine format
    splits_to_check = ["train", "val"] if split == "both" else [split]
    label_dir = dataset_path / "labels" / ("train" if split != "val" else "val")
    detected_format = None

    if label_dir.exists():
        for label_file in label_dir.glob("*.txt"):
            try:
                detected_format = detect_format(label_file=str(label_file))
                break
            except Exception:
                continue

    # Analyze images and annotations
    total_images = 0
    image_counts = {}
    resolution_counter = Counter()

    # Class size statistics: {class_id: [areas...], ...}
    class_pixel_areas = {}
    class_area_ratios = {}

    for s in splits_to_check:
        img_dir = dataset_path / "images" / s
        label_dir = dataset_path / "labels" / s

        if not img_dir.exists():
            continue

        image_files = [f for f in img_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS]
        image_counts[s] = len(image_files)
        total_images += len(image_files)

        # Analyze each image
        for img_file in image_files:
            try:
                # Read image to get resolution
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                height, width = img.shape[:2]
                resolution = f"{width}x{height}"
                resolution_counter[resolution] += 1
                image_area = width * height

                # Analyze corresponding label file
                label_file = label_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    continue

                with open(label_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            continue

                        class_id = int(float(parts[0]))

                        # Calculate pixel area based on format
                        if detected_format == "obb" and len(parts) == 9:
                            # OBB format: 8 coordinates (4 points)
                            coords = np.array([float(x) for x in parts[1:]])
                            points = coords.reshape(-1, 2)
                            # Convert to pixel coordinates
                            points[:, 0] *= width
                            points[:, 1] *= height
                            # Calculate area using cv2.contourArea
                            area = cv2.contourArea(points.astype(np.float32))

                        elif detected_format == "bbox" and len(parts) == 5:
                            # Bbox format: center_x, center_y, w, h
                            _, cx, cy, w, h = [float(x) for x in parts]
                            # Convert to pixel dimensions
                            pixel_w = w * width
                            pixel_h = h * height
                            area = pixel_w * pixel_h
                        else:
                            continue

                        # Record area statistics
                        if class_id not in class_pixel_areas:
                            class_pixel_areas[class_id] = []
                            class_area_ratios[class_id] = []

                        class_pixel_areas[class_id].append(area)
                        class_area_ratios[class_id].append(area / image_area)

            except Exception as e:
                logger.warning(f"Error analyzing {img_file}: {e}")
                continue

    # Calculate statistics for each class
    class_size_stats = {}
    for class_id in class_pixel_areas:
        areas = np.array(class_pixel_areas[class_id])
        ratios = np.array(class_area_ratios[class_id])

        class_size_stats[class_id] = {
            "class_name": class_names.get(class_id, f"class_{class_id}"),
            "pixel_area": {
                "min": float(np.min(areas)),
                "max": float(np.max(areas)),
                "mean": float(np.mean(areas)),
                "median": float(np.median(areas)),
                "std": float(np.std(areas)),
            },
            "area_ratio": {
                "min": float(np.min(ratios)),
                "max": float(np.max(ratios)),
                "mean": float(np.mean(ratios)),
                "median": float(np.median(ratios)),
                "std": float(np.std(ratios)),
            },
            "count": len(areas),
        }

    results = {
        "class_counts": class_counts,
        "total_instances": sum(class_counts.values()),
        "num_classes": len(class_counts),
        "format": detected_format,
        "total_images": total_images,
        "image_counts": image_counts,
        "resolution_distribution": dict(resolution_counter.most_common()),
        "class_size_stats": class_size_stats,
    }

    # Print detailed statistics if requested
    if show_details:
        _print_analysis_details(results, class_names)

    return results


def _print_analysis_details(results: dict, class_names: dict) -> None:
    """
    Print detailed analysis results in a formatted way.

    Args:
        results: Analysis results dictionary
        class_names: Mapping of class IDs to class names
    """
    print("\n" + "=" * 80)
    print("数据集分析报告")
    print("=" * 80)

    # Basic info
    print(f"\n数据集格式: {results['format']}")
    print(f"总图片数量: {results['total_images']}")
    print(f"总标注数量: {results['total_instances']}")
    print(f"类别数量: {results['num_classes']}")

    # Split distribution
    if results["image_counts"]:
        print("\n数据集划分:")
        print("-" * 80)
        for split_name, count in results["image_counts"].items():
            print(f"  {split_name}: {count} 张图片")

    # Resolution distribution
    if results["resolution_distribution"]:
        print("\n图片分辨率分布 (前10名):")
        print("-" * 80)
        for i, (resolution, count) in enumerate(
            list(results["resolution_distribution"].items())[:10], 1
        ):
            percentage = (count / results["total_images"]) * 100
            print(f"  {i:2d}. {resolution:15s}: {count:6d} 张 ({percentage:5.1f}%)")

        if len(results["resolution_distribution"]) > 10:
            remaining = len(results["resolution_distribution"]) - 10
            print(f"  ... 还有 {remaining} 种其他分辨率")

    # Class distribution
    if results["class_counts"]:
        print("\n类别分布:")
        print("-" * 80)
        for class_id in sorted(results["class_counts"].keys()):
            class_name = class_names.get(class_id, f"class_{class_id}")
            count = results["class_counts"][class_id]
            percentage = (count / results["total_instances"]) * 100
            print(f"  类别 {class_id:2d} ({class_name:20s}): {count:6d} 个 ({percentage:5.1f}%)")

    # Class size statistics
    if results["class_size_stats"]:
        print("\n类别尺寸统计:")
        print("=" * 80)

        for class_id in sorted(results["class_size_stats"].keys()):
            stats = results["class_size_stats"][class_id]
            class_name = stats["class_name"]

            print(f"\n类别 {class_id} - {class_name} ({stats['count']} 个实例)")
            print("-" * 80)

            # Pixel area statistics
            pixel_stats = stats["pixel_area"]
            print("  像素面积统计:")
            print(f"    最小值: {pixel_stats['min']:>12.0f} 像素²")
            print(f"    最大值: {pixel_stats['max']:>12.0f} 像素²")
            print(f"    平均值: {pixel_stats['mean']:>12.0f} 像素²")
            print(f"    中位数: {pixel_stats['median']:>12.0f} 像素²")
            print(f"    标准差: {pixel_stats['std']:>12.0f} 像素²")

            # Area ratio statistics
            ratio_stats = stats["area_ratio"]
            print("  占图片面积比例:")
            print(f"    最小值: {ratio_stats['min'] * 100:>10.2f}%")
            print(f"    最大值: {ratio_stats['max'] * 100:>10.2f}%")
            print(f"    平均值: {ratio_stats['mean'] * 100:>10.2f}%")
            print(f"    中位数: {ratio_stats['median'] * 100:>10.2f}%")
            print(f"    标准差: {ratio_stats['std'] * 100:>10.2f}%")

    print("\n" + "=" * 80 + "\n")

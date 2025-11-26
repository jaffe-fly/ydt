"""
YDT Command Line Interface

Provides easy-to-use commands for YOLO dataset processing.
"""

import argparse
import os
import sys

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

__version__ = "0.4.0"


def create_parser():
    """Create command line argument parser"""

    parser = argparse.ArgumentParser(
        prog="ydt",
        description="YOLO Dataset Tools - Process and manage YOLO format datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"ydt {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ========== IMAGE PROCESSING COMMANDS ==========

    # slice - Slice images into tiles
    slice_p = subparsers.add_parser("slice", help="Slice large images into tiles")
    slice_p.add_argument("-i", "--input", required=True, help="Input image file or directory")
    slice_p.add_argument("-o", "--output", required=True, help="Output directory")
    slice_p.add_argument(
        "-c", "--count", type=int, default=3, help="Number of horizontal slices (default: 3)"
    )
    slice_p.add_argument(
        "-d",
        "--vertical-count",
        type=int,
        help="Number of vertical slices (optional, enables grid slicing)",
    )
    slice_p.add_argument(
        "-r",
        "--overlap",
        type=float,
        default=0.1,
        help="Overlap ratio for horizontal slices (default: 0.1)",
    )
    slice_p.add_argument(
        "--overlap-vertical",
        type=float,
        default=0.0,
        help="Overlap ratio for vertical slices (default: 0.0)",
    )

    # augment - Augment dataset with rotations
    aug_p = subparsers.add_parser("augment", help="Augment dataset with rotations")
    aug_p.add_argument(
        "-i", "--input", required=True, help="Input dataset directory or single image file"
    )
    aug_p.add_argument("-o", "--output", required=True, help="Output directory")
    aug_p.add_argument(
        "-a", "--angles", nargs="+", type=int, help="Rotation angles (default: auto)"
    )

    # video - Extract frames from videos
    video_p = subparsers.add_parser("video", help="Extract frames from videos")
    video_p.add_argument(
        "-i", "--input", required=True, help="Video file or directory containing videos"
    )
    video_p.add_argument(
        "-o", "--output", required=True, help="Output directory for extracted frames"
    )
    video_p.add_argument(
        "-s", "--step", type=int, default=40, help="Extract every Nth frame (default: 40)"
    )
    video_p.add_argument(
        "-w", "--workers", type=int, help="Number of parallel workers (default: auto-detect)"
    )
    video_p.add_argument(
        "--parallel", action="store_true", help="Enable parallel processing for multiple videos"
    )

    # crop-coords - Crop images by coordinates
    crop_coords_p = subparsers.add_parser("crop-coords", help="Crop images by coordinates")
    crop_coords_p.add_argument("-i", "--input", required=True, help="Input image directory")
    crop_coords_p.add_argument("-o", "--output", required=True, help="Output directory")
    crop_coords_p.add_argument(
        "-c", "--coords", required=True, help="Crop coordinates (x1,y1,x2,y2)"
    )
    crop_coords_p.add_argument(
        "--no-recursive", action="store_true", help="Don't search subdirectories"
    )

    # crop - Crop objects from images
    crop_p = subparsers.add_parser(
        "crop", help="Crop objects from images using model or dataset labels"
    )
    crop_p.add_argument(
        "-i", "--input", required=True, help="Input image/directory or dataset path"
    )
    crop_p.add_argument("-o", "--output", required=True, help="Output directory")
    crop_p.add_argument(
        "--mode",
        choices=["model", "dataset"],
        required=True,
        help="Crop mode: model (use YOLO model) or dataset (use label files)",
    )
    crop_p.add_argument("--model", help="YOLO model path (required for model mode)")
    crop_p.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="train",
        help="Dataset split to process (dataset mode only, default: train)",
    )
    crop_p.add_argument("--classes", type=int, nargs="+", help="Class IDs to crop (default: all)")
    crop_p.add_argument(
        "--conf", type=float, default=0.5, help="Confidence threshold (model mode, default: 0.5)"
    )
    crop_p.add_argument(
        "--iou", type=float, default=0.5, help="NMS IOU threshold (model mode, default: 0.5)"
    )
    crop_p.add_argument("--obb", action="store_true", help="Use OBB format (model mode)")
    crop_p.add_argument(
        "--padding", type=int, default=0, help="Padding pixels around crop (default: 0)"
    )
    crop_p.add_argument(
        "--min-size", type=int, help="Minimum object dimension (width or height) in pixels"
    )
    crop_p.add_argument(
        "--max-size", type=int, help="Maximum object dimension (width or height) in pixels"
    )
    crop_p.add_argument(
        "--device", type=int, default=0, help="Device for inference (model mode, default: 0)"
    )

    # resize - Resize images or dataset
    resize_p = subparsers.add_parser("resize", help="Resize images or YOLO dataset")
    resize_p.add_argument(
        "-i", "--input", required=True, help="Input image/directory or dataset directory"
    )
    resize_p.add_argument("-o", "--output", required=True, help="Output directory")
    resize_p.add_argument(
        "-t", "--target-size", type=int, default=640, help="Target size (default: 640)"
    )
    resize_p.add_argument(
        "--mode",
        choices=["image", "dataset"],
        default="image",
        help="Resize mode: image (directory) or dataset (YOLO dataset) (default: image)",
    )
    resize_p.add_argument(
        "--interpolation",
        choices=["linear", "lanczos4"],
        default="linear",
        help="Interpolation method (default: linear)",
    )
    resize_p.add_argument(
        "--resize-all",
        action="store_true",
        help="Resize all images (including those larger than target size, proportionally scale down)",
    )
    resize_p.add_argument(
        "--resize-mode",
        choices=["longest", "shortest", "width", "height"],
        default="longest",
        help="Which edge to resize (default: longest)",
    )
    resize_p.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which split to process in dataset mode (default: both)",
    )
    resize_p.add_argument(
        "--no-recursive", action="store_true", help="Don't search subdirectories (image mode only)"
    )

    # concat - Concatenate images
    concat_p = subparsers.add_parser("concat", help="Concatenate two images")
    concat_p.add_argument("-i", "--images", nargs=2, required=True, help="Two input images")
    concat_p.add_argument("-o", "--output", required=True, help="Output image path")
    concat_p.add_argument(
        "-d",
        "--direction",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="Concatenation direction (default: horizontal)",
    )
    concat_p.add_argument(
        "--align",
        choices=["start", "center", "end"],
        default="center",
        help="Alignment (default: center)",
    )

    # ========== DATASET COMMANDS ==========

    # split - Split dataset into train/val
    split_p = subparsers.add_parser("split", help="Split dataset into train/val")
    split_p.add_argument(
        "-i", "--input", required=True, help="Input dataset directory or YAML file"
    )
    split_p.add_argument("-o", "--output", required=True, help="Output directory")
    split_p.add_argument(
        "-r", "--ratio", type=float, default=0.8, help="Train ratio (default: 0.8)"
    )

    # merge - Merge multiple datasets
    merge_p = subparsers.add_parser("merge", help="Merge multiple datasets")
    merge_p.add_argument(
        "-i", "--input", nargs="+", required=True, help="Input dataset directories"
    )
    merge_p.add_argument("-o", "--output", required=True, help="Output directory")

    # extract - Extract specific data from dataset
    extract_p = subparsers.add_parser("extract", help="Extract classes, images, or labels")
    extract_p.add_argument(
        "--mode",
        required=True,
        choices=["class", "images-only", "labels-only"],
        help="Extraction mode: class (images+labels), images-only, or labels-only",
    )
    extract_p.add_argument("-i", "--input", required=True, help="Input dataset directory")
    extract_p.add_argument("-o", "--output", required=True, help="Output directory")
    extract_p.add_argument(
        "--class-ids",
        nargs="+",
        type=int,
        help="Class IDs to extract (e.g., 0 2 5) (required for class/images-only modes)",
    )
    extract_p.add_argument("--image-dir", help="Image directory (required for labels-only mode)")
    extract_p.add_argument(
        "--operation",
        choices=["copy", "move"],
        default="copy",
        help="Copy or move files (default: copy)",
    )
    extract_p.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which split to extract (default: both)",
    )
    extract_p.add_argument(
        "--filter-labels",
        action="store_true",
        help="Filter label content to only keep specified class annotations (class mode only)",
    )
    extract_p.add_argument(
        "--remap-ids",
        action="store_true",
        help="Remap class IDs to sequential 0,1,2... (requires --filter-labels)",
    )

    # synthesize - Generate synthetic dataset
    synth_p = subparsers.add_parser("synthesize", help="Generate synthetic dataset")
    synth_p.add_argument("-t", "--targets", required=True, help="Target objects directory")
    synth_p.add_argument("-b", "--backgrounds", required=True, help="Background images directory")
    synth_p.add_argument("-o", "--output", required=True, help="Output directory")
    synth_p.add_argument(
        "-n", "--num", type=int, default=1000, help="Number of images (default: 1000)"
    )
    synth_p.add_argument(
        "--objects-per-image",
        default="1",
        help="Objects per background image: single number (2) or range (5-10) (default: 1)",
    )
    synth_p.add_argument(
        "--split",
        choices=["train", "trainval"],
        default="trainval",
        help="Generate train only or train+val split (default: trainval)",
    )
    synth_p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio for train/val split (default: 0.8)",
    )
    synth_p.add_argument(
        "--data-yaml",
        help="Path to data.yaml file for class names validation. Target filenames must contain class names (e.g., bn_back.jpg requires class name 'bn' in data.yaml)",
    )
    synth_p.add_argument(
        "--rotation-range",
        default="-90,90",
        metavar="MIN,MAX",
        help='Rotation angle range in degrees, format: "min,max" (default: -90,90). Use --rotation-range=-20,20 (with equals sign) for negative values',
    )
    synth_p.add_argument(
        "--format",
        choices=["obb", "hbb"],
        default="obb",
        help="Annotation format: obb (Oriented Bounding Box) or hbb (Horizontal Bounding Box) (default: obb)",
    )
    synth_p.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Enable balanced sampling mode: distribute class usage evenly across synthesized images",
    )

    # auto-label - Auto-label images using YOLO model
    auto_label_p = subparsers.add_parser("auto-label", help="Auto-label images using YOLO model")
    auto_label_p.add_argument("-i", "--input", required=True, help="Input images directory")
    auto_label_p.add_argument("-m", "--model", required=True, help="YOLO model path")
    auto_label_p.add_argument(
        "--format", required=True, choices=["bbox", "obb"], help="Output format (bbox or obb)"
    )
    auto_label_p.add_argument("-o", "--output", help="Output directory (default: auto-generated)")
    auto_label_p.add_argument("-d", "--device", default=0, help="Device ID (default: 0)")
    auto_label_p.add_argument(
        "--conf-thres", type=float, default=0.25, help="Confidence threshold (default: 0.25)"
    )
    auto_label_p.add_argument(
        "--iou-thres", type=float, default=0.7, help="IOU threshold (default: 0.7)"
    )
    auto_label_p.add_argument(
        "--dry-run", action="store_true", help="Preview mode without making changes"
    )

    # analyze - Analyze dataset statistics
    analyze_p = subparsers.add_parser("analyze", help="Analyze dataset statistics")
    analyze_p.add_argument(
        "-i", "--input", required=True, help="Dataset directory containing data.yaml"
    )
    analyze_p.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="train",
        help="Which split to analyze (default: train)",
    )

    # ========== VISUALIZATION COMMANDS ==========

    # visualize - Visualize dataset
    viz_p = subparsers.add_parser("visualize", help="Visualize YOLO dataset interactively")
    viz_p.add_argument(
        "-i", "--input", required=True, help="Dataset directory or single image file"
    )
    viz_p.add_argument(
        "--filter",
        nargs="+",
        help="Filter by class names (shows only specified classes)",
    )
    viz_p.add_argument(
        "--split",
        choices=["train", "val", "both"],
        default="both",
        help="Which split to visualize (default: both)",
    )

    # viz-letterbox - Visualize letterbox effect
    letterbox_p = subparsers.add_parser("viz-letterbox", help="Visualize letterbox transformation")
    letterbox_p.add_argument("-i", "--input", required=True, help="Input image file")
    letterbox_p.add_argument(
        "-s", "--size", type=int, default=640, help="Target size (default: 640)"
    )

    return parser


def main():
    """Main entry point for CLI"""

    parser = create_parser()
    args = parser.parse_args()

    # Setup logging using core logger system
    import logging

    from ydt.core.logger import get_logger, setup_logger

    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Setup global logger (use default format with line numbers)
    logger = setup_logger(name="ydt", level=log_level)

    if not args.command:
        parser.print_help()
        return 0

    try:
        # ========== IMAGE PROCESSING COMMANDS ==========
        if args.command == "slice":
            from ydt.image import slice_dataset

            logger.info(f"Slicing images from {args.input}")
            slice_dataset(
                args.input,
                args.output,
                horizontal_count=args.count,
                vertical_count=args.vertical_count,
                overlap_ratio_horizontal=args.overlap,
                overlap_ratio_vertical=args.overlap_vertical,
            )

        elif args.command == "augment":
            from ydt.image import augment_dataset

            logger.info(f"Augmenting dataset from {args.input}")
            augment_dataset(args.input, args.output, angles=args.angles)

        elif args.command == "video":
            logger.info(f"Extracting frames from {args.input}")

            if args.parallel:
                from ydt.image import extract_frames_parallel

                total_frames = extract_frames_parallel(
                    args.input,
                    args.output,
                    step=args.step,
                    max_workers=args.workers,
                )
            else:
                from ydt.image import extract_frames

                total_frames = extract_frames(
                    args.input,
                    args.output,
                    step=args.step,
                )
            logger.info(f"Successfully extracted {total_frames} frames")

        elif args.command == "crop-coords":
            from ydt.image import crop_directory_by_coords

            logger.info(f"Cropping images from {args.input}")
            coords = tuple(map(int, args.coords.split(",")))
            if len(coords) != 4:
                logger.error("Coordinates must be in format: x1,y1,x2,y2")
                return 1
            x1, y1, x2, y2 = coords
            success_count, failure_count = crop_directory_by_coords(
                args.input,
                args.output,
                x1,
                y1,
                x2,
                y2,
                recursive=not args.no_recursive,
            )
            logger.info(f"Cropped {success_count} images successfully, {failure_count} failed")

        elif args.command == "crop":
            if args.mode == "model":
                # Model inference mode
                if not args.model:
                    logger.error("--model is required for model mode")
                    return 1

                from ydt.image import crop_with_model

                logger.info(f"Cropping objects using model: {args.model}")
                logger.info(f"Input: {args.input}")
                logger.info(f"Output: {args.output}")

                result = crop_with_model(
                    source=args.input,
                    model_path=args.model,
                    output_dir=args.output,
                    classes=args.classes,
                    conf=args.conf,
                    iou=args.iou,
                    obb=args.obb,
                    padding=args.padding,
                    min_size=args.min_size,
                    max_size=args.max_size,
                    device=args.device,
                )

                logger.info(
                    f"Successfully cropped {result['total_cropped']} objects from {result['total_images']} images"
                )

            elif args.mode == "dataset":
                # Dataset mode
                from ydt.image import crop_from_dataset

                logger.info(f"Cropping objects from dataset: {args.input}")
                logger.info(f"Split: {args.split}")
                logger.info(f"Output: {args.output}")

                result = crop_from_dataset(
                    dataset_path=args.input,
                    output_dir=args.output,
                    split=args.split,
                    classes=args.classes,
                    padding=args.padding,
                    min_size=args.min_size,
                    max_size=args.max_size,
                )

                logger.info(
                    f"Successfully cropped {result['total_cropped']} objects from {result['total_images']} images"
                )

        elif args.command == "resize":
            from pathlib import Path

            input_path = Path(args.input)
            is_yolo_dataset = (input_path / "data.yaml").exists()

            if args.mode == "dataset":
                from ydt.image import resize_dataset

                logger.info(f"Resizing YOLO dataset from {args.input}")
                stats = resize_dataset(
                    args.input,
                    args.output,
                    target_size=args.target_size,
                    resize_mode=args.resize_mode,
                    interpolation=args.interpolation,
                    resize_all=args.resize_all,
                    split=args.split,
                )
                logger.info(
                    f"Dataset resize completed: {stats['resized_count']} resized, "
                    f"{stats['copied_count']} copied, {stats['failed_count']} failed"
                )
            else:  # image mode
                if is_yolo_dataset:
                    logger.error(
                        f"Detected YOLO dataset (data.yaml found in {args.input}).\n"
                        "Use --mode dataset to process YOLO dataset.\n"
                        "Example: ydt resize -i <dataset> -o <output> --mode dataset"
                    )
                    return 1

                from ydt.image import resize_directory

                logger.info(f"Resizing images from {args.input}")
                success_count, failure_count = resize_directory(
                    args.input,
                    args.output,
                    target_size=args.target_size,
                    resize_mode=args.resize_mode,
                    interpolation=args.interpolation,
                    resize_all=args.resize_all,
                    recursive=not args.no_recursive,
                )
                logger.info(f"Resized {success_count} images successfully, {failure_count} failed")

        elif args.command == "concat":
            from ydt.image import concat_images_horizontally, concat_images_vertically

            logger.info(f"Concatenating images: {args.images[0]} + {args.images[1]}")
            if args.direction == "horizontal":
                concat_images_horizontally(
                    args.images[0], args.images[1], args.output, alignment=args.align
                )
            else:  # vertical
                concat_images_vertically(
                    args.images[0], args.images[1], args.output, alignment=args.align
                )
            logger.info(f"Concatenated image saved to: {args.output}")

        # ========== DATASET COMMANDS ==========
        elif args.command == "split":
            from ydt.dataset import split_dataset

            logger.info(f"Splitting dataset from {args.input}")
            split_dataset(args.input, args.output, train_ratio=args.ratio)

        elif args.command == "merge":
            from ydt.dataset import merge_datasets

            logger.info(f"Merging {len(args.input)} datasets")
            merge_datasets(args.input, args.output)

        elif args.command == "extract":
            from ydt.dataset import extract_by_class, extract_images_only, extract_labels_only

            # Determine which splits to extract
            extract_train = args.split in ["train", "both"]
            extract_val = args.split in ["val", "both"]

            if args.mode == "class":
                if not args.class_ids:
                    logger.error("--class-ids is required for 'class' mode")
                    return 1

                logger.info(f"Extracting class IDs: {args.class_ids}")
                logger.info("Mode: images + labels")
                logger.info(f"Operation: {args.operation}")
                extract_by_class(
                    args.input,
                    args.output,
                    args.class_ids,
                    operation=args.operation,
                    extract_train=extract_train,
                    extract_val=extract_val,
                    filter_labels=args.filter_labels,
                    remap_ids=args.remap_ids,
                )

            elif args.mode == "images-only":
                if not args.class_ids:
                    logger.error("--class-ids is required for 'images-only' mode")
                    return 1

                logger.info(f"Extracting images for class IDs: {args.class_ids}")
                logger.info(f"Operation: {args.operation}")
                extract_images_only(
                    args.input,
                    args.output,
                    args.class_ids,
                    operation=args.operation,
                    extract_train=extract_train,
                    extract_val=extract_val,
                )

            elif args.mode == "labels-only":
                if not args.image_dir:
                    logger.error("--image-dir is required for 'labels-only' mode")
                    return 1

                logger.info(f"Extracting labels for images in: {args.image_dir}")
                extract_labels_only(
                    args.input,
                    args.image_dir,
                    args.output,
                    extract_train=extract_train,
                    extract_val=extract_val,
                )

        elif args.command == "synthesize":
            from ydt.dataset import DatasetSynthesizer

            logger.info("Generating synthetic dataset")
            logger.info(f"Objects per image: {args.objects_per_image}")
            logger.info(f"Split mode: {args.split}")
            if args.split == "trainval":
                logger.info(f"Train ratio: {args.train_ratio}")

            # Parse objects_per_image parameter
            objects_per_image = args.objects_per_image
            if "-" in objects_per_image:
                try:
                    min_obj, max_obj = map(int, objects_per_image.split("-"))
                    objects_per_image = (min_obj, max_obj)
                except ValueError:
                    logger.error(
                        f"Invalid objects range format: {objects_per_image}. Use format like '5-10'"
                    )
                    return 1
            else:
                try:
                    objects_per_image = int(objects_per_image)
                except ValueError:
                    logger.error(
                        f"Invalid objects number: {objects_per_image}. Use single number or range"
                    )
                    return 1

            # Parse rotation_range parameter
            rotation_range = None
            if hasattr(args, "rotation_range") and args.rotation_range:
                try:
                    min_angle, max_angle = map(float, args.rotation_range.split(","))
                    rotation_range = (min_angle, max_angle)
                    logger.info(f"Rotation range: {rotation_range[0]}° to {rotation_range[1]}°")
                except ValueError:
                    logger.error(
                        f"Invalid rotation range format: {args.rotation_range}. Use format like '-20,20'"
                    )
                    return 1

            synthesizer = DatasetSynthesizer(
                args.targets,
                args.backgrounds,
                args.output,
                objects_per_image=objects_per_image,
                split_mode=args.split,
                train_ratio=args.train_ratio,
                data_yaml_path=args.data_yaml if hasattr(args, "data_yaml") else None,
                rotation_range=rotation_range,
                annotation_format=args.format,
                balanced_sampling=args.balanced_sampling,
            )
            logger.info(f"Annotation format: {args.format.upper()}")
            if args.balanced_sampling:
                logger.info(
                    "Balanced sampling mode: classes will be cycled evenly across generated images"
                )
            synthesizer.synthesize_dataset(num_images=args.num)

        elif args.command == "auto-label":
            from ydt.auto_label import auto_label_dataset

            logger.info(f"Auto-labeling images from {args.input}")
            result = auto_label_dataset(
                input_dir=args.input,
                model_path=args.model,
                format_type=args.format,
                output_dir=args.output,
                device=args.device,
                conf_threshold=args.conf_thres,
                iou_threshold=args.iou_thres,
                dry_run=args.dry_run,
            )

            if result["success"]:
                logger.info(f"Successfully processed {result['processed_count']} images")
                if result["output_dir"]:
                    logger.info(f"Output saved to: {result['output_dir']}")
            else:
                logger.error(f"Auto-labeling failed: {result.get('message', 'Unknown error')}")
                return 1

        elif args.command == "analyze":
            from ydt.dataset import analyze_dataset

            logger.info(f"Analyzing dataset: {args.input}")
            logger.info(f"Split: {args.split}")
            try:
                _ = analyze_dataset(args.input, split=args.split, show_details=True)
                logger.info("Analysis completed successfully")
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                return 1

        # ========== VISUALIZATION COMMANDS ==========
        elif args.command == "visualize":
            from pathlib import Path

            from ydt.visual import visualize_dataset

            input_path = Path(args.input)

            # Detect if input is a file or directory
            if input_path.is_file():
                logger.info(f"Visualizing single image: {args.input}")

                # Find dataset root by looking for data.yaml
                dataset_root = None
                current = input_path.parent
                for _ in range(5):  # Search up to 5 levels
                    if (current / "data.yaml").exists():
                        dataset_root = current
                        break
                    if current == current.parent:  # Reached filesystem root
                        break
                    current = current.parent

                if dataset_root is None:
                    logger.error(f"Cannot find data.yaml in parent directories of {args.input}")
                    return 1

                visualize_dataset(
                    dataset_path=dataset_root,
                    filter_labels=args.filter,
                    scan_train=False,
                    scan_val=False,
                    single_image_path=args.input,
                )
            else:
                logger.info(f"Visualizing dataset: {args.input}")
                scan_train = args.split in ["train", "both"]
                scan_val = args.split in ["val", "both"]
                visualize_dataset(
                    args.input,
                    filter_labels=args.filter,
                    scan_train=scan_train,
                    scan_val=scan_val,
                    single_image_path=None,
                )

        elif args.command == "viz-letterbox":
            from ydt.visual import visualize_letterbox

            logger.info(f"Visualizing letterbox effect on: {args.input}")
            visualize_letterbox(args.input, letterbox_size=(args.size, args.size))

        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            return 1

        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nOperation cancelled by user (Ctrl+C)")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger = get_logger(__name__)
        logger.exception(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

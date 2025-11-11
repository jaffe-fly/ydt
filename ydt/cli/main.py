"""
YDT Command Line Interface

Provides easy-to-use commands for YOLO dataset processing.
"""

import argparse
import sys

__version__ = "0.2.5"


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

    # IMAGE PROCESSING COMMANDS
    img_parser = subparsers.add_parser("image", help="Image processing operations")
    img_sub = img_parser.add_subparsers(dest="subcommand")

    # image slice
    slice_p = img_sub.add_parser("slice", help="Slice large images into tiles")
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

    # image augment
    aug_p = img_sub.add_parser("augment", help="Augment dataset with rotations")
    aug_p.add_argument(
        "-i", "--input", required=True, help="Input dataset directory or single image file"
    )
    aug_p.add_argument("-o", "--output", required=True, help="Output directory")
    aug_p.add_argument(
        "-a", "--angles", nargs="+", type=int, help="Rotation angles (default: auto)"
    )

    # image video
    video_p = img_sub.add_parser("video", help="Extract frames from videos")
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

    # image crop-coords
    crop_coords_p = img_sub.add_parser("crop-coords", help="Crop images by coordinates")
    crop_coords_p.add_argument("-i", "--input", required=True, help="Input image file or directory")
    crop_coords_p.add_argument("-o", "--output", required=True, help="Output directory")
    crop_coords_p.add_argument(
        "-c", "--coords", required=True, help="Crop coordinates (x1,y1,x2,y2)"
    )
    crop_coords_p.add_argument(
        "--no-recursive", action="store_true", help="Don't search subdirectories"
    )

    # image resize
    resize_p = img_sub.add_parser("resize", help="Resize images using both scale and crop methods")
    resize_p.add_argument("-i", "--input", required=True, help="Input image file or directory")
    resize_p.add_argument("-o", "--output", required=True, help="Output directory")
    resize_p.add_argument(
        "-s",
        "--sizes",
        nargs="+",
        type=int,
        required=True,
        help="Target widths (e.g., -s 640 800 1024)",
    )

    # image concat
    concat_p = img_sub.add_parser("concat", help="Concatenate two images")
    concat_p.add_argument("images", nargs=2, help="Two images to concatenate")
    concat_p.add_argument("-o", "--output", required=True, help="Output image path")
    concat_p.add_argument(
        "-d",
        "--direction",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="Concatenation direction (default: horizontal)",
    )
    concat_p.add_argument(
        "-a",
        "--align",
        choices=["top", "center", "bottom", "left", "right"],
        default="center",
        help="Alignment (horizontal: top/center/bottom, vertical: left/center/right)",
    )

    # DATASET COMMANDS
    ds_parser = subparsers.add_parser("dataset", help="Dataset operations")
    ds_sub = ds_parser.add_subparsers(dest="subcommand")

    # dataset split
    split_p = ds_sub.add_parser("split", help="Split dataset into train/val")
    split_p.add_argument("-i", "--input", required=True, help="Input dataset YAML file")
    split_p.add_argument("-o", "--output", required=True, help="Output directory")
    split_p.add_argument(
        "-r", "--ratio", type=float, default=0.8, help="Train ratio (default: 0.8)"
    )
    split_p.add_argument("--balance", action="store_true", help="Balance rotation angles")

    # dataset merge
    merge_p = ds_sub.add_parser("merge", help="Merge multiple datasets")
    merge_p.add_argument(
        "-i", "--input", nargs="+", required=True, help="Input dataset directories"
    )
    merge_p.add_argument("-o", "--output", required=True, help="Output directory")

    # dataset synthesize
    synth_p = ds_sub.add_parser("synthesize", help="Generate synthetic dataset")
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

    # dataset auto-label
    auto_label_p = ds_sub.add_parser("auto-label", help="Auto-label images using YOLO model")
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

    # VISUALIZATION COMMANDS
    viz_parser = subparsers.add_parser("viz", help="Visualization operations")
    viz_sub = viz_parser.add_subparsers(dest="subcommand")

    # viz dataset
    dataset_p = viz_sub.add_parser("dataset", help="Visualize dataset annotations")
    dataset_p.add_argument("-i", "--input", required=True, help="Dataset directory or single image")
    dataset_p.add_argument("-f", "--filter", nargs="+", type=int, help="Filter specific class IDs")
    dataset_p.add_argument("--train", action="store_true", help="Show training set")
    dataset_p.add_argument("--val", action="store_true", help="Show validation set")

    # viz letterbox
    letter_p = viz_sub.add_parser("letterbox", help="Preview letterbox effect")
    letter_p.add_argument("-i", "--input", required=True, help="Image path")
    letter_p.add_argument("-s", "--save", help="Save output directory")

    # viz augment
    aug_viz_p = viz_sub.add_parser("augment", help="Preview augmentation effects")
    aug_viz_p.add_argument("-i", "--input", required=True, help="Image path")

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
        # Route to appropriate handler
        if args.command == "image":
            return handle_image_command(args)
        elif args.command == "dataset":
            return handle_dataset_command(args)
        elif args.command == "viz":
            return handle_viz_command(args)
        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        logger.warning("\n\nOperation cancelled by user (Ctrl+C)")
        return 130  # Standard exit code for SIGINT

    except Exception as e:
        logger = get_logger(__name__)
        logger.exception(f"Error: {str(e)}")
        return 1


def handle_image_command(args):
    """Handle image processing commands"""
    from ydt.core.logger import get_logger
    from ydt.image import (
        augment_dataset,
        crop_directory_by_coords,
        extract_frames,
        extract_frames_parallel,
        slice_dataset,
    )
    from ydt.image.concat import concat_images_horizontally, concat_images_vertically
    from ydt.image.resize import process_images_multi_method

    logger = get_logger(__name__)

    if args.subcommand == "slice":
        logger.info(f"Slicing images from {args.input}")
        if args.vertical_count:
            logger.info(
                f"Grid slicing: {args.count} horizontal × {args.vertical_count} vertical = {args.count * args.vertical_count} total slices"
            )
        else:
            logger.info(f"Horizontal slicing: {args.count} slices")
        logger.info(
            f"Horizontal overlap: {args.overlap}, Vertical overlap: {args.overlap_vertical}"
        )

        slice_dataset(
            args.input,
            args.output,
            horizontal_count=args.count,
            vertical_count=args.vertical_count,
            overlap_ratio_horizontal=args.overlap,
            overlap_ratio_vertical=args.overlap_vertical,
        )

    elif args.subcommand == "augment":
        logger.info(f"Augmenting dataset from {args.input}")
        augment_dataset(args.input, args.output, angles=args.angles)

    elif args.subcommand == "video":
        logger.info(f"Extracting frames from {args.input}")

        # Check if parallel processing is requested
        if args.parallel:
            logger.info("Using parallel processing for multiple videos")
            total_frames = extract_frames_parallel(
                args.input, args.output, step=args.step, max_workers=args.workers
            )
        else:
            logger.info("Using sequential processing")
            total_frames = extract_frames(args.input, args.output, step=args.step)

        logger.info(f"Successfully extracted {total_frames} frames")

    elif args.subcommand == "crop-coords":
        from pathlib import Path

        import cv2

        from ydt.image.resize import crop_image_by_coords

        # Parse coordinates string
        try:
            coords = [int(x.strip()) for x in args.coords.split(",")]
            if len(coords) != 4:
                logger.error("Coordinates must be in format: x1,y1,x2,y2")
                return 1

            x1, y1, x2, y2 = coords
        except ValueError:
            logger.error("Invalid coordinates format. Use: x1,y1,x2,y2 (e.g., 100,50,600,400)")
            return 1

        logger.info(f"Cropping images from {args.input}")
        logger.info(f"Crop region: ({x1}, {y1}) -> ({x2}, {y2})")

        # Validate coordinates
        if x1 >= x2 or y1 >= y2:
            logger.error("Invalid coordinates: x1 must be < x2 and y1 must be < y2")
            return 1

        input_path = Path(args.input)

        # Check if input is a file or directory
        if input_path.is_file():
            # Single file mode
            logger.info(f"Processing single image file: {input_path.name}")
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)

            try:
                # Read image
                img = cv2.imread(str(input_path))
                if img is None:
                    logger.error(f"Failed to read image: {input_path}")
                    return 1

                # Crop image
                cropped = crop_image_by_coords(img, x1, y1, x2, y2)

                # Save cropped image
                output_file = output_path / input_path.name
                cv2.imwrite(str(output_file), cropped)
                logger.info(f"Cropped image saved to: {output_file}")

            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return 1
        else:
            # Directory mode
            success_count, failure_count = crop_directory_by_coords(
                input_dir=args.input,
                output_dir=args.output,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                recursive=not args.no_recursive,
            )

            logger.info(f"Cropping complete: {success_count} success, {failure_count} failed")

    elif args.subcommand == "resize":
        logger.info(f"Resizing images from {args.input}")
        logger.info(f"Target sizes: {args.sizes}")
        logger.info(f"Output directory: {args.output}")

        total_processed, total_failed = process_images_multi_method(
            input_path=args.input, output_dir=args.output, target_sizes=args.sizes
        )

        logger.info(f"Resize complete: {total_processed} images processed, {total_failed} failed")

    elif args.subcommand == "concat":
        logger.info(f"Concatenating images: {args.images[0]} and {args.images[1]}")
        logger.info(f"Direction: {args.direction}, Alignment: {args.align}")

        if args.direction == "horizontal":
            concat_images_horizontally(
                args.images[0], args.images[1], args.output, alignment=args.align
            )
        else:  # vertical
            concat_images_vertically(
                args.images[0], args.images[1], args.output, alignment=args.align
            )

        logger.info(f"Concatenated image saved to: {args.output}")

    else:
        print("Unknown image subcommand")
        return 1

    return 0


def handle_dataset_command(args):
    """Handle dataset commands"""
    from ydt.auto_label import auto_label_dataset
    from ydt.core.logger import get_logger
    from ydt.dataset import DatasetSynthesizer, merge_datasets, split_dataset

    logger = get_logger(__name__)

    if args.subcommand == "split":
        logger.info(f"Splitting dataset from {args.input}")
        split_dataset(
            args.input, args.output, train_ratio=args.ratio, balance_rotation=args.balance
        )

    elif args.subcommand == "merge":
        logger.info(f"Merging {len(args.input)} datasets")
        merge_datasets(args.input, args.output)

    elif args.subcommand == "synthesize":
        logger.info("Generating synthetic dataset")
        logger.info(f"Objects per image: {args.objects_per_image}")
        logger.info(f"Split mode: {args.split}")
        if args.split == "trainval":
            logger.info(f"Train ratio: {args.train_ratio}")

        # Parse objects_per_image parameter
        objects_per_image = args.objects_per_image
        if "-" in objects_per_image:
            # Range format: "5-10"
            try:
                min_obj, max_obj = map(int, objects_per_image.split("-"))
                objects_per_image = (min_obj, max_obj)
            except ValueError:
                logger.error(
                    f"Invalid objects range format: {objects_per_image}. Use format like '5-10'"
                )
                return 1
        else:
            # Single number format: "2"
            try:
                objects_per_image = int(objects_per_image)
            except ValueError:
                logger.error(
                    f"Invalid objects number: {objects_per_image}. Use single number or range"
                )
                return 1

        synthesizer = DatasetSynthesizer(
            args.targets,
            args.backgrounds,
            args.output,
            objects_per_image=objects_per_image,
            split_mode=args.split,
            train_ratio=args.train_ratio,
        )
        synthesizer.synthesize_dataset(num_images=args.num)

    elif args.subcommand == "auto-label":
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

    else:
        print("Unknown dataset subcommand")
        return 1

    return 0


def handle_viz_command(args):
    """Handle visualization commands"""
    from ydt.core.logger import get_logger
    from ydt.visual import visualize_dataset, visualize_hsv_augmentation, visualize_letterbox

    logger = get_logger(__name__)

    if args.subcommand == "dataset":
        logger.info(f"Visualizing dataset: {args.input}")
        visualize_dataset(
            args.input,
            filter_labels=args.filter,
            scan_train=args.train or (not args.train and not args.val),
            scan_val=args.val or (not args.train and not args.val),
        )

    elif args.subcommand == "letterbox":
        logger.info(f"Showing letterbox effect: {args.input}")
        visualize_letterbox(args.input, output_dir=args.save)

    elif args.subcommand == "augment":
        logger.info(f"Showing HSV augmentation preview: {args.input}")
        visualize_hsv_augmentation(args.input)

    else:
        print("Unknown viz subcommand")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

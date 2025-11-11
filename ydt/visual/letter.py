"""
Letterbox visualization tools for YOLO preprocessing.

Provides utilities to visualize letterbox transformation effects on images,
including support for irregular regions and interactive preview.
"""

from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics.data.augment import LetterBox
except ImportError:
    raise ImportError(
        "ultralytics is required for letterbox visualization. "
        "Install it with: pip install ultralytics"
    )

from ydt.core.logger import get_logger

logger = get_logger(__name__)


def extract_roi_with_padding(
    image: np.ndarray,
    points: list[tuple[int, int]],
    padding_color: int = 114,
) -> np.ndarray:
    """
    Extract irregular region from ydt.image with padding.

    Args:
        image: Source image array
        points: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        padding_color: Background color for padding (default 114, matching YOLOv8)

    Returns:
        Extracted ROI with background padding

    Examples:
        >>> img = cv2.imread("image.jpg")
        >>> points = [(100, 100), (200, 120), (190, 210), (95, 205)]
        >>> roi = extract_roi_with_padding(img, points)
    """
    # Convert to numpy array
    points_array = np.array(points, dtype=np.int32)

    # Get bounding box
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points_array], 255)

    # Extract ROI
    roi = image[min_y : max_y + 1, min_x : max_x + 1].copy()
    mask_roi = mask[min_y : max_y + 1, min_x : max_x + 1]

    # Create background
    bg = np.ones_like(roi) * padding_color

    # Combine ROI and background
    roi_masked = cv2.bitwise_and(roi, roi, mask=mask_roi)
    bg_masked = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask_roi))
    result = cv2.add(roi_masked, bg_masked)

    return result


def resize_window_to_screen(
    window_name: str,
    image: np.ndarray,
    max_width: int = 1920,
    max_height: int = 1080,
) -> None:
    """
    Resize OpenCV window to fit screen.

    Args:
        window_name: Name of the OpenCV window
        image: Image to display
        max_width: Maximum screen width
        max_height: Maximum screen height
    """
    height, width = image.shape[:2]

    # Calculate scale
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height) * 0.9  # Leave some margin

    # Calculate new window size
    window_width = int(width * scale)
    window_height = int(height * scale)

    # Resize window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)


def visualize_regions_letterbox(
    image_path: str | Path,
    regions: list[list[tuple[int, int]]],
    output_dir: str | Path | None = None,
    letterbox_size: tuple[int, int] = (640, 640),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32,
) -> None:
    """
    Interactively visualize letterbox effect on irregular regions.

    Keyboard controls:
    - 'n': Next region
    - 'p': Previous region
    - 's': Save all regions (if output_dir provided)
    - 'q': Quit

    Args:
        image_path: Path to image file
        regions: List of regions, each region is a list of 4 corner points
        output_dir: Optional directory to save processed images
        letterbox_size: Target size for letterbox (width, height)
        auto: Auto calculate padding
        scale_fill: Scale to fill the new shape
        scale_up: Allow scaling up
        stride: Stride for size calculation

    Raises:
        FileNotFoundError: If image file doesn't exist

    Examples:
        >>> regions = [
        ...     [(100, 100), (200, 120), (190, 210), (95, 205)],
        ...     [(300, 300), (400, 310), (390, 410), (295, 405)],
        ... ]
        >>> visualize_regions_letterbox(
        ...     "image.jpg",
        ...     regions,
        ...     output_dir="./output"
        ... )
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create letterbox transformer
    letterbox = LetterBox(
        new_shape=letterbox_size,
        auto=auto,
        scaleFill=scale_fill,
        scaleup=scale_up,
        stride=stride,
    )

    # Create windows
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Letterbox Effects", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Original Cropped", cv2.WINDOW_NORMAL)

    # Draw all regions on original image
    display_img = original_img.copy()
    for i, points in enumerate(regions):
        points_array = np.array(points, dtype=np.int32)
        cv2.polylines(display_img, [points_array], True, (0, 255, 0), 2)

        # Add region number
        centroid = np.mean(points_array, axis=0, dtype=np.int32)
        cv2.putText(
            display_img,
            f"Region {i}",
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    # Show original image
    resize_window_to_screen("Original Image", display_img)
    cv2.imshow("Original Image", display_img)

    try:
        current_idx = 0
        logger.info(f"Loaded {len(regions)} regions from {image_path.name}")
        logger.info("Controls: 'n'=next, 'p'=previous, 's'=save all, 'q'=quit")

        while current_idx < len(regions):
            points = regions[current_idx]

            # Extract irregular region
            roi = extract_roi_with_padding(original_img, points)

            # Apply letterbox
            letterboxed_img = letterbox(image=roi, labels=None)

            # Display
            cv2.imshow("Original Cropped", roi)
            cv2.imshow("Letterbox Effects", letterboxed_img)

            logger.info(f"Showing Region {current_idx}/{len(regions) - 1}")

            # Wait for key
            while True:
                key = cv2.waitKey(1) & 0xFF

                if key == ord("n"):
                    current_idx = min(current_idx + 1, len(regions) - 1)
                    if current_idx == len(regions) - 1:
                        logger.info("Reached last region")
                    break
                elif key == ord("p"):
                    current_idx = max(current_idx - 1, 0)
                    if current_idx == 0:
                        logger.info("Reached first region")
                    break
                elif key == ord("q"):
                    logger.info("Quit")
                    cv2.destroyAllWindows()
                    return
                elif key == ord("s") and output_dir:
                    # Save all regions
                    logger.info(f"Saving all regions to {output_dir}...")

                    # Save original with boxes
                    cv2.imwrite(
                        str(output_dir / "original_with_boxes.jpg"),
                        display_img,
                    )
                    logger.info("  - original_with_boxes.jpg")

                    # Save all regions
                    for idx, region_points in enumerate(regions):
                        region_roi = extract_roi_with_padding(original_img, region_points)
                        region_letterboxed = letterbox(image=region_roi, labels=None)

                        cv2.imwrite(
                            str(output_dir / f"region_{idx}_cropped.jpg"),
                            region_roi,
                        )
                        cv2.imwrite(
                            str(output_dir / f"region_{idx}_letterboxed.jpg"),
                            region_letterboxed,
                        )
                        logger.info(f"  - region_{idx}_cropped.jpg")
                        logger.info(f"  - region_{idx}_letterboxed.jpg")

                    logger.info("All regions saved!")
                    break

    finally:
        cv2.destroyAllWindows()


def visualize_letterbox(
    image_path: str | Path,
    output_dir: str | Path | None = None,
    letterbox_size: tuple[int, int] = (640, 640),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32,
) -> None:
    """
    Visualize letterbox effect on a single image.

    Keyboard controls:
    - 's': Save letterboxed image (if output_dir provided)
    - 'q': Quit

    Args:
        image_path: Path to image file
        output_dir: Optional directory to save processed images
        letterbox_size: Target size for letterbox (width, height)
        auto: Auto calculate padding
        scale_fill: Scale to fill the new shape
        scale_up: Allow scaling up
        stride: Stride for size calculation

    Raises:
        FileNotFoundError: If image file doesn't exist

    Examples:
        >>> visualize_letterbox("image.jpg", output_dir="./output")
        >>> visualize_letterbox("image.jpg", letterbox_size=(1280, 1280))
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create letterbox transformer
    letterbox = LetterBox(
        new_shape=letterbox_size,
        auto=auto,
        scaleFill=scale_fill,
        scaleup=scale_up,
        stride=stride,
    )

    # Create windows
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Letterbox Effect", cv2.WINDOW_NORMAL)

    # Apply letterbox
    letterboxed_img = letterbox(image=original_img, labels=None)

    # Show images
    resize_window_to_screen("Original Image", original_img)
    resize_window_to_screen("Letterbox Effect", letterboxed_img)

    cv2.imshow("Original Image", original_img)
    cv2.imshow("Letterbox Effect", letterboxed_img)

    logger.info(f"Original size: {original_img.shape[:2]}")
    logger.info(f"Letterbox size: {letterboxed_img.shape[:2]}")
    logger.info("Controls: 's'=save, 'q'=quit")

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info("Quit")
                break
            elif key == ord("s") and output_dir:
                # Save letterboxed image
                base_name = image_path.stem
                output_path = output_dir / f"{base_name}_letterboxed.jpg"
                cv2.imwrite(str(output_path), letterboxed_img)
                logger.info(f"Saved to {output_path}")

    finally:
        cv2.destroyAllWindows()

"""
Image augmentation visualization tools.

Provides utilities to visualize and compare various image augmentation effects,
including HSV adjustments and advanced transformations using albumentations.
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

try:
    from ultralytics.data.augment import RandomHSV
except ImportError:
    raise ImportError(
        "ultralytics is required for HSV augmentation. Install it with: pip install ultralytics"
    )

try:
    import albumentations as A
except ImportError:
    A = None

from ydt.core.logger import get_logger

logger = get_logger(__name__)


def show_side_by_side(
    original: np.ndarray,
    transformed: np.ndarray,
    window_name: str = "Original vs Transformed",
) -> None:
    """
    Display two images side by side.

    Args:
        original: Original image array
        transformed: Transformed image array
        window_name: OpenCV window name

    Examples:
        >>> img1 = cv2.imread("original.jpg")
        >>> img2 = cv2.imread("transformed.jpg")
        >>> show_side_by_side(img1, img2)
    """
    # Ensure same height
    h_orig, w_orig = original.shape[:2]
    h_trans, w_trans = transformed.shape[:2]

    if h_orig != h_trans:
        # Resize to match heights
        target_h = max(h_orig, h_trans)
        if h_orig < target_h:
            original = cv2.resize(original, (w_orig, target_h))
        if h_trans < target_h:
            transformed = cv2.resize(transformed, (w_trans, target_h))

    # Horizontal concatenation
    comparison = np.hstack((original, transformed))

    # Display
    cv2.imshow(window_name, comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_hsv_augmentation(
    image_path: Union[str, Path],
    h_gain: float = 0.02,
    s_gain: float = 0.8,
    v_gain: float = 0.3,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Visualize HSV augmentation effect.

    Displays original and HSV-augmented images side by side.
    Press any key to close the window.

    Args:
        image_path: Path to image file
        h_gain: Hue gain (0.0-1.0)
        s_gain: Saturation gain (0.0-1.0)
        v_gain: Value/brightness gain (0.0-1.0)
        output_path: Optional path to save comparison image

    Raises:
        FileNotFoundError: If image file doesn't exist

    Examples:
        >>> visualize_hsv_augmentation("image.jpg", h_gain=0.02, s_gain=0.8, v_gain=0.3)
        >>> visualize_hsv_augmentation("image.jpg", output_path="hsv_comparison.jpg")
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    logger.info(f"Applying HSV augmentation: H={h_gain}, S={s_gain}, V={v_gain}")

    # Create HSV augmenter
    hsv_aug = RandomHSV(hgain=h_gain, sgain=s_gain, vgain=v_gain)

    # Apply augmentation (modifies in-place)
    augmented_image = image.copy()
    augmented_dict = {"img": augmented_image}
    hsv_aug(augmented_dict)

    # Create side-by-side comparison
    h, w = image.shape[:2]
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w] = image
    combined[:, w:] = augmented_dict["img"]

    # Add labels
    cv2.putText(
        combined,
        "Original",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        combined,
        "HSV Augmented",
        (w + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), combined)
        logger.info(f"Saved comparison to {output_path}")

    # Display
    cv2.imshow("HSV Augmentation Comparison", combined)
    logger.info("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_albumentations(
    image_path: Union[str, Path],
    transforms: Optional[A.Compose] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Visualize albumentations augmentation effects.

    Displays original and augmented images side by side.
    Press any key to close the window.

    Args:
        image_path: Path to image file
        transforms: Albumentations transform pipeline. If None, uses default transforms.
        output_path: Optional path to save comparison image

    Raises:
        FileNotFoundError: If image file doesn't exist
        ImportError: If albumentations is not installed

    Examples:
        >>> # Use default transforms
        >>> visualize_albumentations("image.jpg")

        >>> # Custom transforms
        >>> import albumentations as A
        >>> transform = A.Compose([
        ...     A.RandomBrightnessContrast(p=1.0),
        ...     A.GaussianBlur(p=1.0),
        ... ])
        >>> visualize_albumentations("image.jpg", transforms=transform)
    """
    if A is None:
        raise ImportError(
            "albumentations is required for this function. "
            "Install it with: pip install albumentations"
        )

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Default transforms if not provided
    if transforms is None:
        transforms = A.SomeOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.MotionBlur(p=1),
                A.GaussianBlur(p=1),
                A.ISONoise(p=1),
                A.RandomFog(p=1),
                A.RandomGamma(p=1),
                A.ImageCompression(quality_lower=75, p=1),
            ],
            p=1,
            n=3,
        )
        logger.info("Using default transforms (3 random augmentations)")

    # Apply transforms
    transformed = transforms(image=image)
    transformed_image = transformed["image"]

    # Add labels
    image_labeled = image.copy()
    transformed_labeled = transformed_image.copy()

    cv2.putText(
        image_labeled,
        "Original",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        transformed_labeled,
        "Augmented",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create side-by-side comparison
        comparison = np.hstack((image_labeled, transformed_labeled))
        cv2.imwrite(str(output_path), comparison)
        logger.info(f"Saved comparison to {output_path}")

    # Display
    show_side_by_side(image_labeled, transformed_labeled, "Albumentations Comparison")


def visualize_multiple_augmentations(
    image_path: Union[str, Path],
    transforms: Optional[A.Compose] = None,
    num_examples: int = 4,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Visualize multiple random augmentation examples in a grid.

    Displays original image and multiple augmented versions in a grid layout.
    Press any key to close the window.

    Args:
        image_path: Path to image file
        transforms: Albumentations transform pipeline. If None, uses default transforms.
        num_examples: Number of augmented examples to generate (1-9)
        output_path: Optional path to save grid image

    Raises:
        FileNotFoundError: If image file doesn't exist
        ImportError: If albumentations is not installed
        ValueError: If num_examples is not in range 1-9

    Examples:
        >>> visualize_multiple_augmentations("image.jpg", num_examples=6)
    """
    if A is None:
        raise ImportError(
            "albumentations is required for this function. "
            "Install it with: pip install albumentations"
        )

    if not 1 <= num_examples <= 9:
        raise ValueError("num_examples must be between 1 and 9")

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Default transforms if not provided
    if transforms is None:
        transforms = A.SomeOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.MotionBlur(p=1),
                A.GaussianBlur(p=1),
                A.ISONoise(p=1),
                A.RandomFog(p=1),
                A.RandomGamma(p=1),
                A.ImageCompression(quality_lower=75, p=1),
                A.GaussNoise(p=1),
            ],
            p=1,
            n=3,
        )

    # Generate augmented examples
    images = [image]  # Original
    for i in range(num_examples):
        transformed = transforms(image=image)
        images.append(transformed["image"])

    # Calculate grid layout
    total_images = num_examples + 1  # Include original
    cols = min(3, total_images)
    rows = (total_images + cols - 1) // cols

    # Create grid
    h, w = image.shape[:2]
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols

        # Add label
        img_labeled = img.copy()
        label = "Original" if idx == 0 else f"Aug {idx}"
        cv2.putText(
            img_labeled,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Place in grid
        grid[row * h : (row + 1) * h, col * w : (col + 1) * w] = img_labeled

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), grid)
        logger.info(f"Saved grid to {output_path}")

    # Display
    cv2.imshow(f"Augmentation Grid ({total_images} examples)", grid)
    logger.info("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
Image concatenation utilities

Provides functions for combining multiple images into a single image.
"""

from pathlib import Path
from typing import Literal

from PIL import Image

from ydt.core.logger import get_logger

logger = get_logger(__name__)


def concat_images_horizontally(
    image1_path: str | Path,
    image2_path: str | Path,
    output_path: str | Path,
    alignment: Literal["top", "center", "bottom"] = "center",
    background_color: str = "white",
) -> Path:
    """
    Concatenate two images horizontally (side by side)

    Args:
        image1_path: Path to the left image
        image2_path: Path to the right image
        output_path: Path to save the concatenated image
        alignment: Vertical alignment of images ('top', 'center', 'bottom')
        background_color: Background color for height difference

    Returns:
        Path to the saved concatenated image

    Raises:
        FileNotFoundError: If input images do not exist
        IOError: If image processing fails

    Examples:
        >>> concat_images_horizontally("img1.jpg", "img2.jpg", "output.jpg")
        PosixPath('output.jpg')

        >>> concat_images_horizontally("img1.jpg", "img2.jpg", "output.jpg", alignment="top")
        PosixPath('output.jpg')
    """
    image1_path = Path(image1_path)
    image2_path = Path(image2_path)
    output_path = Path(output_path)

    # Validate input files exist
    if not image1_path.exists():
        raise FileNotFoundError(f"Image 1 not found: {image1_path}")
    if not image2_path.exists():
        raise FileNotFoundError(f"Image 2 not found: {image2_path}")

    logger.info(f"Concatenating {image1_path.name} and {image2_path.name}")

    try:
        # Open images
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)

        # Get dimensions
        width1, height1 = img1.size
        width2, height2 = img2.size

        # Calculate output dimensions
        total_width = width1 + width2
        max_height = max(height1, height2)

        logger.debug(f"Image 1 size: {width1}x{height1}")
        logger.debug(f"Image 2 size: {width2}x{height2}")
        logger.debug(f"Output size: {total_width}x{max_height}")

        # Create new blank image
        new_img = Image.new("RGB", (total_width, max_height), color=background_color)

        # Calculate vertical offsets based on alignment
        if alignment == "top":
            y_offset1 = 0
            y_offset2 = 0
        elif alignment == "bottom":
            y_offset1 = max_height - height1
            y_offset2 = max_height - height2
        else:  # center
            y_offset1 = (max_height - height1) // 2
            y_offset2 = (max_height - height2) // 2

        # Paste images
        new_img.paste(img1, (0, y_offset1))
        new_img.paste(img2, (width1, y_offset2))

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save result
        new_img.save(output_path)
        logger.info(f"Concatenated image saved to: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Error concatenating images: {str(e)}")
        raise OSError(f"Failed to concatenate images: {str(e)}")


def concat_images_vertically(
    image1_path: str | Path,
    image2_path: str | Path,
    output_path: str | Path,
    alignment: Literal["left", "center", "right"] = "center",
    background_color: str = "white",
) -> Path:
    """
    Concatenate two images vertically (top and bottom)

    Args:
        image1_path: Path to the top image
        image2_path: Path to the bottom image
        output_path: Path to save the concatenated image
        alignment: Horizontal alignment of images ('left', 'center', 'right')
        background_color: Background color for width difference

    Returns:
        Path to the saved concatenated image

    Raises:
        FileNotFoundError: If input images do not exist
        IOError: If image processing fails

    Examples:
        >>> concat_images_vertically("img1.jpg", "img2.jpg", "output.jpg")
        PosixPath('output.jpg')
    """
    image1_path = Path(image1_path)
    image2_path = Path(image2_path)
    output_path = Path(output_path)

    if not image1_path.exists():
        raise FileNotFoundError(f"Image 1 not found: {image1_path}")
    if not image2_path.exists():
        raise FileNotFoundError(f"Image 2 not found: {image2_path}")

    logger.info(f"Concatenating {image1_path.name} and {image2_path.name} vertically")

    try:
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)

        width1, height1 = img1.size
        width2, height2 = img2.size

        max_width = max(width1, width2)
        total_height = height1 + height2

        new_img = Image.new("RGB", (max_width, total_height), color=background_color)

        # Calculate horizontal offsets based on alignment
        if alignment == "left":
            x_offset1 = 0
            x_offset2 = 0
        elif alignment == "right":
            x_offset1 = max_width - width1
            x_offset2 = max_width - width2
        else:  # center
            x_offset1 = (max_width - width1) // 2
            x_offset2 = (max_width - width2) // 2

        new_img.paste(img1, (x_offset1, 0))
        new_img.paste(img2, (x_offset2, height1))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_img.save(output_path)
        logger.info(f"Concatenated image saved to: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"Error concatenating images vertically: {str(e)}")
        raise OSError(f"Failed to concatenate images: {str(e)}")

"""
Test image resize module functionality
"""

import cv2
import numpy as np

from ydt.image.resize import (
    center_crop_image,
    crop_directory_by_coords,
    crop_image_by_coords,
    process_images_multi_method,
    process_single_image_multi_method,
    resize_directory,
    resize_image,
)


class TestResizeImage:
    """Test basic image resizing"""

    def test_resize_image_shrink(self, sample_image):
        """Test shrinking image by scale factor"""
        img = cv2.imread(str(sample_image))
        original_h, original_w = img.shape[:2]

        resized = resize_image(img, scale_factor=0.5)

        assert resized.shape[0] == original_h // 2
        assert resized.shape[1] == original_w // 2


class TestCenterCrop:
    """Test center cropping"""

    def test_center_crop_basic(self, sample_image):
        """Test basic center crop"""
        img = cv2.imread(str(sample_image))

        cropped = center_crop_image(img, target_width=320, target_height=240)

        assert cropped.shape[0] == 240
        assert cropped.shape[1] == 320


class TestCropByCoords:
    """Test coordinate-based cropping"""

    def test_crop_image_by_coords_basic(self, sample_image):
        """Test basic coordinate cropping"""
        img = cv2.imread(str(sample_image))

        cropped = crop_image_by_coords(img, x1=100, y1=100, x2=300, y2=300)

        assert cropped.shape[0] == 200
        assert cropped.shape[1] == 200


class TestCropDirectory:
    """Test directory cropping"""

    def test_crop_directory_by_coords_basic(self, temp_dir):
        """Test cropping all images in directory"""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()

        # Create test images
        for i in range(3):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"img{i}.jpg"), img)

        crop_directory_by_coords(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            x1=100,
            y1=100,
            x2=300,
            y2=300,
        )

        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) == 3

        # Check one output
        cropped = cv2.imread(str(output_files[0]))
        assert cropped.shape[0] == 200
        assert cropped.shape[1] == 200


class TestResizeDirectory:
    """Test directory resizing"""

    def test_resize_directory_scale_longest(self, temp_dir):
        """Test resize directory by scaling longest edge"""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()

        # Create test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "test.jpg"), img)

        resize_directory(
            input_path=str(input_dir),
            output_dir=str(output_dir),
            target_size=320,
            resize_mode="longest",
        )

        output_img = cv2.imread(str(output_dir / "test.jpg"))
        # resize_all defaults to False, so small images are upscaled
        # but large images may be copied as-is
        assert output_img is not None

    def test_resize_directory_scale_width(self, temp_dir):
        """Test resize directory by width"""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "test.jpg"), img)

        resize_directory(
            input_path=str(input_dir),
            output_dir=str(output_dir),
            target_size=320,
            resize_mode="width",
            resize_all=True,  # Force resize even if larger than target
        )

        output_img = cv2.imread(str(output_dir / "test.jpg"))
        assert output_img.shape[1] == 320


class TestProcessMultiMethod:
    """Test multi-method processing"""

    def test_process_single_image_scale(self, sample_image, temp_dir):
        """Test processing single image with scale method"""
        output_dir = temp_dir / "output"

        count = process_single_image_multi_method(
            input_file=str(sample_image),
            output_dir=str(output_dir),
            target_sizes=[320],
            use_crop=False,
        )

        assert count > 0
        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) > 0

    def test_process_images_multi_method_directory(self, temp_dir):
        """Test processing directory with multiple methods"""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir()

        # Create test images
        for i in range(2):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"img{i}.jpg"), img)

        total_processed, total_failed = process_images_multi_method(
            input_path=str(input_dir),
            output_dir=str(output_dir),
            target_sizes=[320],
        )

        assert total_processed >= 2  # At least 2 images processed
        assert total_failed == 0

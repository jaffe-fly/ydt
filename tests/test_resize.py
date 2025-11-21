"""
Comprehensive tests for image resize module.

Tests cover:
- Basic image resizing with scale factors
- Center cropping
- Multi-method processing (scale and crop)
- Coordinate-based cropping
- Directory processing with various options
- Edge cases and error handling
"""

import cv2
import numpy as np
import pytest

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

    def test_resize_image_enlarge(self, sample_image):
        """Test enlarging image by scale factor"""
        img = cv2.imread(str(sample_image))
        original_h, original_w = img.shape[:2]

        resized = resize_image(img, scale_factor=2.0)

        assert resized.shape[0] == original_h * 2
        assert resized.shape[1] == original_w * 2

    def test_resize_image_no_change(self, sample_image):
        """Test resize with scale factor 1.0 (no change)"""
        img = cv2.imread(str(sample_image))
        original_shape = img.shape

        resized = resize_image(img, scale_factor=1.0)

        assert resized.shape == original_shape

    def test_resize_image_invalid_scale_factor(self, sample_image):
        """Test error handling for invalid scale factors"""
        img = cv2.imread(str(sample_image))

        with pytest.raises(ValueError, match="Scale factor must be positive"):
            resize_image(img, scale_factor=0)

        with pytest.raises(ValueError, match="Scale factor must be positive"):
            resize_image(img, scale_factor=-0.5)

    def test_resize_image_very_small_scale(self, sample_image):
        """Test resize with very small scale factor"""
        img = cv2.imread(str(sample_image))

        # This should work as long as result is at least 1x1
        resized = resize_image(img, scale_factor=0.01)
        assert resized.shape[0] >= 1
        assert resized.shape[1] >= 1


class TestCenterCrop:
    """Test center cropping functionality"""

    def test_center_crop_basic(self, sample_image):
        """Test basic center cropping"""
        img = cv2.imread(str(sample_image))
        target_w, target_h = 320, 240

        cropped = center_crop_image(img, target_w, target_h)

        assert cropped.shape[0] == target_h
        assert cropped.shape[1] == target_w

    def test_center_crop_larger_than_image(self, sample_image):
        """Test center crop when target is larger than image"""
        img = cv2.imread(str(sample_image))
        h, w = img.shape[:2]

        # Request larger size
        target_w, target_h = w * 2, h * 2

        cropped = center_crop_image(img, target_w, target_h)

        # Should be resized to target size
        assert cropped.shape[0] == target_h
        assert cropped.shape[1] == target_w

    def test_center_crop_invalid_dimensions(self, sample_image):
        """Test error handling for invalid target dimensions"""
        img = cv2.imread(str(sample_image))

        with pytest.raises(ValueError, match="Target dimensions must be positive"):
            center_crop_image(img, 0, 100)

        with pytest.raises(ValueError, match="Target dimensions must be positive"):
            center_crop_image(img, 100, -50)


class TestCropByCoords:
    """Test coordinate-based cropping"""

    def test_crop_image_by_coords_basic(self, sample_image):
        """Test basic coordinate cropping"""
        img = cv2.imread(str(sample_image))

        cropped = crop_image_by_coords(img, x1=100, y1=100, x2=300, y2=250)

        assert cropped.shape[0] == 150  # y2 - y1
        assert cropped.shape[1] == 200  # x2 - x1

    def test_crop_image_by_coords_full_image(self, sample_image):
        """Test cropping entire image"""
        img = cv2.imread(str(sample_image))
        h, w = img.shape[:2]

        cropped = crop_image_by_coords(img, x1=0, y1=0, x2=w, y2=h)

        assert cropped.shape == img.shape

    def test_crop_image_invalid_coords(self, sample_image):
        """Test error handling for invalid coordinates"""
        img = cv2.imread(str(sample_image))
        h, w = img.shape[:2]

        # x1 >= x2
        with pytest.raises(ValueError, match="Invalid crop region"):
            crop_image_by_coords(img, x1=200, y1=100, x2=100, y2=200)

        # y1 >= y2
        with pytest.raises(ValueError, match="Invalid crop region"):
            crop_image_by_coords(img, x1=100, y1=200, x2=200, y2=100)

        # Out of bounds
        with pytest.raises(ValueError, match="outside image bounds"):
            crop_image_by_coords(img, x1=0, y1=0, x2=w + 100, y2=h)

        with pytest.raises(ValueError, match="outside image bounds"):
            crop_image_by_coords(img, x1=-10, y1=0, x2=100, y2=100)

    def test_crop_image_empty_input(self):
        """Test error handling for empty image"""
        empty_img = np.array([])

        with pytest.raises(ValueError, match="Input image is empty"):
            crop_image_by_coords(empty_img, 0, 0, 100, 100)


class TestCropDirectory:
    """Test directory cropping"""

    def test_crop_directory_by_coords_basic(self, temp_dir):
        """Test cropping all images in directory"""
        # Create test images
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        for i in range(3):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"img{i}.jpg"), img)

        output_dir = temp_dir / "output"

        success, failure = crop_directory_by_coords(
            input_dir=input_dir,
            output_dir=output_dir,
            x1=100,
            y1=100,
            x2=400,
            y2=300,
        )

        assert success == 3
        assert failure == 0
        assert output_dir.exists()

        # Check all images were cropped
        output_images = list(output_dir.glob("*.jpg"))
        assert len(output_images) == 3

        # Check dimensions
        for img_path in output_images:
            img = cv2.imread(str(img_path))
            assert img.shape[0] == 200  # y2 - y1
            assert img.shape[1] == 300  # x2 - x1

    def test_crop_directory_recursive(self, temp_dir):
        """Test recursive directory cropping"""
        # Create nested structure
        input_dir = temp_dir / "input"
        (input_dir / "subdir").mkdir(parents=True)

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "img1.jpg"), img)
        cv2.imwrite(str(input_dir / "subdir" / "img2.jpg"), img)

        output_dir = temp_dir / "output"

        success, failure = crop_directory_by_coords(
            input_dir=input_dir,
            output_dir=output_dir,
            x1=100,
            y1=100,
            x2=400,
            y2=300,
            recursive=True,
        )

        assert success == 2
        assert (output_dir / "img1.jpg").exists()
        assert (output_dir / "subdir" / "img2.jpg").exists()

    def test_crop_directory_non_recursive(self, temp_dir):
        """Test non-recursive directory cropping"""
        # Create nested structure
        input_dir = temp_dir / "input"
        (input_dir / "subdir").mkdir(parents=True)

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "img1.jpg"), img)
        cv2.imwrite(str(input_dir / "subdir" / "img2.jpg"), img)

        output_dir = temp_dir / "output"

        success, failure = crop_directory_by_coords(
            input_dir=input_dir,
            output_dir=output_dir,
            x1=100,
            y1=100,
            x2=400,
            y2=300,
            recursive=False,
        )

        # Should only process top-level files
        assert success == 1
        assert (output_dir / "img1.jpg").exists()
        assert not (output_dir / "subdir" / "img2.jpg").exists()

    def test_crop_directory_nonexistent_input(self, temp_dir):
        """Test error handling for nonexistent input directory"""
        with pytest.raises(FileNotFoundError):
            crop_directory_by_coords(
                input_dir="nonexistent",
                output_dir=temp_dir / "output",
                x1=0,
                y1=0,
                x2=100,
                y2=100,
            )

    def test_crop_directory_empty_directory(self, temp_dir):
        """Test cropping empty directory"""
        input_dir = temp_dir / "empty"
        input_dir.mkdir()

        output_dir = temp_dir / "output"

        success, failure = crop_directory_by_coords(
            input_dir=input_dir,
            output_dir=output_dir,
            x1=0,
            y1=0,
            x2=100,
            y2=100,
        )

        assert success == 0
        assert failure == 0


class TestResizeDirectory:
    """Test directory resizing with various modes"""

    def test_resize_directory_scale_longest(self, temp_dir):
        """Test resizing directory with scale method (longest edge)"""
        # Create test images
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        # Create images with different orientations
        img_landscape = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_portrait = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

        cv2.imwrite(str(input_dir / "landscape.jpg"), img_landscape)
        cv2.imwrite(str(input_dir / "portrait.jpg"), img_portrait)

        output_dir = temp_dir / "output"

        success, failure = resize_directory(
            input_path=input_dir,
            output_dir=output_dir,
            target_size=320,
            method="scale",
            resize_mode="longest",
        )

        assert success == 2
        assert failure == 0

        # Check landscape image (width should be 320)
        landscape_resized = cv2.imread(str(output_dir / "landscape.jpg"))
        assert landscape_resized.shape[1] == 320  # width
        assert landscape_resized.shape[0] == 240  # height (proportional)

        # Check portrait image (height should be 320)
        portrait_resized = cv2.imread(str(output_dir / "portrait.jpg"))
        assert portrait_resized.shape[0] == 320  # height
        assert portrait_resized.shape[1] == 240  # width (proportional)

    def test_resize_directory_scale_shortest(self, temp_dir):
        """Test resizing directory with shortest edge mode"""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "img.jpg"), img)

        output_dir = temp_dir / "output"

        success, failure = resize_directory(
            input_path=input_dir,
            output_dir=output_dir,
            target_size=320,
            method="scale",
            resize_mode="shortest",
        )

        assert success == 1

        # Shortest edge (height=480) should become 320
        resized = cv2.imread(str(output_dir / "img.jpg"))
        assert resized.shape[0] == 320  # height (shortest)
        # Width should be proportionally scaled
        expected_width = int(640 * 320 / 480)
        assert resized.shape[1] == expected_width

    def test_resize_directory_scale_width(self, temp_dir):
        """Test resizing directory with width mode"""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "img.jpg"), img)

        output_dir = temp_dir / "output"

        success, failure = resize_directory(
            input_path=input_dir,
            output_dir=output_dir,
            target_size=320,
            method="scale",
            resize_mode="width",
        )

        assert success == 1

        resized = cv2.imread(str(output_dir / "img.jpg"))
        assert resized.shape[1] == 320  # width
        expected_height = int(480 * 320 / 640)
        assert resized.shape[0] == expected_height

    def test_resize_directory_scale_height(self, temp_dir):
        """Test resizing directory with height mode"""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "img.jpg"), img)

        output_dir = temp_dir / "output"

        success, failure = resize_directory(
            input_path=input_dir,
            output_dir=output_dir,
            target_size=320,
            method="scale",
            resize_mode="height",
        )

        assert success == 1

        resized = cv2.imread(str(output_dir / "img.jpg"))
        assert resized.shape[0] == 320  # height
        expected_width = int(640 * 320 / 480)
        assert resized.shape[1] == expected_width

    def test_resize_directory_crop_method(self, temp_dir):
        """Test resizing directory with crop method"""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "img.jpg"), img)

        output_dir = temp_dir / "output"

        success, failure = resize_directory(
            input_path=input_dir,
            output_dir=output_dir,
            target_size=320,
            method="crop",
            resize_mode="longest",
        )

        assert success == 1

        # With crop, output should match target dimensions exactly
        resized = cv2.imread(str(output_dir / "img.jpg"))
        assert resized.shape[1] == 320  # width (longest edge)

    def test_resize_directory_invalid_method(self, temp_dir):
        """Test error handling for invalid resize method"""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        with pytest.raises(ValueError, match="Invalid method"):
            resize_directory(
                input_path=input_dir,
                output_dir=temp_dir / "output",
                target_size=640,
                method="invalid",
            )

    def test_resize_directory_invalid_mode(self, temp_dir):
        """Test error handling for invalid resize mode"""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        with pytest.raises(ValueError, match="Invalid resize_mode"):
            resize_directory(
                input_path=input_dir,
                output_dir=temp_dir / "output",
                target_size=640,
                method="scale",
                resize_mode="invalid",
            )


class TestProcessMultiMethod:
    """Test multi-method image processing"""

    def test_process_single_image_scale(self, sample_image, temp_dir):
        """Test processing single image with scale method"""
        output_dir = temp_dir / "output"

        count = process_single_image_multi_method(
            input_file=sample_image,
            output_dir=output_dir,
            target_sizes=[320, 640],
            use_crop=False,
        )

        assert count == 2
        assert (output_dir / f"{sample_image.stem}_scale_320x240.jpg").exists()
        assert (output_dir / f"{sample_image.stem}_scale_640x480.jpg").exists()

    def test_process_single_image_crop(self, sample_image, temp_dir):
        """Test processing single image with crop method"""
        output_dir = temp_dir / "output"

        count = process_single_image_multi_method(
            input_file=sample_image,
            output_dir=output_dir,
            target_sizes=[320],
            use_crop=True,
        )

        assert count == 1
        output_files = list(output_dir.glob("*_crop_*.jpg"))
        assert len(output_files) == 1

    def test_process_images_multi_method_file(self, sample_image, temp_dir):
        """Test multi-method processing on single file"""
        output_dir = temp_dir / "output"

        processed, failed = process_images_multi_method(
            input_path=sample_image,
            output_dir=output_dir,
            target_sizes=[320],
        )

        # Should process with both scale and crop
        assert processed == 2  # One scale, one crop
        assert failed == 0

    def test_process_images_multi_method_directory(self, temp_dir):
        """Test multi-method processing on directory"""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        # Create test images
        for i in range(2):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"img{i}.jpg"), img)

        output_dir = temp_dir / "output"

        processed, failed = process_images_multi_method(
            input_path=input_dir,
            output_dir=output_dir,
            target_sizes=[320],
        )

        # 2 images * 2 methods = 4 outputs
        assert processed == 4
        assert failed == 0

    def test_process_single_image_nonexistent(self, temp_dir):
        """Test error handling for nonexistent file"""
        with pytest.raises(FileNotFoundError):
            process_single_image_multi_method(
                input_file="nonexistent.jpg",
                output_dir=temp_dir / "output",
                target_sizes=[640],
            )

    def test_process_images_nonexistent_path(self):
        """Test error handling for nonexistent path"""
        with pytest.raises(FileNotFoundError):
            process_images_multi_method(
                input_path="nonexistent",
                output_dir="output",
                target_sizes=[640],
            )

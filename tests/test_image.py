"""
Test image processing module functionality
"""

import cv2
import numpy as np

from ydt.image.augment import augment_dataset, rotate_image_with_labels
from ydt.image.concat import concat_images_horizontally, concat_images_vertically
from ydt.image.slice import slice_dataset
from ydt.image.video import extract_frames


class TestVideoExtraction:
    """Test video frame extraction"""

    def test_extract_frames_basic(self, sample_video, temp_dir):
        """Test basic frame extraction"""
        output_dir = temp_dir / "frames"

        count = extract_frames(
            video_path=str(sample_video),
            frames_output_dir=str(output_dir),
            step=5,
        )

        assert count > 0
        # Frames are saved to video_name_frames subdirectory
        frames = list(output_dir.rglob("*.jpg"))
        assert len(frames) > 0


class TestImageRotation:
    """Test image rotation with labels"""

    def test_rotate_image_90_degrees(self, temp_dir):
        """Test rotating image 90 degrees with bbox labels"""
        # Create test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create bbox label lines (string format)
        label_lines = ["0 0.5 0.5 0.2 0.3"]  # class, cx, cy, w, h

        rotated_img, rotated_labels = rotate_image_with_labels(
            img, label_lines, angle=90, format_type="bbox"
        )

        assert rotated_img.shape[0] == 640  # Height becomes old width
        assert rotated_img.shape[1] == 480  # Width becomes old height
        assert len(rotated_labels) == 1

    def test_rotate_image_with_obb(self, temp_dir):
        """Test rotating image with OBB labels"""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # OBB format: class, x1, y1, x2, y2, x3, y3, x4, y4 (normalized)
        label_lines = ["0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7"]

        rotated_img, rotated_labels = rotate_image_with_labels(
            img, label_lines, angle=45, format_type="obb"
        )

        assert rotated_img.shape[0] > 0
        assert rotated_img.shape[1] > 0
        assert len(rotated_labels) == 1


class TestImageSlicing:
    """Test image slicing/tiling"""

    def test_slice_single_image(self, temp_dir):
        """Test slicing a single image"""
        # Create test image and label
        input_dir = temp_dir / "input"
        (input_dir / "images" / "train").mkdir(parents=True)
        (input_dir / "labels" / "train").mkdir(parents=True)

        img = np.random.randint(0, 255, (1280, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "images" / "train" / "test.jpg"), img)

        label_path = input_dir / "labels" / "train" / "test.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        output_dir = temp_dir / "sliced"

        # slice_dataset uses horizontal_count and overlap_ratio_horizontal
        result = slice_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            horizontal_count=3,
            overlap_ratio_horizontal=0.2,
        )

        # Check that slices were created
        assert result["total_slices"] > 0


class TestDataAugmentation:
    """Test data augmentation"""

    def test_augment_dataset_basic(self, temp_dir):
        """Test basic dataset augmentation"""
        # Create simple dataset structure
        input_dir = temp_dir / "dataset"
        (input_dir / "images" / "train").mkdir(parents=True)
        (input_dir / "labels" / "train").mkdir(parents=True)

        # Create test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "images" / "train" / "img.jpg"), img)

        # Create label
        label_path = input_dir / "labels" / "train" / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        output_dir = temp_dir / "augmented"

        # augment_dataset uses dataset_path and output_path
        result = augment_dataset(
            dataset_path=str(input_dir),
            output_path=str(output_dir),
            angles=[90],
            format_type="bbox",
        )

        assert result["processed"] >= 1

    def test_augment_with_multiple_angles(self, temp_dir):
        """Test augmentation with multiple rotation angles"""
        input_dir = temp_dir / "dataset"
        (input_dir / "images" / "train").mkdir(parents=True)
        (input_dir / "labels" / "train").mkdir(parents=True)

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "images" / "train" / "img.jpg"), img)

        label_path = input_dir / "labels" / "train" / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        output_dir = temp_dir / "augmented"

        result = augment_dataset(
            dataset_path=str(input_dir),
            output_path=str(output_dir),
            angles=[90, 180, 270],
            format_type="bbox",
        )

        assert result["rotations"] >= 1


class TestImageConcatenation:
    """Test image concatenation"""

    def test_concat_horizontally(self, temp_dir):
        """Test horizontal image concatenation"""
        # Create two test images and save to files
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        img1_path = temp_dir / "img1.jpg"
        img2_path = temp_dir / "img2.jpg"
        output_path = temp_dir / "concat_h.jpg"

        cv2.imwrite(str(img1_path), img1)
        cv2.imwrite(str(img2_path), img2)

        # concat_images_horizontally takes file paths
        result_path = concat_images_horizontally(str(img1_path), str(img2_path), str(output_path))

        assert result_path.exists()
        result = cv2.imread(str(result_path))
        assert result.shape[0] == 480
        assert result.shape[1] == 1280  # 640 * 2

    def test_concat_vertically(self, temp_dir):
        """Test vertical image concatenation"""
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        img1_path = temp_dir / "img1.jpg"
        img2_path = temp_dir / "img2.jpg"
        output_path = temp_dir / "concat_v.jpg"

        cv2.imwrite(str(img1_path), img1)
        cv2.imwrite(str(img2_path), img2)

        result_path = concat_images_vertically(str(img1_path), str(img2_path), str(output_path))

        assert result_path.exists()
        result = cv2.imread(str(result_path))
        assert result.shape[0] == 960  # 480 * 2
        assert result.shape[1] == 640

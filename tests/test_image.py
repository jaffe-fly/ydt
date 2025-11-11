"""
Test image processing module functionality
"""

from pathlib import Path

import cv2
import pytest

from ydt.image.augment import augment_dataset, rotate_image_with_labels
from ydt.image.slice import slice_dataset
from ydt.image.video import extract_frames


class TestVideoExtraction:
    """Test video frame extraction"""

    def test_extract_frames_single_video(self, sample_video, temp_dir):
        """Test frame extraction from single video"""
        output_dir = temp_dir / "frames"

        count = extract_frames(video_path=sample_video, frames_output_dir=output_dir, step=3)

        assert count > 0
        assert output_dir.exists()
        frames = list(output_dir.rglob("*.jpg"))
        assert len(frames) == count

    def test_extract_frames_directory(self, sample_video, temp_dir):
        """Test frame extraction from directory of videos"""
        video_dir = temp_dir / "videos"
        video_dir.mkdir()
        output_dir = temp_dir / "frames"

        # Copy video to directory
        import shutil

        shutil.copy(sample_video, video_dir / "video1.mp4")

        count = extract_frames(video_path=video_dir, frames_output_dir=output_dir, step=5)

        assert count > 0
        assert output_dir.exists()

        # Check video-specific directory
        video_output_dir = output_dir / "video1_frames"
        assert video_output_dir.exists()
        frames = list(video_output_dir.rglob("*.jpg"))
        assert len(frames) > 0

    def test_extract_frames_nonexistent_input(self, temp_dir):
        """Test error handling for nonexistent input"""
        with pytest.raises(FileNotFoundError):
            extract_frames("nonexistent.mp4", temp_dir / "output")

    def test_extract_frames_unsupported_format(self, temp_dir):
        """Test error handling for unsupported video format"""
        unsupported_file = temp_dir / "test.txt"
        unsupported_file.write_text("not a video")

        with pytest.raises(ValueError):
            extract_frames(unsupported_file, temp_dir / "output")


class TestImageRotation:
    """Test image rotation with label transformation"""

    def test_rotate_image_90_degrees(self, sample_image, sample_obb_labels):
        """Test 90-degree rotation"""
        # Read original image
        image = cv2.imread(str(sample_image))
        with open(sample_obb_labels) as f:
            labels = [line.strip() for line in f.readlines()]

        # Rotate 90 degrees
        rotated_image, rotated_labels = rotate_image_with_labels(
            image, labels, 90, format_type="obb"
        )

        # Check image dimensions
        original_height, original_width = image.shape[:2]
        rotated_height, rotated_width = rotated_image.shape[:2]
        assert rotated_height == original_width
        assert rotated_width == original_height

        # Check labels are transformed
        assert len(rotated_labels) == len(labels)

    def test_rotate_image_180_degrees(self, sample_image, sample_obb_labels):
        """Test 180-degree rotation"""
        image = cv2.imread(str(sample_image))
        with open(sample_obb_labels) as f:
            labels = [line.strip() for line in f.readlines()]

        rotated_image, rotated_labels = rotate_image_with_labels(
            image, labels, 180, format_type="obb"
        )

        # Image dimensions should be the same
        assert rotated_image.shape[:2] == image.shape[:2]
        assert len(rotated_labels) == len(labels)

    def test_rotate_bbox_format(self, sample_image, sample_bbox_labels):
        """Test rotation with BBox format"""
        image = cv2.imread(str(sample_image))
        with open(sample_bbox_labels) as f:
            labels = [line.strip() for line in f.readlines()]

        rotated_image, rotated_labels = rotate_image_with_labels(
            image, labels, 90, format_type="bbox"
        )

        assert len(rotated_labels) == len(labels)


class TestImageSlicing:
    """Test image slicing functionality"""

    def test_slice_dataset_basic(self, sample_dataset, temp_dir):
        """Test basic dataset slicing"""
        output_dir = temp_dir / "sliced"

        result = slice_dataset(
            input_dir=sample_dataset,
            output_dir=output_dir,
            horizontal_count=2,
            overlap_ratio_horizontal=0.1,
        )

        assert "processed_files" in result
        assert "total_slices" in result
        assert result["processed_files"] > 0
        assert result["total_slices"] > 0

        # Check output structure
        assert output_dir.exists()
        sliced_images = list(output_dir.rglob("*.jpg"))
        assert len(sliced_images) > 0

        # Check corresponding label files
        for img_path in sliced_images[:3]:  # Check first few
            # Convert image path to label path (images -> labels, .jpg -> .txt)
            label_path = Path(
                str(img_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
            ).with_suffix(".txt")
            assert label_path.exists()

    def test_slice_dataset_nonexistent_input(self, temp_dir):
        """Test error handling for nonexistent input"""
        with pytest.raises(FileNotFoundError):
            slice_dataset(input_dir="nonexistent", output_dir=temp_dir / "output")


# TestImageResizing class removed - resize_images function does not exist
# Use process_images_multi_method from ydt.image.resize instead


class TestDataAugmentation:
    """Test data augmentation functionality"""

    def test_augment_dataset_auto_angles(self, sample_dataset, temp_dir):
        """Test dataset augmentation with auto-selected angles"""
        output_dir = temp_dir / "augmented"

        result = augment_dataset(dataset_path=sample_dataset, output_path=output_dir)

        assert "processed" in result
        assert "rotations" in result
        assert result["processed"] > 0
        assert result["rotations"] > 0
        assert output_dir.exists()

        # Check augmented images
        aug_images = list(output_dir.rglob("*.jpg"))
        assert len(aug_images) > 0

        # Check corresponding label files
        for img_path in aug_images[:3]:
            # Convert image path to label path (images -> labels, .jpg -> .txt)
            label_path = Path(
                str(img_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
            ).with_suffix(".txt")
            assert label_path.exists()

    def test_augment_dataset_specific_angles(self, sample_dataset, temp_dir):
        """Test dataset augmentation with specific angles"""
        output_dir = temp_dir / "augmented"

        result = augment_dataset(
            dataset_path=sample_dataset, output_path=output_dir, angles=[0, 90, 180]
        )

        assert "processed" in result
        assert result["processed"] > 0

    def test_augment_dataset_nonexistent_yaml(self, temp_dir):
        """Test error handling for nonexistent YAML"""
        with pytest.raises(FileNotFoundError):
            augment_dataset(dataset_path="nonexistent.yaml", output_path=temp_dir / "output")

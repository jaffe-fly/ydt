"""
Comprehensive tests for visualization module.

Tests cover:
- OBB drawing functionality
- Dataset visualization with filtering
- Support for both OBB and HBB formats
- Single image visualization
- Interactive keyboard controls
- Error handling
"""

import cv2
import numpy as np
import pytest

from ydt.visual.dataset import draw_obb, visualize_dataset


class TestDrawOBB:
    """Test OBB drawing functionality"""

    def test_draw_obb_basic(self):
        """Test basic OBB drawing"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw a square OBB
        points = [100, 100, 200, 100, 200, 200, 100, 200]

        draw_obb(points, img, color=(0, 255, 0), label="test")

        # Check that image was modified (not all black anymore)
        assert np.sum(img) > 0

    def test_draw_obb_no_color(self):
        """Test OBB drawing with random color"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        points = [100, 100, 200, 100, 200, 200, 100, 200]

        # No color specified - should use random color
        draw_obb(points, img)

        assert np.sum(img) > 0

    def test_draw_obb_no_label(self):
        """Test OBB drawing without label"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        points = [100, 100, 200, 100, 200, 200, 100, 200]

        draw_obb(points, img, color=(255, 0, 0))

        assert np.sum(img) > 0

    def test_draw_obb_custom_thickness(self):
        """Test OBB drawing with custom line thickness"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        points = [100, 100, 200, 100, 200, 200, 100, 200]

        draw_obb(points, img, color=(0, 0, 255), line_thickness=5)

        assert np.sum(img) > 0

    def test_draw_obb_rotated(self):
        """Test drawing rotated OBB"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Rotated rectangle
        points = [150, 100, 250, 120, 230, 180, 130, 160]

        draw_obb(points, img, color=(0, 255, 255), label="rotated")

        assert np.sum(img) > 0

    def test_draw_obb_multiple(self):
        """Test drawing multiple OBBs"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Draw multiple OBBs
        obbs = [
            ([100, 100, 200, 100, 200, 200, 100, 200], (255, 0, 0), "obj1"),
            ([300, 150, 400, 150, 400, 250, 300, 250], (0, 255, 0), "obj2"),
        ]

        for points, color, label in obbs:
            draw_obb(points, img, color=color, label=label)

        assert np.sum(img) > 0


class TestVisualizeDataset:
    """Test dataset visualization functionality"""

    def test_visualize_dataset_basic(self, sample_dataset):
        """Test basic dataset visualization"""
        # Use wait_key=1 to avoid blocking
        count = visualize_dataset(
            dataset_path=sample_dataset,
            scan_train=True,
            scan_val=False,
            wait_key=1,
        )

        assert count >= 0

    def test_visualize_dataset_both_splits(self, temp_dir):
        """Test visualizing both train and val splits"""
        # Create dataset with both splits
        dataset_dir = temp_dir / "dataset"
        for split in ["train", "val"]:
            (dataset_dir / "images" / split).mkdir(parents=True)
            (dataset_dir / "labels" / split).mkdir(parents=True)

            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(dataset_dir / "images" / split / "img.jpg"), img)

            label_path = dataset_dir / "labels" / split / "img.txt"
            label_path.write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(
            "path: .\ntrain: images/train\nval: images/val\nnc: 2\nnames: ['class_0', 'class_1']\n"
        )

        count = visualize_dataset(
            dataset_path=dataset_dir,
            scan_train=True,
            scan_val=True,
            wait_key=1,
        )

        assert count == 2  # One from train, one from val

    def test_visualize_dataset_with_filter(self, temp_dir):
        """Test dataset visualization with label filtering"""
        # Create dataset with multiple classes
        dataset_dir = temp_dir / "dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        # Image 1: class 0
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / "img1.jpg"), img1)
        (dataset_dir / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        # Image 2: class 1
        img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / "img2.jpg"), img2)
        (dataset_dir / "labels" / "train" / "img2.txt").write_text("1 0.5 0.5 0.2 0.2\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 2\nnames: ['cat', 'dog']\n")

        # Filter to only show 'cat' class
        count = visualize_dataset(
            dataset_path=dataset_dir,
            filter_labels=["cat"],
            scan_train=True,
            wait_key=1,
        )

        assert count == 1  # Only one image with 'cat' label

    def test_visualize_single_image(self, temp_dir):
        """Test visualizing a single image"""
        # Create dataset
        dataset_dir = temp_dir / "dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        img_path = dataset_dir / "images" / "train" / "test.jpg"
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        label_path = dataset_dir / "labels" / "train" / "test.txt"
        label_path.write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: ['object']\n")

        count = visualize_dataset(
            dataset_path=dataset_dir,
            single_image_path=img_path,
            wait_key=1,
        )

        assert count == 1

    def test_visualize_dataset_obb_format(self, temp_dir):
        """Test visualizing dataset with OBB format"""
        dataset_dir = temp_dir / "obb_dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / "img.jpg"), img)

        # OBB format label
        label_path = dataset_dir / "labels" / "train" / "img.txt"
        label_path.write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: ['card']\n")

        count = visualize_dataset(
            dataset_path=dataset_dir,
            scan_train=True,
            wait_key=1,
        )

        assert count == 1

    def test_visualize_dataset_bbox_format(self, temp_dir):
        """Test visualizing dataset with BBox format"""
        dataset_dir = temp_dir / "bbox_dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / "img.jpg"), img)

        # BBox format label
        label_path = dataset_dir / "labels" / "train" / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: ['object']\n")

        count = visualize_dataset(
            dataset_path=dataset_dir,
            scan_train=True,
            wait_key=1,
        )

        assert count == 1

    def test_visualize_dataset_nonexistent_path(self):
        """Test error handling for nonexistent dataset path"""
        with pytest.raises(FileNotFoundError):
            visualize_dataset(
                dataset_path="nonexistent_path",
                wait_key=1,
            )

    def test_visualize_dataset_no_data_yaml(self, temp_dir):
        """Test error handling when data.yaml is missing"""
        dataset_dir = temp_dir / "no_yaml"
        dataset_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="data.yaml not found"):
            visualize_dataset(
                dataset_path=dataset_dir,
                wait_key=1,
            )

    def test_visualize_dataset_invalid_split_params(self, sample_dataset):
        """Test error handling for invalid split parameters"""
        with pytest.raises(ValueError, match="At least one of scan_train or scan_val"):
            visualize_dataset(
                dataset_path=sample_dataset,
                scan_train=False,
                scan_val=False,
                wait_key=1,
            )

    def test_visualize_dataset_empty_dataset(self, temp_dir):
        """Test visualizing empty dataset"""
        dataset_dir = temp_dir / "empty"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: ['class_0']\n")

        count = visualize_dataset(
            dataset_path=dataset_dir,
            scan_train=True,
            wait_key=1,
        )

        assert count == 0

    def test_visualize_dataset_with_list_names(self, temp_dir):
        """Test visualization with list-format class names in YAML"""
        dataset_dir = temp_dir / "dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / "img.jpg"), img)

        label_path = dataset_dir / "labels" / "train" / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        # Use list format for names
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 2\nnames: ['cat', 'dog']\n")

        count = visualize_dataset(
            dataset_path=dataset_dir,
            scan_train=True,
            wait_key=1,
        )

        assert count == 1

    def test_visualize_single_image_nonexistent(self, sample_dataset):
        """Test error handling for nonexistent single image"""
        with pytest.raises(FileNotFoundError):
            visualize_dataset(
                dataset_path=sample_dataset,
                single_image_path="nonexistent.jpg",
                wait_key=1,
            )

    def test_visualize_dataset_mixed_formats(self, temp_dir):
        """Test visualization with mixed valid and invalid labels"""
        dataset_dir = temp_dir / "dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        # Create image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / "img.jpg"), img)

        # Label with valid and invalid lines
        label_path = dataset_dir / "labels" / "train" / "img.txt"
        label_path.write_text(
            "0 0.5 0.5 0.2 0.2\n"  # Valid bbox
            "invalid line\n"  # Invalid
            "1 0.3 0.3 0.1 0.1\n"  # Valid bbox
        )

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 2\nnames: ['a', 'b']\n")

        # Should handle gracefully and visualize valid labels
        count = visualize_dataset(
            dataset_path=dataset_dir,
            scan_train=True,
            wait_key=1,
        )

        assert count == 1

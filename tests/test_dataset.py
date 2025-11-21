"""
Comprehensive tests for dataset module functionality.

Tests cover:
- Dataset splitting with class balancing
- Dataset merging with duplicate handling
- Dataset synthesis (OBB and HBB formats)
- Dataset analysis and statistics
- Label counting
"""

import cv2
import numpy as np
import pytest
import yaml

from ydt.dataset import (
    DatasetSynthesizer,
    analyze_dataset,
    count_labels,
    merge_datasets,
    split_dataset,
)


class TestDatasetSplit:
    """Test dataset splitting functionality"""

    def test_split_dataset_basic(self, sample_dataset, temp_dir):
        """Test basic dataset splitting"""
        output_dir = temp_dir / "split_output"

        result = split_dataset(
            data_yaml_path=sample_dataset / "data.yaml",
            output_dir=output_dir,
            train_ratio=0.8,
        )

        assert result["train_count"] >= 0
        assert result["val_count"] >= 0
        assert result["train_count"] + result["val_count"] > 0
        assert result["output_dir"] == str(output_dir)

        # Check directory structure
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "images" / "val").exists()
        assert (output_dir / "labels" / "train").exists()
        assert (output_dir / "labels" / "val").exists()
        assert (output_dir / "data.yaml").exists()

    def test_split_dataset_with_directory_input(self, sample_dataset, temp_dir):
        """Test split with directory input (auto-detect data.yaml)"""
        output_dir = temp_dir / "split_output"

        result = split_dataset(
            data_yaml_path=sample_dataset,  # Pass directory instead of yaml file
            output_dir=output_dir,
            train_ratio=0.8,
        )

        assert result["train_count"] + result["val_count"] > 0

    def test_split_dataset_different_ratios(self, sample_dataset, temp_dir):
        """Test splitting with different train ratios"""
        ratios = [0.7, 0.8, 0.9]

        for ratio in ratios:
            output_dir = temp_dir / f"split_{int(ratio * 100)}"

            result = split_dataset(
                data_yaml_path=sample_dataset / "data.yaml",
                output_dir=output_dir,
                train_ratio=ratio,
            )

            total_count = result["train_count"] + result["val_count"]
            assert total_count > 0

    def test_split_dataset_with_multiple_classes(self, temp_dir):
        """Test splitting dataset with multiple classes"""
        # Create dataset with multiple classes
        dataset_dir = temp_dir / "multi_class_dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        # Create images and labels for multiple classes
        for i in range(10):
            # Create image
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img_path = dataset_dir / "images" / "train" / f"img{i:03d}.jpg"
            cv2.imwrite(str(img_path), img)

            # Create labels with different classes
            label_path = dataset_dir / "labels" / "train" / f"img{i:03d}.txt"
            class_id = i % 3  # 3 classes (0, 1, 2)
            label_path.write_text(f"{class_id} 0.5 0.5 0.2 0.2\n")

        # Create data.yaml
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(
            """
path: .
train: images/train
nc: 3
names: ['class_0', 'class_1', 'class_2']
"""
        )

        output_dir = temp_dir / "split_multi_class"

        result = split_dataset(
            data_yaml_path=data_yaml,
            output_dir=output_dir,
            train_ratio=0.7,
            balance_classes=True,
        )

        assert result["train_count"] > 0
        assert result["val_count"] > 0
        # With balanced classes, each class should appear in both sets
        assert result["train_count"] + result["val_count"] == 10

    def test_split_dataset_invalid_ratio(self, sample_dataset, temp_dir):
        """Test error handling for invalid train ratio"""
        output_dir = temp_dir / "split_output"

        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            split_dataset(
                data_yaml_path=sample_dataset / "data.yaml",
                output_dir=output_dir,
                train_ratio=1.5,
            )

        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            split_dataset(
                data_yaml_path=sample_dataset / "data.yaml",
                output_dir=output_dir,
                train_ratio=-0.1,
            )

    def test_split_dataset_nonexistent_yaml(self, temp_dir):
        """Test error handling for nonexistent YAML file"""
        with pytest.raises(FileNotFoundError):
            split_dataset(
                data_yaml_path="nonexistent.yaml",
                output_dir=temp_dir / "output",
            )

    def test_split_dataset_no_images(self, temp_dir):
        """Test error handling for dataset with no images"""
        # Create empty dataset
        dataset_dir = temp_dir / "empty_dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(
            """
path: .
train: images/train
nc: 1
names: ['class_0']
"""
        )

        output_dir = temp_dir / "split_output"

        result = split_dataset(
            data_yaml_path=data_yaml,
            output_dir=output_dir,
            train_ratio=0.8,
        )

        # Should handle gracefully with 0 images
        assert result["train_count"] == 0
        assert result["val_count"] == 0


class TestDatasetMerge:
    """Test dataset merging functionality"""

    def test_merge_two_datasets(self, temp_dir):
        """Test merging two datasets"""
        # Create two simple datasets
        datasets = []
        for i in range(2):
            dataset_dir = temp_dir / f"dataset_{i}"
            (dataset_dir / "images" / "train").mkdir(parents=True)
            (dataset_dir / "labels" / "train").mkdir(parents=True)

            # Create a few images
            for j in range(3):
                img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                img_path = dataset_dir / "images" / "train" / f"img{j}.jpg"
                cv2.imwrite(str(img_path), img)

                label_path = dataset_dir / "labels" / "train" / f"img{j}.txt"
                label_path.write_text("0 0.5 0.5 0.2 0.2\n")

            # Create data.yaml
            data_yaml = dataset_dir / "data.yaml"
            data_yaml.write_text(
                """
path: .
train: images/train
nc: 1
names: ['class_0']
"""
            )

            datasets.append(dataset_dir)

        output_dir = temp_dir / "merged"

        result = merge_datasets(
            dataset_dirs=datasets,
            output_dir=output_dir,
        )

        assert result["train_images"] == 6  # 3 + 3
        assert result["train_labels"] == 6
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "labels" / "train").exists()
        assert (output_dir / "data.yaml").exists()

    def test_merge_with_duplicates(self, temp_dir):
        """Test merging with duplicate filenames - auto-renames with counter suffix"""
        # Create two datasets with same filenames
        datasets = []
        for i in range(2):
            dataset_dir = temp_dir / f"dataset_{i}"
            (dataset_dir / "images" / "train").mkdir(parents=True)
            (dataset_dir / "labels" / "train").mkdir(parents=True)

            # Use same filename in both datasets
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img_path = dataset_dir / "images" / "train" / "duplicate.jpg"
            cv2.imwrite(str(img_path), img)

            label_path = dataset_dir / "labels" / "train" / "duplicate.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n")

            # Create data.yaml
            data_yaml = dataset_dir / "data.yaml"
            data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: ['class_0']\n")

            datasets.append(dataset_dir)

        output_dir = temp_dir / "merged"

        result = merge_datasets(
            dataset_dirs=datasets,
            output_dir=output_dir,
        )

        # Should have both files (original and renamed)
        assert result["train_images"] == 2
        train_images = list((output_dir / "images" / "train").glob("*.jpg"))
        assert len(train_images) == 2

    def test_merge_train_and_val(self, temp_dir):
        """Test merging both train and val sets"""
        # Create dataset with train and val
        dataset_dir = temp_dir / "dataset_1"
        for split in ["train", "val"]:
            (dataset_dir / "images" / split).mkdir(parents=True)
            (dataset_dir / "labels" / split).mkdir(parents=True)

            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img_path = dataset_dir / "images" / split / "img.jpg"
            cv2.imwrite(str(img_path), img)

            label_path = dataset_dir / "labels" / split / "img.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(
            "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['class_0']\n"
        )

        output_dir = temp_dir / "merged"

        result = merge_datasets(
            dataset_dirs=[dataset_dir],
            output_dir=output_dir,
            merge_train=True,
            merge_val=True,
        )

        assert result["train_images"] == 1
        assert result["val_images"] == 1

    def test_merge_invalid_parameters(self, temp_dir):
        """Test error handling for invalid parameters"""
        with pytest.raises(ValueError, match="At least one of merge_train or merge_val"):
            merge_datasets(
                dataset_dirs=[temp_dir],
                output_dir=temp_dir / "output",
                merge_train=False,
                merge_val=False,
            )

    def test_merge_nonexistent_dataset(self, temp_dir):
        """Test error handling for nonexistent dataset"""
        with pytest.raises(FileNotFoundError):
            merge_datasets(
                dataset_dirs=["nonexistent_dir"],
                output_dir=temp_dir / "output",
            )


class TestDatasetSynthesizer:
    """Test dataset synthesis functionality"""

    def test_synthesizer_init_obb_format(
        self, synthetic_objects_dir, synthetic_backgrounds_dir, temp_dir
    ):
        """Test synthesizer initialization with OBB format"""
        output_dir = temp_dir / "synthetic_obb"

        synthesizer = DatasetSynthesizer(
            target_dir=synthetic_objects_dir,
            background_dir=synthetic_backgrounds_dir,
            output_dir=output_dir,
            annotation_format="obb",
        )

        assert synthesizer.annotation_format == "obb"
        assert len(synthesizer.target_data) > 0
        assert len(synthesizer.background_images) > 0

    def test_synthesizer_init_hbb_format(
        self, synthetic_objects_dir, synthetic_backgrounds_dir, temp_dir
    ):
        """Test synthesizer initialization with HBB format"""
        output_dir = temp_dir / "synthetic_hbb"

        synthesizer = DatasetSynthesizer(
            target_dir=synthetic_objects_dir,
            background_dir=synthetic_backgrounds_dir,
            output_dir=output_dir,
            annotation_format="hbb",
        )

        assert synthesizer.annotation_format == "hbb"

    def test_synthesizer_invalid_format(
        self, synthetic_objects_dir, synthetic_backgrounds_dir, temp_dir
    ):
        """Test error handling for invalid annotation format"""
        with pytest.raises(ValueError, match="annotation_format must be 'obb' or 'hbb'"):
            DatasetSynthesizer(
                target_dir=synthetic_objects_dir,
                background_dir=synthetic_backgrounds_dir,
                output_dir=temp_dir / "output",
                annotation_format="invalid",
            )

    def test_synthesizer_with_data_yaml(
        self, synthetic_objects_dir, synthetic_backgrounds_dir, temp_dir
    ):
        """Test synthesizer with data.yaml validation"""
        # Create data.yaml
        data_yaml = temp_dir / "data.yaml"
        data_yaml.write_text(
            """
nc: 2
names:
  0: object
  1: item
"""
        )

        # Rename object files to match class names
        for obj_file in synthetic_objects_dir.glob("*.png"):
            new_name = f"object_{obj_file.stem}.png"
            obj_file.rename(synthetic_objects_dir / new_name)

        output_dir = temp_dir / "synthetic"

        synthesizer = DatasetSynthesizer(
            target_dir=synthetic_objects_dir,
            background_dir=synthetic_backgrounds_dir,
            output_dir=output_dir,
            data_yaml_path=data_yaml,
        )

        assert synthesizer.class_names == {0: "object", 1: "item"}

    def test_synthesizer_rotation_range(
        self, synthetic_objects_dir, synthetic_backgrounds_dir, temp_dir
    ):
        """Test synthesizer with custom rotation range"""
        output_dir = temp_dir / "synthetic"

        synthesizer = DatasetSynthesizer(
            target_dir=synthetic_objects_dir,
            background_dir=synthetic_backgrounds_dir,
            output_dir=output_dir,
            rotation_range=(-45, 45),
        )

        assert synthesizer.rotation_range == (-45, 45)

    def test_synthesizer_objects_per_image_single(
        self, synthetic_objects_dir, synthetic_backgrounds_dir, temp_dir
    ):
        """Test synthesizer with fixed objects per image"""
        output_dir = temp_dir / "synthetic"

        synthesizer = DatasetSynthesizer(
            target_dir=synthetic_objects_dir,
            background_dir=synthetic_backgrounds_dir,
            output_dir=output_dir,
            objects_per_image=3,
        )

        assert synthesizer.min_objects_per_image == 3
        assert synthesizer.max_objects_per_image == 3

    def test_synthesizer_objects_per_image_range(
        self, synthetic_objects_dir, synthetic_backgrounds_dir, temp_dir
    ):
        """Test synthesizer with range of objects per image"""
        output_dir = temp_dir / "synthetic"

        synthesizer = DatasetSynthesizer(
            target_dir=synthetic_objects_dir,
            background_dir=synthetic_backgrounds_dir,
            output_dir=output_dir,
            objects_per_image=(2, 5),
        )

        assert synthesizer.min_objects_per_image == 2
        assert synthesizer.max_objects_per_image == 5

    def test_synthesize_dataset_train_only(
        self, synthetic_objects_dir, synthetic_backgrounds_dir, temp_dir
    ):
        """Test dataset synthesis with train-only mode"""
        output_dir = temp_dir / "synthetic"

        synthesizer = DatasetSynthesizer(
            target_dir=synthetic_objects_dir,
            background_dir=synthetic_backgrounds_dir,
            output_dir=output_dir,
            split_mode="train",
        )

        result = synthesizer.synthesize_dataset(num_images=5)

        assert result["train_count"] == 5
        assert result["val_count"] == 0

        # Check files were created
        train_images = list((output_dir / "images" / "train").glob("*.jpg"))
        assert len(train_images) == 5

        # Check labels were created
        for img in train_images:
            label_path = output_dir / "labels" / "train" / f"{img.stem}.txt"
            assert label_path.exists()

    def test_synthesize_dataset_trainval(
        self, synthetic_objects_dir, synthetic_backgrounds_dir, temp_dir
    ):
        """Test dataset synthesis with train/val split"""
        output_dir = temp_dir / "synthetic"

        synthesizer = DatasetSynthesizer(
            target_dir=synthetic_objects_dir,
            background_dir=synthetic_backgrounds_dir,
            output_dir=output_dir,
            split_mode="trainval",
            train_ratio=0.8,
        )

        result = synthesizer.synthesize_dataset(num_images=10)

        assert result["train_count"] == 8
        assert result["val_count"] == 2

        # Check files were created
        train_images = list((output_dir / "images" / "train").glob("*.jpg"))
        val_images = list((output_dir / "images" / "val").glob("*.jpg"))
        assert len(train_images) == 8
        assert len(val_images) == 2

    def test_synthesize_dataset_creates_yaml(
        self, synthetic_objects_dir, synthetic_backgrounds_dir, temp_dir
    ):
        """Test that synthesis creates data.yaml"""
        output_dir = temp_dir / "synthetic"

        synthesizer = DatasetSynthesizer(
            target_dir=synthetic_objects_dir,
            background_dir=synthetic_backgrounds_dir,
            output_dir=output_dir,
            class_names={0: "object_a", 1: "object_b"},
        )

        synthesizer.synthesize_dataset(num_images=2)

        # Check data.yaml was created
        yaml_path = output_dir / "data.yaml"
        assert yaml_path.exists()

        # Verify content
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            assert "names" in data
            assert "train" in data

    def test_synthesizer_no_targets(self, temp_dir):
        """Test error handling when no target images found"""
        empty_dir = temp_dir / "empty_targets"
        empty_dir.mkdir()

        backgrounds_dir = temp_dir / "backgrounds"
        backgrounds_dir.mkdir()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(backgrounds_dir / "bg.jpg"), img)

        with pytest.raises(RuntimeError, match="No valid target images found"):
            DatasetSynthesizer(
                target_dir=empty_dir,
                background_dir=backgrounds_dir,
                output_dir=temp_dir / "output",
            )

    def test_synthesizer_no_backgrounds(self, temp_dir):
        """Test error handling when no background images found"""
        targets_dir = temp_dir / "targets"
        targets_dir.mkdir()
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :, 3] = 255  # Alpha channel
        cv2.imwrite(str(targets_dir / "target.png"), img)

        empty_dir = temp_dir / "empty_backgrounds"
        empty_dir.mkdir()

        with pytest.raises(RuntimeError, match="No valid background images found"):
            DatasetSynthesizer(
                target_dir=targets_dir,
                background_dir=empty_dir,
                output_dir=temp_dir / "output",
            )


class TestCountLabels:
    """Test label counting functionality"""

    def test_count_labels_train(self, sample_dataset):
        """Test counting labels in training set"""
        counts = count_labels(
            dataset_path=sample_dataset,
            split="train",
            show_details=False,
        )

        assert isinstance(counts, dict)
        assert len(counts) > 0
        # Check that class IDs are in the result
        assert 0 in counts or 1 in counts

    def test_count_labels_val(self, temp_dir):
        """Test counting labels in validation set"""
        # Create dataset with val set
        dataset_dir = temp_dir / "dataset"
        (dataset_dir / "images" / "val").mkdir(parents=True)
        (dataset_dir / "labels" / "val").mkdir(parents=True)

        # Create sample image and label
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "val" / "img.jpg"), img)

        label_path = dataset_dir / "labels" / "val" / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")

        # Create data.yaml
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\nval: images/val\nnc: 2\nnames: ['class_0', 'class_1']\n")

        counts = count_labels(
            dataset_path=dataset_dir,
            split="val",
            show_details=False,
        )

        assert counts[0] == 1
        assert counts[1] == 1

    def test_count_labels_both(self, temp_dir):
        """Test counting labels in both train and val"""
        # Create dataset with both train and val
        dataset_dir = temp_dir / "dataset"
        for split in ["train", "val"]:
            (dataset_dir / "images" / split).mkdir(parents=True)
            (dataset_dir / "labels" / split).mkdir(parents=True)

            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(dataset_dir / "images" / split / "img.jpg"), img)

            label_path = dataset_dir / "labels" / split / "img.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(
            "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['class_0']\n"
        )

        counts = count_labels(
            dataset_path=dataset_dir,
            split="both",
            show_details=False,
        )

        assert counts[0] == 2  # One from train, one from val

    def test_count_labels_invalid_split(self, sample_dataset):
        """Test error handling for invalid split parameter"""
        with pytest.raises(ValueError, match="split must be one of"):
            count_labels(
                dataset_path=sample_dataset,
                split="invalid",
            )

    def test_count_labels_nonexistent_dataset(self):
        """Test error handling for nonexistent dataset"""
        with pytest.raises(FileNotFoundError):
            count_labels(dataset_path="nonexistent")

    def test_count_labels_no_data_yaml(self, temp_dir):
        """Test error handling when data.yaml is missing"""
        empty_dir = temp_dir / "no_yaml"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="data.yaml not found"):
            count_labels(dataset_path=empty_dir)


class TestAnalyzeDataset:
    """Test dataset analysis functionality"""

    def test_analyze_dataset_basic(self, sample_dataset):
        """Test basic dataset analysis"""
        result = analyze_dataset(
            dataset_path=sample_dataset,
            split="train",
            show_details=False,
        )

        assert "class_counts" in result
        assert "total_instances" in result
        assert "num_classes" in result
        assert "format" in result
        assert "total_images" in result
        assert "image_counts" in result
        assert "resolution_distribution" in result
        assert "class_size_stats" in result

        assert result["total_instances"] >= 0
        assert result["num_classes"] >= 0
        assert result["total_images"] >= 0

    def test_analyze_dataset_with_obb(self, temp_dir):
        """Test analysis with OBB format dataset"""
        # Create OBB dataset
        dataset_dir = temp_dir / "obb_dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        # Create image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / "img.jpg"), img)

        # Create OBB label
        label_path = dataset_dir / "labels" / "train" / "img.txt"
        label_path.write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n")

        # Create data.yaml
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: ['card']\n")

        result = analyze_dataset(
            dataset_path=dataset_dir,
            split="train",
            show_details=False,
        )

        assert result["format"] == "obb"
        assert result["total_images"] == 1
        assert 0 in result["class_counts"]

    def test_analyze_dataset_with_bbox(self, temp_dir):
        """Test analysis with BBox format dataset"""
        # Create BBox dataset
        dataset_dir = temp_dir / "bbox_dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        # Create image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / "img.jpg"), img)

        # Create BBox label
        label_path = dataset_dir / "labels" / "train" / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        # Create data.yaml
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: ['object']\n")

        result = analyze_dataset(
            dataset_path=dataset_dir,
            split="train",
            show_details=False,
        )

        assert result["format"] == "bbox"
        assert result["total_images"] == 1

    def test_analyze_dataset_both_splits(self, temp_dir):
        """Test analyzing both train and val splits"""
        # Create dataset with both splits
        dataset_dir = temp_dir / "dataset"
        for split in ["train", "val"]:
            (dataset_dir / "images" / split).mkdir(parents=True)
            (dataset_dir / "labels" / split).mkdir(parents=True)

            for i in range(2):
                img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.imwrite(str(dataset_dir / "images" / split / f"img{i}.jpg"), img)

                label_path = dataset_dir / "labels" / split / f"img{i}.txt"
                label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(
            "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['class_0']\n"
        )

        result = analyze_dataset(
            dataset_path=dataset_dir,
            split="both",
            show_details=False,
        )

        assert result["total_images"] == 4  # 2 train + 2 val
        assert "train" in result["image_counts"]
        assert "val" in result["image_counts"]
        assert result["image_counts"]["train"] == 2
        assert result["image_counts"]["val"] == 2

    def test_analyze_dataset_class_size_stats(self, temp_dir):
        """Test that class size statistics are calculated"""
        # Create dataset
        dataset_dir = temp_dir / "dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        # Create image with known size
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / "img.jpg"), img)

        # Create label with known bbox
        label_path = dataset_dir / "labels" / "train" / "img.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n")  # 20% of image width/height

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: ['object']\n")

        result = analyze_dataset(
            dataset_path=dataset_dir,
            split="train",
            show_details=False,
        )

        # Check class size stats
        assert 0 in result["class_size_stats"]
        stats = result["class_size_stats"][0]
        assert "pixel_area" in stats
        assert "area_ratio" in stats
        assert "count" in stats
        assert stats["count"] == 1

        # Check that pixel_area stats have expected fields
        assert "min" in stats["pixel_area"]
        assert "max" in stats["pixel_area"]
        assert "mean" in stats["pixel_area"]
        assert "median" in stats["pixel_area"]
        assert "std" in stats["pixel_area"]

    def test_analyze_dataset_resolution_distribution(self, temp_dir):
        """Test resolution distribution calculation"""
        # Create dataset with different resolutions
        dataset_dir = temp_dir / "dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        resolutions = [(640, 480), (1280, 720), (640, 480)]  # Two with same resolution

        for i, (w, h) in enumerate(resolutions):
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            cv2.imwrite(str(dataset_dir / "images" / "train" / f"img{i}.jpg"), img)

            label_path = dataset_dir / "labels" / "train" / f"img{i}.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: ['object']\n")

        result = analyze_dataset(
            dataset_path=dataset_dir,
            split="train",
            show_details=False,
        )

        # Check resolution distribution
        assert len(result["resolution_distribution"]) == 2  # Two unique resolutions
        assert "640x480" in result["resolution_distribution"]
        assert "1280x720" in result["resolution_distribution"]
        assert result["resolution_distribution"]["640x480"] == 2  # Two images with this resolution

    def test_analyze_dataset_empty(self, temp_dir):
        """Test analyzing empty dataset"""
        dataset_dir = temp_dir / "empty_dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text("path: .\ntrain: images/train\nnc: 1\nnames: ['class_0']\n")

        result = analyze_dataset(
            dataset_path=dataset_dir,
            split="train",
            show_details=False,
        )

        assert result["total_images"] == 0
        assert result["total_instances"] == 0
        assert len(result["class_counts"]) == 0

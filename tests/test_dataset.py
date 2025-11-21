"""
Tests for dataset module functionality.

Core tests for:
- Dataset splitting
- Dataset merging
- Dataset synthesis
- Dataset analysis
- Label counting
"""

import cv2
import numpy as np
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
            input_path=sample_dataset / "data.yaml",
            output_dir=output_dir,
            train_ratio=0.8,
        )

        assert result["train_count"] >= 0
        assert result["val_count"] >= 0
        assert result["train_count"] + result["val_count"] > 0
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "images" / "val").exists()
        assert (output_dir / "data.yaml").exists()

    def test_split_dataset_different_ratios(self, sample_dataset, temp_dir):
        """Test splitting with different train ratios"""
        output_dir = temp_dir / "split_70"

        result = split_dataset(
            input_path=sample_dataset / "data.yaml",
            output_dir=output_dir,
            train_ratio=0.7,
        )

        assert result["train_count"] + result["val_count"] > 0


class TestDatasetMerge:
    """Test dataset merging functionality"""

    def test_merge_two_datasets(self, temp_dir):
        """Test merging two datasets"""
        # Create two simple datasets
        datasets = []
        for i in range(2):
            dataset_dir = temp_dir / f"dataset{i}"
            (dataset_dir / "images" / "train").mkdir(parents=True)
            (dataset_dir / "labels" / "train").mkdir(parents=True)

            # Create sample image and label
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img_path = dataset_dir / "images" / "train" / f"img{i}.jpg"
            cv2.imwrite(str(img_path), img)

            label_path = dataset_dir / "labels" / "train" / f"img{i}.txt"
            label_path.write_text(f"{i} 0.5 0.5 0.2 0.2\n")

            # Create data.yaml
            data_yaml = dataset_dir / "data.yaml"
            data_yaml.write_text(yaml.dump({"names": [f"class{i}"], "nc": 1}))
            datasets.append(str(dataset_dir))

        output_dir = temp_dir / "merged"

        result = merge_datasets(
            dataset_dirs=datasets,
            output_dir=output_dir,
        )

        assert result["train_images"] == 2
        assert (output_dir / "data.yaml").exists()

    def test_merge_with_duplicates(self, temp_dir):
        """Test merging with duplicate handling"""
        # Create dataset with duplicate images
        datasets = []
        for i in range(2):
            dataset_dir = temp_dir / f"dataset_dup{i}"
            (dataset_dir / "images" / "train").mkdir(parents=True)
            (dataset_dir / "labels" / "train").mkdir(parents=True)

            # Use same filename in both datasets
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = dataset_dir / "images" / "train" / "duplicate.jpg"
            cv2.imwrite(str(img_path), img)

            label_path = dataset_dir / "labels" / "train" / "duplicate.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n")

            data_yaml = dataset_dir / "data.yaml"
            data_yaml.write_text(yaml.dump({"names": ["class0"], "nc": 1}))
            datasets.append(str(dataset_dir))

        output_dir = temp_dir / "merged_dup"

        result = merge_datasets(
            dataset_dirs=datasets,
            output_dir=output_dir,
        )

        # Should handle duplicates by renaming
        assert result["train_images"] == 2


class TestDatasetSynthesizer:
    """Test dataset synthesis functionality"""

    def test_synthesizer_init_obb_format(self, temp_dir):
        """Test synthesizer initialization with OBB format"""
        target_dir = temp_dir / "targets"
        target_dir.mkdir()
        background_dir = temp_dir / "backgrounds"
        background_dir.mkdir()
        output_dir = temp_dir / "synthetic"

        # Create a target image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(target_dir / "target.jpg"), img)

        # Create background
        bg = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        cv2.imwrite(str(background_dir / "bg.jpg"), bg)

        synthesizer = DatasetSynthesizer(
            target_dir=str(target_dir),
            background_dir=str(background_dir),
            output_dir=str(output_dir),
            annotation_format="obb",
        )

        assert synthesizer.annotation_format == "obb"
        assert len(synthesizer.target_data) > 0
        assert len(synthesizer.background_images) > 0

    def test_synthesize_dataset_trainval(self, temp_dir):
        """Test synthesizing dataset with train and val splits"""
        target_dir = temp_dir / "targets"
        target_dir.mkdir()
        background_dir = temp_dir / "backgrounds"
        background_dir.mkdir()
        output_dir = temp_dir / "synthetic"

        # Create target and background
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(target_dir / "target.jpg"), img)
        bg = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        cv2.imwrite(str(background_dir / "bg.jpg"), bg)

        synthesizer = DatasetSynthesizer(
            target_dir=str(target_dir),
            background_dir=str(background_dir),
            output_dir=str(output_dir),
            annotation_format="hbb",
        )

        synthesizer.synthesize_dataset(num_images=5)

        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "images" / "val").exists()
        assert (output_dir / "data.yaml").exists()


class TestCountLabels:
    """Test label counting functionality"""

    def test_count_labels_train(self, sample_dataset):
        """Test counting labels in train split"""
        counts = count_labels(
            dataset_path=str(sample_dataset),
            split="train",
        )

        assert isinstance(counts, dict)
        assert len(counts) > 0

    def test_count_labels_both(self, temp_dir):
        """Test counting labels in both splits"""
        # Create dataset with train and val
        dataset_dir = temp_dir / "count_dataset"
        for split in ["train", "val"]:
            (dataset_dir / "images" / split).mkdir(parents=True)
            (dataset_dir / "labels" / split).mkdir(parents=True)

            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(dataset_dir / "images" / split / "img.jpg"), img)

            label_path = dataset_dir / "labels" / split / "img.txt"
            label_path.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(yaml.dump({"names": ["class0", "class1"], "nc": 2}))

        counts = count_labels(
            dataset_path=str(dataset_dir),
            split="both",
        )

        # count_labels returns dict[int, int] mapping class_id to count
        assert isinstance(counts, dict)
        assert 0 in counts  # class 0 should exist
        assert 1 in counts  # class 1 should exist


class TestAnalyzeDataset:
    """Test dataset analysis functionality"""

    def test_analyze_dataset_basic(self, sample_dataset):
        """Test basic dataset analysis"""
        stats = analyze_dataset(
            dataset_path=str(sample_dataset),
            split="train",
        )

        assert "total_images" in stats
        assert "total_instances" in stats
        assert "class_counts" in stats

    def test_analyze_dataset_with_obb(self, temp_dir):
        """Test analysis with OBB format"""
        dataset_dir = temp_dir / "obb_dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        # Create OBB label
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(dataset_dir / "images" / "train" / "img.jpg"), img)

        label_path = dataset_dir / "labels" / "train" / "img.txt"
        label_path.write_text("0 0.5 0.5 0.3 0.3 0.4 0.4 0.6 0.6\n")

        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(yaml.dump({"names": ["class0"], "nc": 1}))

        stats = analyze_dataset(
            dataset_path=str(dataset_dir),
            split="train",
        )

        assert stats["total_images"] > 0
        assert stats["format"] in ["obb", "mixed"]

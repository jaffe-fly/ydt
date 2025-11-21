"""
Test core module functionality
"""

import pytest

from ydt.core.formats import detect_format, parse_label_line
from ydt.core.utils import count_class_distribution, find_label_files


class TestFormatDetection:
    """Test format detection functionality"""

    def test_detect_obb_format(self, sample_obb_labels):
        """Test OBB format detection"""
        format_type = detect_format(sample_obb_labels)
        assert format_type == "obb"

    def test_detect_bbox_format(self, sample_bbox_labels):
        """Test BBox format detection"""
        format_type = detect_format(sample_bbox_labels)
        assert format_type == "bbox"

    def test_detect_format_nonexistent_file(self):
        """Test format detection with non-existent file"""
        with pytest.raises(FileNotFoundError):
            detect_format("nonexistent.txt")

    def test_detect_format_empty_file(self, temp_dir):
        """Test format detection with empty file"""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        # Should default to bbox for empty files
        format_type = detect_format(empty_file)
        assert format_type == "bbox"


class TestOBBFormat:
    """Test OBB format handling"""

    def test_parse_obb_line(self):
        """Test parsing OBB format line"""
        obb = parse_label_line("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3", "obb")
        assert obb.class_id == 0
        assert len(obb.points) == 4
        assert tuple(obb.points[0]) == (0.1, 0.1)


class TestBBoxFormat:
    """Test BBox format handling"""

    def test_parse_bbox_line(self):
        """Test parsing BBox format line"""
        bbox = parse_label_line("0 0.5 0.5 0.2 0.1", "bbox")
        assert bbox.class_id == 0
        assert bbox.x_center == 0.5
        assert bbox.y_center == 0.5
        assert bbox.width == 0.2
        assert bbox.height == 0.1


class TestUtils:
    """Test utility functions"""

    def test_count_class_distribution(self, sample_dataset):
        """Test class distribution counting"""
        labels_dir = sample_dataset / "labels" / "train"
        distribution = count_class_distribution(labels_dir)
        assert 0 in distribution
        assert 1 in distribution
        assert distribution[0] == 1  # One instance of class 0

    def test_find_label_files(self, sample_dataset):
        """Test finding label files"""
        labels_dir = sample_dataset / "labels" / "train"
        label_files = find_label_files(labels_dir)
        assert len(label_files) == 1
        assert label_files[0].name == "img001.txt"

    def test_find_label_files_empty_dir(self, temp_dir):
        """Test finding label files in empty directory"""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        label_files = find_label_files(empty_dir)
        assert len(label_files) == 0

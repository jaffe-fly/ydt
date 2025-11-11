"""
pytest fixtures and configuration for YDT tests
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image"""
    image_path = temp_dir / "test_image.jpg"
    # Create a 640x480 RGB test image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)
    return image_path


@pytest.fixture
def sample_video(temp_dir):
    """Create a sample test video"""
    video_path = temp_dir / "test_video.mp4"
    # Create a simple test video with 10 frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))

    for i in range(10):
        # Create a different image for each frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add frame number text
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    return video_path


@pytest.fixture
def sample_obb_labels(temp_dir, sample_image):
    """Create sample OBB format labels"""
    label_path = temp_dir / "test_image.txt"
    # OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    labels = [
        "0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3",  # Square
        "1 0.5 0.5 0.7 0.5 0.7 0.7 0.5 0.7",  # Square
    ]
    label_path.write_text("\n".join(labels))
    return label_path


@pytest.fixture
def sample_bbox_labels(temp_dir, sample_image):
    """Create sample BBox format labels"""
    label_path = temp_dir / "test_image.txt"
    # BBox format: class_id x_center y_center width height
    labels = [
        "0 0.2 0.2 0.2 0.2",  # Square at (0.2, 0.2) with size 0.2
        "1 0.6 0.6 0.2 0.2",  # Square at (0.6, 0.6) with size 0.2
    ]
    label_path.write_text("\n".join(labels))
    return label_path


@pytest.fixture
def sample_dataset(temp_dir, sample_image, sample_obb_labels):
    """Create a minimal YOLO dataset structure"""
    dataset_dir = temp_dir / "dataset"

    # Create directory structure
    (dataset_dir / "images" / "train").mkdir(parents=True)
    (dataset_dir / "images" / "val").mkdir(parents=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True)

    # Copy sample files
    shutil.copy(sample_image, dataset_dir / "images" / "train" / "img001.jpg")
    shutil.copy(sample_obb_labels, dataset_dir / "labels" / "train" / "img001.txt")

    # Create data.yaml
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        """
path: .
train: images/train
val: images/val

nc: 2
names: ['class_0', 'class_1']
"""
    )

    return dataset_dir


@pytest.fixture
def synthetic_objects_dir(temp_dir):
    """Create directory with synthetic object images"""
    objects_dir = temp_dir / "objects"
    objects_dir.mkdir()

    # Create some small object images with transparent backgrounds
    for i in range(3):
        obj_path = objects_dir / f"object_{i}.png"
        # Create a simple colored shape
        image = np.zeros((100, 100, 4), dtype=np.uint8)
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]
        # Draw a circle
        cv2.circle(image, (50, 50), 30, (*color, 255), -1)
        cv2.imwrite(str(obj_path), image)

    return objects_dir


@pytest.fixture
def synthetic_backgrounds_dir(temp_dir):
    """Create directory with background images"""
    backgrounds_dir = temp_dir / "backgrounds"
    backgrounds_dir.mkdir()

    # Create some background images
    for i in range(2):
        bg_path = backgrounds_dir / f"bg_{i}.jpg"
        image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(bg_path), image)

    return backgrounds_dir

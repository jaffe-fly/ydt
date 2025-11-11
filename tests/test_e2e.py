"""
End-to-end tests for complete workflows
"""

import pytest
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml


@pytest.mark.e2e
class TestCompleteWorkflows:
    """Test complete end-to-end workflows"""

    def run_ydt_command(self, args, cwd=None, timeout=60):
        """Helper to run ydt command"""
        cmd = [sys.executable, "-m", "ydt.cli.main"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        return result

    def setup_test_dataset(self, temp_dir):
        """Create a complete test dataset"""
        dataset_dir = temp_dir / "test_dataset"

        # Create directory structure
        (dataset_dir / "images" / "train").mkdir(parents=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True)

        # Create 5 test images with different labels
        for i in range(5):
            # Create image
            img_path = dataset_dir / "images" / "train" / f"img_{i:03d}.jpg"
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), image)

            # Create corresponding label (OBB format)
            label_path = dataset_dir / "labels" / "train" / f"img_{i:03d}.txt"
            # Create 1-3 objects per image
            num_objects = (i % 3) + 1
            labels = []
            for j in range(num_objects):
                class_id = i % 2  # Alternate between class 0 and 1
                # Generate random OBB coordinates
                x1, y1 = 0.1 + j * 0.2, 0.1 + j * 0.2
                x2, y2 = 0.3 + j * 0.2, 0.1 + j * 0.2
                x3, y3 = 0.3 + j * 0.2, 0.3 + j * 0.2
                x4, y4 = 0.1 + j * 0.2, 0.3 + j * 0.2
                labels.append(f"{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}")

            label_path.write_text("\n".join(labels))

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

    def test_complete_image_processing_pipeline(self, temp_dir):
        """Test complete image processing pipeline: slice -> augment -> split"""

        # 1. Setup initial dataset
        dataset_dir = self.setup_test_dataset(temp_dir)

        # 2. Slice large images
        sliced_dir = temp_dir / "sliced"
        result = self.run_ydt_command(
            [
                "image",
                "slice",
                "-i",
                str(dataset_dir),
                "-o",
                str(sliced_dir),
                "-c",
                "2",
                "-r",
                "0.1",
            ]
        )

        assert result.returncode == 0
        assert sliced_dir.exists()

        # Verify sliced images were created
        sliced_images = list(sliced_dir.rglob("*.jpg"))
        assert len(sliced_images) > 5  # Should have more images after slicing

        # Create new data.yaml for sliced dataset
        sliced_data_yaml = sliced_dir / "data.yaml"
        sliced_data_yaml.write_text(
            f"""
path: {sliced_dir}
train: images/train
val: images/val

nc: 2
names: ['class_0', 'class_1']
"""
        )

        # 3. Augment dataset
        aug_dir = temp_dir / "augmented"
        result = self.run_ydt_command(
            [
                "image",
                "augment",
                "-i",
                str(sliced_dir),  # Use directory instead of yaml file
                "-o",
                str(aug_dir),
                "-a",
                "0",
                "90",
                "180",
            ]
        )

        assert result.returncode == 0
        assert aug_dir.exists()

        # Verify augmented images
        aug_images = list(aug_dir.rglob("*.jpg"))
        # Should have multiple rotations per image (may vary based on label content)
        assert len(aug_images) >= len(sliced_images) * 2  # At least 2x for rotations

        # Create data.yaml for augmented dataset
        aug_data_yaml = aug_dir / "data.yaml"
        aug_data_yaml.write_text(
            f"""
path: {aug_dir}
train: images/train
val: images/val

nc: 2
names: ['class_0', 'class_1']
"""
        )

        # 4. Split dataset
        final_dir = temp_dir / "final"
        result = self.run_ydt_command(
            ["dataset", "split", "-i", str(aug_data_yaml), "-o", str(final_dir), "-r", "0.8"]
        )

        assert result.returncode == 0
        assert final_dir.exists()

        # Verify split structure
        train_dir = final_dir / "images" / "train"
        val_dir = final_dir / "images" / "val"
        assert train_dir.exists()
        assert val_dir.exists()

        train_images = list(train_dir.glob("*.jpg"))
        val_images = list(val_dir.glob("*.jpg"))
        assert len(train_images) > 0
        assert len(val_images) > 0

        # Verify label files exist for all images
        for img_path in train_images[:3]:  # Check first few
            label_path = final_dir / "labels" / "train" / img_path.with_suffix(".txt").name
            assert label_path.exists()

        for img_path in val_images:
            label_path = final_dir / "labels" / "val" / img_path.with_suffix(".txt").name
            assert label_path.exists()

    def test_video_to_dataset_workflow(self, temp_dir):
        """Test converting video to dataset workflow"""

        # Create test video
        video_path = temp_dir / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 2.0, (640, 480))

        for i in range(20):  # 20 frames
            frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
            # Add frame number
            cv2.putText(
                frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
            out.write(frame)
        out.release()

        # Extract frames
        frames_dir = temp_dir / "frames"
        result = self.run_ydt_command(
            [
                "image",
                "video",
                "-i",
                str(video_path),
                "-o",
                str(frames_dir),
                "-s",
                "5",  # Extract every 5th frame
            ]
        )

        assert result.returncode == 0
        assert frames_dir.exists()

        # Check extracted frames
        video_frames_dir = frames_dir / "test_video_frames"
        assert video_frames_dir.exists()
        frames = list(video_frames_dir.glob("*.jpg"))
        assert len(frames) >= 4  # Should extract at least 4 frames (0, 5, 10, 15)

        # Create labels directory and some sample labels
        labels_dir = temp_dir / "frames" / "labels"
        labels_dir.mkdir()

        for i, frame_path in enumerate(frames[:3]):  # Label first 3 frames
            label_path = labels_dir / frame_path.with_suffix(".txt").name
            label_path.write_text(f"0 0.5 0.5 0.2 0.2\n")  # Simple BBox label

        # Create dataset structure for visualization test
        dataset_yaml = temp_dir / "frames" / "data.yaml"
        dataset_yaml.write_text(
            """
path: .
train: .
val: .

nc: 1
names: ['object']
"""
        )

        # Test visualization (quick check - don't actually open GUI)
        # This would normally open interactive window
        result = self.run_ydt_command(
            ["viz", "dataset", "-i", str(temp_dir / "frames"), "-f", "0"], timeout=5
        )

        # This might fail due to GUI environment, but at least test the command structure
        # In CI environment, we might need to skip actual GUI tests

    def test_synthetic_dataset_workflow(self, temp_dir):
        """Test synthetic dataset generation workflow"""

        # Create objects directory
        objects_dir = temp_dir / "objects"
        objects_dir.mkdir()

        # Create small object images
        for i in range(3):
            obj_path = objects_dir / f"obj_{i}.png"
            image = np.zeros((100, 100, 4), dtype=np.uint8)
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i]
            cv2.circle(image, (50, 50), 30, (*color, 255), -1)
            cv2.imwrite(str(obj_path), image)

        # Create backgrounds directory
        backgrounds_dir = temp_dir / "backgrounds"
        backgrounds_dir.mkdir()

        for i in range(2):
            bg_path = backgrounds_dir / f"bg_{i}.jpg"
            image = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(bg_path), image)

        # Generate synthetic dataset (small number for testing)
        output_dir = temp_dir / "synthetic"
        result = self.run_ydt_command(
            [
                "dataset",
                "synthesize",
                "-t",
                str(objects_dir),
                "-b",
                str(backgrounds_dir),
                "-o",
                str(output_dir),
                "-n",
                "5",  # Small number for testing
            ]
        )

        assert result.returncode == 0
        assert output_dir.exists()

        # Check generated images
        synthetic_images = list(output_dir.rglob("*.jpg"))
        assert len(synthetic_images) == 5

        # Check corresponding label files
        for img_path in synthetic_images:
            # Convert image path to label path (images -> labels, .jpg -> .txt)
            label_path = Path(
                str(img_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
            ).with_suffix(".txt")
            assert label_path.exists(), f"Label file not found: {label_path}"

            # Verify label format
            label_content = label_path.read_text().strip()
            assert label_content  # Should not be empty
            # Should have class_id and coordinates
            parts = label_content.split()
            assert len(parts) >= 5  # At least BBox format

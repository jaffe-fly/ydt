"""Debug script to test analyze command"""
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np

# Create test dataset
temp_dir = Path(tempfile.mkdtemp())
dataset_dir = temp_dir / "dataset"

# Create directory structure
(dataset_dir / "images" / "train").mkdir(parents=True)
(dataset_dir / "labels" / "train").mkdir(parents=True)

# Create a test image
image_path = dataset_dir / "images" / "train" / "img001.jpg"
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
cv2.imwrite(str(image_path), image)

# Create a label file
label_path = dataset_dir / "labels" / "train" / "img001.txt"
labels = ["0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"]
label_path.write_text("\n".join(labels))

# Create data.yaml
data_yaml = dataset_dir / "data.yaml"
data_yaml.write_text("""
path: .
train: images/train
val: images/val

nc: 2
names: ['class_0', 'class_1']
""")

print(f"Created test dataset at: {dataset_dir}")
print(f"Running analyze command...")

# Run the analyze command
cmd = [sys.executable, "-m", "ydt.cli.main", "analyze", "-i", str(dataset_dir), "--split", "train"]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

print(f"\n{'='*70}")
print(f"Return code: {result.returncode}")
print(f"{'='*70}")
print(f"\nSTDOUT type: {type(result.stdout)}")
print(f"STDOUT value: {result.stdout!r}")
print(f"\n{'='*70}")
print(f"STDERR type: {type(result.stderr)}")
print(f"STDERR value: {result.stderr!r}")
print(f"{'='*70}")

# Cleanup
shutil.rmtree(temp_dir)

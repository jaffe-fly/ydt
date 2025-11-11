<div align="center">

# 🎯 YDT - YOLO Dataset Tools

**A Professional Toolkit for YOLO Dataset Processing**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checked](https://img.shields.io/badge/type--checked-mypy-informational.svg)](https://mypy.readthedocs.io/)

[English](README.md) | [简体中文](README_CN.md)

---

</div>

## ✨ Features

<table>
<tr>
<td width="50%">

### 🖼️ **Image Processing**
- 📐 SAHI-powered smart slicing
- 🔄 Rotation with OBB transformation
- 📏 Multi-method resize (scale & crop)
- 📍 Coordinate-based precision cropping
- 🎨 HSV color augmentation
- 🎬 Video frame extraction
- 🚀 Multi-threaded video processing

</td>
<td width="50%">

### 📊 **Dataset Operations**
- ✂️ Smart train/val split
- 🔗 Multi-dataset merging
- 🎲 Synthetic data generation
- 🤖 YOLO auto-labeling
- 📦 Batch processing

</td>
</tr>
<tr>
<td width="50%">

### 👁️ **Visualization**
- 🖼️ Interactive dataset browser
- ⌨️ Keyboard controls (n/p/q)
- 🎯 Class filtering
- 📸 Letterbox preview
- 🎨 Augmentation effects

</td>
<td width="50%">

### 🎯 **Format Support**
- 📐 OBB (Oriented Bounding Box)
- 📦 Standard BBox
- 🔄 Auto format detection
- ✨ Seamless conversion

</td>
</tr>
</table>

## 📦 Installation

Install from PyPI:

```bash
pip install yolodt
```

Or install from source:

```bash
git clone https://github.com/yourusername/ydt.git
cd ydt
pip install -e .
```

## 🚀 Quick Start

```bash
# Try it out!
ydt image slice -i ./images -o ./output
ydt image slice -i ./images -o ./output -c 3 -d 2
ydt viz dataset -i ./dataset

ydt image slice -i image.jpg -o ./output -c 2
ydt image augment -i image.jpg -o ./output -a 45 90
ydt image crop-coords -i image.jpg -o ./output -c "100,50,600,400"
```

## 💻 Usage

### Command Line

```bash
# Image processing
# Slice images (directory or single file)
ydt image slice -i ./imgs -o ./out -c 3
ydt image slice -i image.jpg -o ./out -c 2  # NEW: Single file support
ydt image slice -i ./imgs -o ./out -c 3 -d 2 -r 0.1 --overlap-vertical 0.05

# Resize images
ydt image resize -i ./images -o ./resized -s 640 800 1024
ydt image resize -i image.jpg -o ./resized -s 640  # Single file support

# Augment images (directory or single file)
ydt image augment -i data.yaml -o ./aug
ydt image augment -i image.jpg -o ./aug -a 45 90  # NEW: Single file support

# Extract video frames
ydt image video -i ./videos -o ./frames -s 30
ydt image video -i ./videos -o ./frames --parallel -w 4

# Crop by coordinates (directory or single file)
ydt image crop-coords -i ./images -o ./cropped -c "100,50,600,400"
ydt image crop-coords -i image.jpg -o ./cropped -c "100,50,600,400"  # NEW: Single file support

# Dataset operations
ydt dataset split -i data.yaml -o ./split -r 0.8
ydt dataset merge -i ./ds1 ./ds2 -o ./merged
ydt dataset synthesize -t ./targets -b ./backgrounds -o ./synthetic --objects-per-image 2-5 --split train
ydt dataset synthesize -t ./targets -b ./backgrounds -o ./synthetic --data-yaml ./data.yaml --rotation-range=-20,20  # With class validation and limited rotation
ydt dataset auto-label -i ./images -m ./yolo11n.pt --format bbox -o ./labeled

# Visualization
ydt viz dataset -i ./dataset
ydt viz letterbox -i ./image.jpg
ydt viz augment -i ./image.jpg
```

### Python API

```python
from ydt.image import (
    slice_dataset,
    augment_dataset,
    extract_frames,
    process_images_multi_method,
    concat_images_horizontally,
    concat_images_vertically
)
from ydt.dataset import split_dataset, DatasetSynthesizer, auto_label_dataset
from ydt.visual import visualize_dataset, visualize_letterbox

# Slice large images (directory or single file)
slice_dataset("./dataset", "./sliced", horizontal_count=3)
slice_dataset("image.jpg", "./sliced", horizontal_count=2)  # NEW: Single file support

# Grid slicing (2×3 = 6 slices)
slice_dataset("./dataset", "./sliced", horizontal_count=2, vertical_count=3)

# Resize images with both scale and crop methods
process_images_multi_method("./images", "./resized", target_sizes=[640, 800, 1024])

# Concatenate images
concat_images_horizontally("img1.jpg", "img2.jpg", "output.jpg", alignment="center")

# Extract frames from videos
extract_frames("./videos", "./frames", step=30)

# Split dataset
split_dataset("./data.yaml", "./split", train_ratio=0.8)

# Auto-label images
result = auto_label_dataset(
    input_dir="./images",
    model_path="./yolo11n.pt",
    format_type="bbox",
    output_dir="./labeled"
)

# Generate synthetic dataset with custom parameters
synthesizer = DatasetSynthesizer(
    target_dir="./targets",
    background_dir="./backgrounds",
    output_dir="./synthetic",
    objects_per_image=(2, 5),       # 2-5 objects per image
    split_mode="trainval",           # Generate train+val
    train_ratio=0.8,                 # 80% train, 20% val
    data_yaml_path="./data.yaml",    # Validate class names (e.g., bn_back.jpg requires 'bn' in names)
    rotation_range=(-20, 20)         # Limit rotation to ±20 degrees
)
stats = synthesizer.synthesize_dataset(num_images=1000)

# Visualize dataset
visualize_dataset("./dataset", scan_train=True)
visualize_letterbox("./image.jpg", output_dir="./output")
```

## 📦 What's Inside

```
ydt/
├── 🖼️  image/       # Image processing
├── 📊  dataset/     # Dataset operations
├── 👁️  visual/      # Visualization
├── 🛠️  core/        # Core utilities
├── 🤖  auto_label/  # Auto-labeling
└── ⚡  cli/         # CLI interface
```

## 🎯 Key Features

### Dual Format Support

Automatically detects and handles both formats:

| Format | Values | Description |
|--------|--------|-------------|
| **OBB** | 9 values | `class_id x1 y1 x2 y2 x3 y3 x4 y4` |
| **BBox** | 5 values | `class_id x_center y_center width height` |

### Smart Slicing

Powered by SAHI, intelligently slice large images while preserving label accuracy. Supports both horizontal and grid slicing with configurable overlap ratios.

```bash
# Horizontal slicing (default)
ydt image slice -i ./images -o ./sliced -c 3 -r 0.1

# Grid slicing (3×2 = 6 slices)
ydt image slice -i ./images -o ./sliced -c 3 -d 2 -r 0.1 --overlap-vertical 0.05

# Fine grid slicing with custom overlap
ydt image slice -i ./images -o ./sliced -c 4 -d 3 -r 0.05 --overlap-vertical 0.02
```

### Video Frame Extraction

Extract frames from video files for dataset creation. Supports both sequential and parallel processing.

```bash
# Sequential processing (default)
ydt image video -i ./videos -o ./frames -s 30

# Parallel processing for multiple videos
ydt image video -i ./videos -o ./frames --parallel -w 4
```

**Features:**
- 🎯 Smart worker count auto-detection
- ⚡ Concurrent video decoding
- 📊 Progress tracking per video
- 🔄 Automatic fallback for single videos

### Auto-Labeling

Automatically label images using YOLO models with support for both BBox and OBB formats:

```bash
ydt dataset auto-label -i ./images -m ./yolo11n.pt --format bbox -o ./labeled
```

**Features:**
- 🎯 Support for both BBox and OBB formats
- 🤖 Automatic format detection and conversion
- 📁 Clean output directory structure
- ⚙️ Configurable confidence and IOU thresholds
- 🔍 Preview mode with `--dry-run`

### Interactive Visualization

Browse your dataset with keyboard controls:
- `n` - Next image
- `p` - Previous image
- `q` - Quit

```bash
ydt viz dataset -i ./dataset
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO framework
- [SAHI](https://github.com/obss/sahi) - Slicing aided hyper inference
- [Albumentations](https://github.com/albumentations-team/albumentations) - Image augmentation

---

# 🎯 Usage Guide

Comprehensive guide for using YDT (YOLO Dataset Tools).

## Table of Contents

- [Command Overview](#command-overview)
- [Image Processing](#image-processing)
- [Dataset Operations](#dataset-operations)
- [Quality Control](#quality-control)
- [Visualization](#visualization)
- [Python API](#python-api)
- [Best Practices](#best-practices)

## Command Overview

All commands follow this structure:

```bash
ydt <category> <command> [options]
```

| Category | Commands | Description |
|----------|----------|-------------|
| `image` | slice, resize, augment | Image processing operations |
| `dataset` | split, merge, synthesize | Dataset manipulation |
| `quality` | check, clean, stats | Quality control |
| `viz` | dataset, letterbox, augment | Visualization tools |

## Image Processing

### 1. Image Slicing

Slice large images into smaller tiles using SAHI.

**Command:**
```bash
ydt image slice -i <input> -o <output> [options]
```

**Parameters:**

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | path | required | Input directory containing images |
| `--output` | `-o` | path | required | Output directory for sliced images |
| `--count` | `-c` | int | 3 | Number of slices per dimension |
| `--overlap` | `-r` | float | 0.1 | Overlap ratio between slices |

**Examples:**

```bash
# Basic slicing (3x3 grid, 10% overlap)
ydt image slice -i ./images -o ./sliced

# Custom grid (4x4 with 20% overlap)
ydt image slice -i ./images -o ./sliced -c 4 -r 0.2

# Slice entire dataset
ydt image slice -i ./dataset/images/train -o ./sliced_dataset
```

**Output Structure:**
```
sliced_dataset/
├── original_image_1_slice_0.jpg
├── original_image_1_slice_1.jpg
├── original_image_1_slice_0.txt  # Transformed labels
├── original_image_1_slice_1.txt
└── ...
```

### 2. Image Resizing

Resize or crop images to target size.

**Command:**
```bash
ydt image resize -i <input> -o <output> [options]
```

**Parameters:**

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | path | required | Input directory |
| `--output` | `-o` | path | required | Output directory |
| `--size` | `-s` | int | 640 | Target size (width=height) |
| `--method` | | choice | scale | Resize method: `scale` or `crop` |

**Examples:**

```bash
# Scale to 640x640
ydt image resize -i ./images -o ./resized -s 640 --method scale

# Center crop to 1024x1024
ydt image resize -i ./images -o ./cropped -s 1024 --method crop
```

### 3. Data Augmentation

Rotate images and automatically transform labels.

**Command:**
```bash
ydt image augment -i <data.yaml> -o <output> [options]
```

**Parameters:**

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | path | required | Dataset YAML file |
| `--output` | `-o` | path | required | Output directory |
| `--angles` | `-a` | list | auto | Rotation angles (e.g., `90 180 270`) |

**Examples:**

```bash
# Auto-select rotation angles
ydt image augment -i ./dataset/data.yaml -o ./augmented

# Specific angles
ydt image augment -i ./data.yaml -o ./aug -a 0 90 180 270
```

### 3. Video Frame Extraction

Extract frames from video files for dataset creation.

**Command:**
```bash
ydt image video -i <input> -o <output> [options]
```

**Parameters:**

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | path | required | Video file or directory containing videos |
| `--output` | `-o` | path | required | Output directory for extracted frames |
| `--step` | `-s` | int | 40 | Extract every Nth frame |

**Examples:**

```bash
# Extract frames from single video
ydt image video -i ./video.mp4 -o ./frames

# Extract from directory of videos
ydt image video -i ./videos -o ./all_frames

# Extract every 30th frame (more frames)
ydt image video -i ./video.mp4 -o ./frames -s 30

# Extract every 60th frame (fewer frames)
ydt image video -i ./video.mp4 -o ./frames -s 60
```

**Output Structure:**
```
frames/
├── video1_frames/
│   ├── frame_000000.jpg
│   ├── frame_000040.jpg
│   └── ...
└── video2_frames/
    ├── frame_000000.jpg
    └── ...
```

**Video Information Displayed:**
- Total frame count
- FPS (frames per second)
- Video resolution
- Video duration
- Processing progress

### Real-World Examples

**Example 1: Surveillance Video Key Frame Extraction**
```bash
# Extract key frames from surveillance video (every 2 seconds at 30fps)
ydt image video -i ./surveillance/monitoring_001.mp4 -o ./key_frames -s 60

# Output will show video details and processing progress
```

**Example 2: Training Dataset Creation**
```bash
# Process multiple training videos for dataset creation
ydt image video -i ./training_videos -o ./dataset_frames -s 30

# Each video gets its own subdirectory
```

**Example 3: Detailed Analysis**
```bash
# Extract dense frames for detailed analysis
ydt image video -i ./experiment/high_speed_camera.mp4 -o ./analysis_frames -s 5
```

**Expected Output:**
```
Found 1 video files
Processing video: monitoring_001.mp4
  Total frames: 3000
  FPS: 30.00
  Resolution: 1920x1080
  Duration: 100.00 seconds
  Output directory: ./key_frames/monitoring_001_frames
  Progress: 1800/3000 frames (60.0%), saved: 30 images
  Completed: monitoring_001.mp4
  Processed frames: 3000
  Saved images: 50
All videos processed!
Total images saved: 50
Output directory: ./key_frames
```

**Python API Usage Example:**
```python
from image import extract_frames

# Process a single video for dataset creation
count = extract_frames(
    video_path="./data/raw_videos/interview_001.mp4",
    frames_output_dir="./data/frames/interview_001",
    step=30  # One frame per second for 30fps video
)

print(f"Successfully extracted {count} frames")

# Batch process multiple videos
video_dir = Path("./data/raw_videos")
output_dir = Path("./data/frames")

for video_file in video_dir.glob("*.mp4"):
    extract_frames(
        video_path=video_file,
        frames_output_dir=output_dir / f"{video_file.stem}_frames",
        step=60
    )
```

## Dataset Operations

### 1. Dataset Splitting

Split dataset into train/val sets with class balancing.

**Command:**
```bash
ydt dataset split -i <data.yaml> -o <output> [options]
```

**Parameters:**

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | path | required | Dataset YAML file |
| `--output` | `-o` | path | required | Output directory |
| `--ratio` | `-r` | float | 0.8 | Train ratio (0.0-1.0) |
| `--balance` | | flag | false | Balance rotation angles |

**Examples:**

```bash
# 80/20 split
ydt dataset split -i ./data.yaml -o ./split -r 0.8

# With rotation balancing
ydt dataset split -i ./data.yaml -o ./split -r 0.8 --balance

# 70/30 split
ydt dataset split -i ./data.yaml -o ./split -r 0.7
```

### 2. Dataset Merging

Merge multiple datasets into one.

**Command:**
```bash
ydt dataset merge -i <dir1> <dir2> ... -o <output>
```

**Parameters:**

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | paths | required | Input dataset directories |
| `--output` | `-o` | path | required | Output directory |

**Examples:**

```bash
# Merge two datasets
ydt dataset merge -i ./dataset1 ./dataset2 -o ./merged

# Merge multiple datasets
ydt dataset merge -i ./ds1 ./ds2 ./ds3 ./ds4 -o ./combined
```

### 3. Synthetic Dataset Generation

Generate synthetic datasets by compositing objects on backgrounds.

**Command:**
```bash
ydt dataset synthesize -t <targets> -b <backgrounds> -o <output> [options]
```

**Parameters:**

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--targets` | `-t` | path | required | Target objects directory |
| `--backgrounds` | `-b` | path | required | Background images directory |
| `--output` | `-o` | path | required | Output directory |
| `--num` | `-n` | int | 1000 | Number of images to generate |

**Examples:**

```bash
# Generate 1000 images
ydt dataset synthesize -t ./objects -b ./backgrounds -o ./synthetic

# Generate 5000 images
ydt dataset synthesize -t ./objects -b ./bgs -o ./syn -n 5000
```

## Quality Control

### 1. Quality Check

Check dataset for duplicates and label errors.

**Command:**
```bash
ydt quality check -i <dataset> [options]
```

**Parameters:**

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | path | required | Dataset directory |
| `--type` | `-t` | choice | all | Check type: `duplicates`, `labels`, `all` |

**Examples:**

```bash
# Check everything
ydt quality check -i ./dataset -t all

# Check only duplicates
ydt quality check -i ./dataset -t duplicates

# Check only labels
ydt quality check -i ./dataset -t labels
```

### 2. Dataset Cleaning

Clean dataset by removing unwanted files.

**Command:**
```bash
ydt quality clean -i <dataset> [options]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `--input`, `-i` | path | Dataset directory |
| `--unlabeled` | flag | Remove unlabeled images |
| `--duplicates` | flag | Remove duplicate images |
| `--empty` | flag | Remove empty label files |
| `--dry-run` | flag | Preview without deleting |

**Examples:**

```bash
# Preview cleaning
ydt quality clean -i ./dataset --unlabeled --duplicates --dry-run

# Actually clean
ydt quality clean -i ./dataset --unlabeled --duplicates

# Remove all issues
ydt quality clean -i ./dataset --unlabeled --duplicates --empty
```

### 3. Dataset Statistics

Show dataset statistics.

**Command:**
```bash
ydt quality stats -i <dataset>
```

**Examples:**

```bash
ydt quality stats -i ./dataset
```

**Output:**
```
Dataset Statistics:
- Total images: 1000
- Train images: 800
- Val images: 200
- Classes: 5
- Class distribution:
  - class_0: 450 instances
  - class_1: 320 instances
  ...
```

## Visualization

### 1. Dataset Visualization

Interactively browse dataset with annotations.

**Command:**
```bash
ydt viz dataset -i <input> [options]
```

**Parameters:**

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--input` | `-i` | path | required | Dataset path or single image |
| `--filter` | `-f` | list | all | Filter specific class IDs |
| `--train` | | flag | false | Show training set |
| `--val` | | flag | false | Show validation set |
| `--window-size` | | ints | 1920 1080 | Window size (width height) |

**Keyboard Controls:**
- `n` or `→` - Next image
- `p` or `←` - Previous image
- `q` or `ESC` - Quit
- `s` - Save current visualization

**Examples:**

```bash
# Visualize entire dataset
ydt viz dataset -i ./dataset

# Show only training set
ydt viz dataset -i ./dataset --train

# Filter classes 0 and 1
ydt viz dataset -i ./dataset -f 0 1

# Custom window size
ydt viz dataset -i ./dataset --window-size 2560 1440

# Visualize single image
ydt viz dataset -i ./dataset/images/train/img001.jpg
```

### 2. Letterbox Preview

Preview YOLO letterbox preprocessing effect.

**Command:**
```bash
ydt viz letterbox -i <image> [options]
```

**Parameters:**

| Parameter | Short | Type | Description |
|-----------|-------|------|-------------|
| `--input` | `-i` | path | Image file path |
| `--save` | `-s` | path | Save output directory |

**Keyboard Controls:**
- `s` - Save letterboxed image
- `q` - Quit

**Examples:**

```bash
# Preview letterbox effect
ydt viz letterbox -i ./image.jpg

# Save output
ydt viz letterbox -i ./image.jpg -s ./output
```

### 3. Augmentation Preview

Preview augmentation effects (HSV, etc.).

**Command:**
```bash
ydt viz augment -i <image>
```

**Examples:**

```bash
ydt viz augment -i ./image.jpg
```

## Python API

### Basic Usage

```python
from image import slice_dataset, augment_dataset, resize_images
from dataset import split_dataset, merge_datasets, DatasetSynthesizer
from quality import find_duplicate_images, check_labels
from visual import visualize_dataset
```

### Image Processing

```python
# Slice images
slice_dataset(
    dataset_dir="./dataset",
    output_dir="./sliced",
    slice_count=3,
    overlap_ratio=0.1
)

# Augment dataset
augment_dataset(
    data_yaml="./data.yaml",
    output_dir="./augmented",
    angles=[0, 90, 180, 270]
)

# Resize images
resize_images(
    input_dir="./images",
    output_dir="./resized",
    target_size=640,
    method="scale"
)
```

### Dataset Operations

```python
# Split dataset
split_dataset(
    data_yaml_path="./data.yaml",
    output_dir="./split",
    train_ratio=0.8,
    balance_rotation=True
)

# Merge datasets
merge_datasets(
    dataset_dirs=["./ds1", "./ds2"],
    output_dir="./merged"
)

# Synthesize dataset
synthesizer = DatasetSynthesizer(
    target_dir="./targets",
    background_dir="./backgrounds",
    output_dir="./synthetic"
)
synthesizer.synthesize_dataset(num_images=1000)
```

### Quality Control

```python
# Find duplicates
duplicates = find_duplicate_images("./dataset")
print(f"Found {len(duplicates)} duplicate groups")

# Check labels
errors = check_labels("./dataset")
for error in errors:
    print(f"Error in {error['file']}: {error['message']}")

# Remove unlabeled images
from quality import remove_unlabeled_images
removed_count = remove_unlabeled_images("./dataset", dry_run=False)
print(f"Removed {removed_count} unlabeled images")
```

### Visualization

```python
# Visualize dataset
visualize_dataset(
    dataset_path="./dataset",
    filter_labels=[0, 1, 2],
    scan_train=True,
    scan_val=True
)

# Visualize letterbox
from visual import visualize_letterbox
visualize_letterbox(
    image_path="./image.jpg",
    output_dir="./output",
    letterbox_size=(640, 640)
)
```

## Best Practices

### 1. Dataset Preparation Workflow

```bash
# Step 1: Check quality
ydt quality check -i ./raw_dataset -t all

# Step 2: Clean dataset
ydt quality clean -i ./raw_dataset --unlabeled --duplicates

# Step 3: Split dataset
ydt dataset split -i ./cleaned/data.yaml -o ./final -r 0.8

# Step 4: Visualize to verify
ydt viz dataset -i ./final
```

### 2. Large Image Processing

```bash
# For images larger than 2000x2000
ydt image slice -i ./large_images -o ./sliced -c 4 -r 0.2
```

### 3. Augmentation Strategy

```bash
# Use appropriate angles based on your domain
# For objects with rotational symmetry:
ydt image augment -i ./data.yaml -o ./aug -a 0 90 180 270

# For less symmetric objects:
ydt image augment -i ./data.yaml -o ./aug -a 0 45 90 135 180 225 270 315
```

### 4. Quality Control Checklist

1. Check for duplicates
2. Validate label formats
3. Remove unlabeled images
4. Verify class distribution
5. Visual inspection

## Troubleshooting

### Common Issues

**Issue**: Slice command takes too long

**Solution**: Reduce slice count or process in batches
```bash
ydt image slice -i ./batch1 -o ./out1 -c 3
ydt image slice -i ./batch2 -o ./out2 -c 3
```

**Issue**: Labels not displaying in visualization

**Solution**: Check label file format and paths
```python
from quality import check_labels
errors = check_labels("./dataset")
print(errors)
```

**Issue**: Out of memory during processing

**Solution**: Process in smaller batches or reduce slice count

---

[⬆ Back to Top](#-usage-guide) | [➡️ Next: API Reference](api-reference.md)

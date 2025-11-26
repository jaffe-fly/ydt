<div align="center">

### YDT - YOLO Dataset Tools

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)[![Type Checked](https://img.shields.io/badge/type--checked-mypy-informational.svg)](https://mypy.readthedocs.io/)

[English](README.md) | [简体中文](README_CN.md)



</div>

#### Features

- Auto-detects and handles both OBB (9 values: `class_id x1 y1 x2 y2 x3 y3 x4 y4`) and BBox (5 values: `class_id x_center y_center width height`) formats
- SAHI-powered smart slicing for large images with horizontal/grid modes and configurable overlap
- Resize (scale & crop) with custom interpolation (linear/lanczos4),image or yolo dataset
- Coordinate-based precision cropping
- Object cropping from model inference or dataset labels with padding and size filters
- Video frame extraction with parallel processing support
- Smart train/val split with class balancing
- Multi-dataset merging
- Dataset extraction by class IDs with optional label filtering and ID remapping
- Synthetic dataset generation with configurable objects per image, rotation ranges, and balanced class sampling
- YOLO auto-labeling with BBox/OBB format support
- Interactive dataset browser with keyboard controls (n/p/q)

#### Installation


```bash
pip install yolodt
```

#### Usage

```bash
ydt --help

usage: ydt [-h] [--version] [-v]
           {slice,augment,video,crop-coords,resize,concat,split,merge,extract,synthesize,auto-label,analyze,visualize,viz-letterbox}
           ...

YOLO Dataset Tools - Process and manage YOLO format datasets

positional arguments:
  {slice,augment,video,crop-coords,crop,resize,concat,split,merge,extract,synthesize,auto-label,analyze,visualize,viz-letterbox}
                        Available commands
    slice               Slice large images into tiles
    augment             Augment dataset with rotations
    video               Extract frames from videos
    crop-coords         Crop images by coordinates
    crop                Crop objects from images using model or dataset labels
    resize              Resize images or YOLO dataset
    concat              Concatenate two images
    split               Split dataset into train/val
    merge               Merge multiple datasets
    extract             Extract classes, images, or labels
    synthesize          Generate synthetic dataset
    auto-label          Auto-label images using YOLO model
    analyze             Analyze dataset statistics
    visualize           Visualize YOLO dataset interactively
    viz-letterbox       Visualize letterbox transformation

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -v, --verbose         Verbose output
```

#### 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO framework
- [SAHI](https://github.com/obss/sahi) - Slicing aided hyper inference
- [Albumentations](https://github.com/albumentations-team/albumentations) - Image augmentation

---

# Changelog

All notable changes to YDT (YOLO Dataset Tools) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [0.2.0] - 2025-11-07

### Added
- **Image Processing Module**
  - SAHI-powered smart image slicing with automatic label transformation
  - Grid slicing support: specify both horizontal (-c) and vertical (-d) slice counts
  - Horizontal slicing: traditional single-direction slicing with overlap control
  - Multi-method image resizing: generates both scaled and cropped versions (`ydt image resize`)
  - Rotation-based data augmentation with OBB coordinate conversion
  - Coordinate-based precision cropping (`ydt image crop-coords`)
  - Image concatenation tools (`ydt image concat`) with horizontal/vertical direction support
  - Video frame extraction with parallel processing support (`ydt image video --parallel`)

- **Dataset Operations Module**
  - Enhanced synthetic dataset generation with flexible object placement
  - Objects per image control: single number (2) or range (5-10)
  - Dataset split options: train-only or train+val with configurable ratios
  - Smart dataset splitting with class balancing (`ydt dataset split`)
  - Multi-dataset merging with conflict resolution (`ydt dataset merge`)
  - Synthetic dataset generation with alpha blending (`ydt dataset synthesize`)
  - Auto-labeling with YOLO models (`ydt dataset auto-label`)

- **Visualization Module**
  - Interactive dataset browser with keyboard controls (`ydt viz dataset`)
  - Letterbox effect preview (`ydt viz letterbox`)
  - HSV augmentation visualization (`ydt viz augment`)
  - Albumentations effect comparison
  - Multi-example augmentation grid

- **Core Module**
  - Automatic format detection (OBB vs BBox)
  - Format conversion utilities
  - Unified logging system
  - Common utility functions

- **CLI Interface**
  - Complete command-line interface with 13 commands
  - Three main categories: image, dataset, viz
  - Detailed help for all commands
  - Progress bars and status indicators

### Removed
- **Quality Control Module**: Removed entire quality control module and related CLI commands
  - Removed `ydt quality` command group
  - Removed duplicate detection functionality
  - Removed label validation tools
  - Removed cleanup utilities
  - Users should use external tools or implement custom quality control as needed

### Changed
- Updated CLI command structure to focus on core functionality
- Simplified module imports in main package
- Improved documentation to reflect current feature set

- **Documentation**
  - Comprehensive README (English and Chinese)
  - Installation guide
  - Usage guide with examples
  - Complete API reference
  - Publishing guide
  - Contributing guidelines

- **Project Infrastructure**
  - Modern pyproject.toml configuration
  - Full type annotations throughout
  - MIT License
  - uv package manager support
  - Windows batch script for quick launch

### Format Support
- OBB format (9 values): class_id + 4 corner points
- BBox format (5 values): class_id + center + width/height
- Automatic format detection

### Dependencies
- Python >= 3.8
- OpenCV >= 4.5.0
- Ultralytics >= 8.0.0
- SAHI >= 0.11.0
- Albumentations >= 1.3.0
- And more (see pyproject.toml)

### Development Tools
- Black for code formatting
- Ruff for linting
- mypy for type checking
- pytest for testing
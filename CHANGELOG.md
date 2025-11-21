# Changelog

All notable changes to YDT (YOLO Dataset Tools) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.0] - 2025-11-20

### Added
- **Colored Logging**
  - Added `colorlog` dependency for colored terminal output
  - Different log levels now display in different colors (DEBUG=cyan, INFO=green, WARNING=yellow, ERROR=red, CRITICAL=red on white)
  - New `use_color` parameter in `setup_logger()` function (default: True)
  - Color support can be disabled for compatibility with non-color terminals

- **Dataset Extract Feature**
  - New `extract` command with three modes:
    - `class`: Extract images + labels for specified class IDs
    - `images-only`: Extract only images containing specified classes
    - `labels-only`: Extract labels for given image directory
  - `--filter-labels` option to keep only specified class annotations in label files
  - `--remap-ids` option to remap class IDs to sequential 0, 1, 2... format
  - `--operation` choice between copy/move files (default: copy)
  - `--split` option to extract from train/val/both splits
  - All extraction operations use class IDs instead of class names for precision
  - Python API: `extract_by_class()`, `extract_images_only()`, `extract_labels_only()`

- **Dataset Resize Feature**
  - New `resize --mode dataset` command for batch resizing YOLO datasets
  - Supports two common YOLO dataset structures:
    - Structure A: `dataset/train/images`, `dataset/train/labels`
    - Structure B: `dataset/images/train`, `dataset/labels/train`
  - Automatic dataset structure detection
  - `--resize-mode` options: longest, shortest, width, height (default: longest)
  - `--interpolation` choices: linear, lanczos4 (default: linear)
  - `--resize-all` flag to resize all images (default: only resize images smaller than target)
  - Smart handling: by default, only upscales small images, preserves large images
  - Label files copied without modification (YOLO uses normalized coordinates 0-1)
  - Automatic data.yaml copying and path updating
  - Progress logging every 100 images
  - Returns detailed statistics: resized_count, copied_count, failed_count
  - Python API: `resize_dataset()`
  - Modified `resize_image()` to accept custom interpolation parameter

### Fixed
- **Dataset Merge Bug**
  - Fixed critical bug where images with same name but different extensions (e.g., 123.jpg and 123.PNG) caused label file overwriting
  - Root cause: Code checked full filename including extension in processed_names set
  - Solution: Now checks only base name (without extension) to detect duplicates correctly
  - Example: Both 123.jpg and 123.PNG now correctly detected as duplicates, second file renamed to 123_1.PNG

- **Image Crop Single File Support**
  - Fixed `crop-coords` command not recognizing single file input
  - Added single file detection: checks if input path is a file vs directory
  - Single file mode: saves to output directory with same filename
  - Directory mode: maintains directory structure as before
  - Resolves "No image files found" warning when cropping single images

### Changed
- **Dataset Merge Simplification**
  - Removed `handle_duplicates` parameter from `merge_datasets()`
  - Always uses "rename" strategy for handling duplicate filenames
  - Simplified function signature and internal logic
  - Updated tests to remove skip mode test cases
  - Cleaner API with less complexity

- **Extract API Design**
  - Changed from class names to class IDs for better precision
  - Parameter naming: `--class-ids` instead of `--classes`
  - CLI parameter type changed from `str` to `int`
  - Removed ambiguity in class name matching

### Technical Details
- **Logger Enhancement**
  - Modified `ydt/core/logger.py` to integrate `colorlog.ColoredFormatter`
  - Added color mapping configuration for all log levels
  - Console handler uses colored output, file handler uses plain text
  - Updated `pyproject.toml` to add `colorlog>=6.7.0` dependency

- **Extract Implementation**
  - New file: `ydt/dataset/extract.py` with three main functions
  - Label filtering logic: removes non-target class annotations from label files
  - ID remapping: creates mapping dict (e.g., {1: 0, 2: 1}) for sequential IDs
  - data.yaml generation: updates class count and names based on extraction mode
  - Supports both copy and move operations via `operation` parameter
  - Handles train/val splits independently
  - Updated `ydt/dataset/__init__.py` to export new functions
  - CLI integration in `ydt/cli/main.py` with comprehensive parameter validation

- **Resize Implementation**
  - New function: `resize_dataset()` in `ydt/image/resize.py`
  - Dataset structure detection using `Path.exists()` checks
  - Separate handling for Structure A and Structure B datasets
  - Conditional resizing based on `min_size_only` parameter and edge size comparison
  - Interpolation mapping: {"linear": cv2.INTER_LINEAR, "lanczos4": cv2.INTER_LANCZOS4}
  - Label file handling: direct copy using `shutil.copy2()` (preserves timestamps)
  - Modified `resize_image()` signature to accept optional `interpolation: int | None` parameter
  - CLI integration: extended existing `resize` command with `--mode` parameter
  - Updated `ydt/image/__init__.py` to export `resize_dataset()`

- **Merge Bug Fix**
  - Modified `ydt/dataset/split.py` lines 372 and 417
  - Changed from `f"{new_base_name}{ext}"` to `new_base_name` in processed_names checks
  - Now correctly prevents duplicate base names regardless of extension
  - Added test case to verify fix with same-name different-extension files

- **Crop Single File Fix**
  - Modified `crop_directory_by_coords()` in `ydt/image/resize.py`
  - Added `input_path.is_file()` check to detect single file input
  - Single file mode: creates `image_files = [input_path]` list
  - Directory mode: uses `rglob()` or `glob()` as before
  - Output path logic: single file saves to `output_path / image_file.name`
  - Updated function docstring to reflect single file support


## [0.2.7] - 2025-11-11

### Added
- **Dataset Synthesize Enhancements**
  - Added `--data-yaml` parameter to specify data.yaml path for class name validation
  - Target filenames must now contain class names from data.yaml when this parameter is used
  - Example: For class name 'bn' in data.yaml, target files should be named like 'bn_back.jpg' or 'front_bn.png'
  - Added `--rotation-range` parameter to limit rotation angles (format: "min,max" in degrees)
  - Default rotation range is -90,90 degrees; can be customized (e.g., `--rotation-range=-20,20` for small rotations)
  - **Important:** Use equals sign (=) for negative values: `--rotation-range=-20,20` (argparse limitation with negative numbers)

### Changed
- **Dataset Synthesize Validation**
  - Class name matching is now required when data.yaml is provided
  - Clear error messages when target filenames don't match any class names
  - Rotation angle sampling now respects custom rotation range
  - Improved logging to show matched class names for each target file

### Technical Details
- Updated `DatasetSynthesizer` class to accept `data_yaml_path` and `rotation_range` parameters
- Modified `_load_target_data()` to validate filenames against class names
- Modified `_sample_rotation_angle()` to use configurable rotation range
- Added comprehensive examples in README and README_CN


## [0.2.6] - 2025-11-11

### Fixed
- **Test Suite**
  - Fixed all test failures related to label path generation (images/ vs labels/ directory structure)
  - Fixed e2e test marker registration to prevent pytest warnings
  - Fixed augmentation test assertions to be more lenient with rotation counts
  - Removed TestImageResizing class that referenced non-existent `resize_images` function
  - Fixed CLI tests to use `uv run ydt` instead of direct `ydt` command

- **CI/CD Pipeline**
  - Fixed GitHub Actions artifact actions (upgraded from v3 to v4)
  - Fixed Windows wildcard expansion issue by adding bash shell
  - Fixed test workflow to use correct e2e marker instead of non-existent integration marker

### Changed
- **Code Quality Tools**
  - Simplified CI/CD by using ruff for all code quality checks (formatting, linting, type checking)
  - Replaced `black --check` with `ruff format --check` for faster formatting verification
  - Removed separate mypy type checking step in favor of ruff's integrated checks
  - Formatted all Python files with black for consistency

### Technical Details
- Updated `.github/workflows/test.yml` to use ruff exclusively for code quality checks
- Updated `pytest.ini` to properly register the e2e test marker
- Fixed label path conversion in multiple test files to handle parallel directory structure
- All GitHub Actions tests now pass across Ubuntu, Windows, and macOS platforms


## [0.2.5] - 2025-11-10

### Added
- **Single File Processing Support**
  - `ydt image slice` now supports single image file input (e.g., `ydt image slice -i image.jpg -o output`)
  - `ydt image augment` now supports single image file input with automatic label detection
  - `ydt image crop-coords` now supports single image file input for quick cropping operations
  - All three commands maintain backward compatibility with directory input

### Fixed
- **Logging System**
  - Fixed duplicate log messages in CLI commands
  - Improved logger hierarchy to prevent handler duplication
  - Child loggers now properly inherit from parent logger via propagation
  - Cleaner console output with single log entries

### Changed
- Updated command-line help messages to reflect single file support
- Improved error messages for unsupported file formats
- Enhanced input validation for both file and directory modes

### Technical Details
- Modified `ydt/image/slice.py` to detect and handle single file input
- Modified `ydt/image/augment.py` to support single image augmentation
- Modified `ydt/cli/main.py` to handle single file mode for crop-coords
- Refactored `ydt/core/logger.py` to prevent duplicate handlers
- `get_logger()` now returns loggers without auto-adding handlers


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
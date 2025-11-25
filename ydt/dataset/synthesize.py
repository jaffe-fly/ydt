"""
Dataset synthesis for object detection with OBB annotations.

Provides tools for generating synthetic datasets by compositing target objects
onto background images with automatic OBB label generation.
"""

import math
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from ydt.core import IMAGE_EXTENSIONS
from ydt.core.logger import get_logger

logger = get_logger(__name__)


class DatasetSynthesizer:
    """
    Synthesize OBB (Oriented Bounding Box) datasets by compositing targets onto backgrounds.

    This class handles:
    - Loading target objects and background images
    - Rotating and scaling targets with proper mask handling
    - Placing objects on backgrounds with overlap constraints
    - Generating YOLO OBB format annotations
    - Splitting into train/val sets

    Examples:
        >>> synthesizer = DatasetSynthesizer(
        ...     target_dir="./targets",
        ...     background_dir="./backgrounds",
        ...     output_dir="./synthetic_dataset",
        ...     class_names={0: "card", 1: "dice"}
        ... )
        >>> synthesizer.synthesize_dataset(num_images=1000, class_names={0: "card", 1: "dice"})
    """

    def __init__(
        self,
        target_dir: str | Path,
        background_dir: str | Path,
        output_dir: str | Path,
        target_size_range: tuple[float, float] = (0.1, 0.3),
        max_overlap_ratio: float = 0.5,
        min_objects_per_image: int = 1,
        max_objects_per_image: int = 12,
        train_ratio: float = 0.8,
        class_names: dict[int, str] | None = None,
        target_area_ratio: tuple[float, float] = (0.04, 0.06),
        objects_per_image: int | tuple[int, int] | None = None,
        split_mode: str = "trainval",
        data_yaml_path: str | Path | None = None,
        rotation_range: tuple[float, float] | None = None,
        annotation_format: str = "obb",
    ):
        """
        Initialize dataset synthesizer.

        Args:
            target_dir: Directory containing target object images
            background_dir: Directory containing background images
            output_dir: Output directory for synthetic dataset
            target_size_range: Min and max relative size of targets (0-1)
            max_overlap_ratio: Maximum allowed overlap ratio between objects
            min_objects_per_image: Minimum objects to place per image
            max_objects_per_image: Maximum objects to place per image
            train_ratio: Ratio of training data (0-1)
            class_names: Mapping of class IDs to names
            target_area_ratio: Target area as fraction of background area
            objects_per_image: Objects per image (int) or range (tuple) overriding min/max_objects
            split_mode: Dataset split mode ("train" or "trainval")
            data_yaml_path: Optional path to data.yaml for class name validation
            rotation_range: Optional tuple (min_angle, max_angle) for rotation limits in degrees
            annotation_format: Annotation format - "obb" (Oriented Bounding Box) or "hbb" (Horizontal Bounding Box)

        Raises:
            RuntimeError: If no valid target or background images found
            ValueError: If target filenames don't match class names in data.yaml
        """
        self.target_dir = Path(target_dir)
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        self.target_size_range = target_size_range
        self.max_overlap_ratio = max_overlap_ratio
        self.train_ratio = train_ratio
        self.target_area_ratio = target_area_ratio
        self.split_mode = split_mode
        self.annotation_format = annotation_format.lower()

        if self.annotation_format not in ["obb", "hbb"]:
            raise ValueError(
                f"Invalid annotation_format: {annotation_format}. Must be 'obb' or 'hbb'"
            )

        # Load class names from data.yaml if provided
        if data_yaml_path:
            data_yaml_path = Path(data_yaml_path)
            if not data_yaml_path.exists():
                raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

            with open(data_yaml_path, encoding="utf-8") as f:
                data_config = yaml.safe_load(f)

            # Extract class names from data.yaml
            if "names" not in data_config:
                raise ValueError(f"No 'names' field found in {data_yaml_path}")

            yaml_names = data_config["names"]
            if isinstance(yaml_names, dict):
                # Format: {0: "class1", 1: "class2"}
                self.class_names = yaml_names
            elif isinstance(yaml_names, list):
                # Format: ["class1", "class2"]
                self.class_names = dict(enumerate(yaml_names))
            else:
                raise ValueError(f"Invalid 'names' format in {data_yaml_path}")

            logger.info(f"Loaded class names from {data_yaml_path}: {self.class_names}")
        else:
            self.class_names = class_names or {}

        self.name_to_id = {name: cid for cid, name in self.class_names.items()}

        # Set rotation range
        if rotation_range:
            self.rotation_range = rotation_range
            logger.info(
                f"Using custom rotation range: {rotation_range[0]}° to {rotation_range[1]}°"
            )
        else:
            self.rotation_range = (-90.0, 90.0)  # Default range

        # Handle objects_per_image parameter
        if objects_per_image is not None:
            if isinstance(objects_per_image, int):
                # Single number
                self.min_objects_per_image = objects_per_image
                self.max_objects_per_image = objects_per_image
            elif isinstance(objects_per_image, tuple) and len(objects_per_image) == 2:
                # Range
                self.min_objects_per_image, self.max_objects_per_image = objects_per_image
            else:
                raise ValueError(f"Invalid objects_per_image format: {objects_per_image}")
        else:
            self.min_objects_per_image = min_objects_per_image
            self.max_objects_per_image = max_objects_per_image

        self._create_output_directories()
        self.target_data = self._load_target_data()
        self.background_images = self._load_background_images()

        logger.info(f"Loaded {len(self.target_data)} target samples")
        logger.info(f"Loaded {len(self.background_images)} background images")

    def _create_output_directories(self) -> None:
        """Create output directory structure"""
        # Always create train directories
        for sub in [
            self.output_dir / "images" / "train",
            self.output_dir / "labels" / "train",
        ]:
            sub.mkdir(parents=True, exist_ok=True)

        # Create val directories only if split_mode is "trainval"
        if self.split_mode == "trainval":
            for sub in [
                self.output_dir / "images" / "val",
                self.output_dir / "labels" / "val",
            ]:
                sub.mkdir(parents=True, exist_ok=True)

    def _load_target_data(self) -> list[dict]:
        """
        Load target object images with masks and class information.

        Returns:
            List of target data dictionaries

        Raises:
            RuntimeError: If no valid target images found
            ValueError: If class names are required but filename doesn't match any class
        """
        class_name_list = sorted(self.name_to_id.keys(), key=len, reverse=True)
        targets: list[dict] = []
        require_class_validation = bool(self.class_names)  # Validate if class names are provided

        for img_path in self.target_dir.rglob("*"):
            if img_path.suffix not in IMAGE_EXTENSIONS:
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning(f"Cannot read target image: {img_path}")
                continue

            # Handle different image formats
            mask: np.ndarray | None = None
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3:
                if img.shape[2] == 4:
                    mask = img[:, :, 3]
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif img.shape[2] > 4:
                    img = img[:, :, :3]

            # Validate mask
            if mask is not None and np.count_nonzero(mask) == 0:
                mask = None

            # Infer class from filename
            class_id = 0
            matched_class_name = None
            stem_lower = img_path.stem.lower()

            for name in class_name_list:
                if name.lower() in stem_lower:
                    class_id = self.name_to_id[name]
                    matched_class_name = name
                    break

            # Validate that filename contains a class name if class names are provided
            if require_class_validation and matched_class_name is None:
                available_names = ", ".join([f"'{name}'" for name in class_name_list])
                raise ValueError(
                    f"Target filename '{img_path.name}' does not contain any class name from data.yaml.\n"
                    f"Available class names: {available_names}\n"
                    f"Example: For class name 'bn', filename should be 'bn_back.jpg' or 'front_bn.png'"
                )

            if matched_class_name:
                logger.info(
                    f"Matched '{img_path.name}' to class '{matched_class_name}' (ID: {class_id})"
                )

            targets.append(
                {
                    "image": img,
                    "annotations": [
                        {
                            "class_id": class_id,
                            "points": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                        }
                    ],
                    "filename": img_path.name,
                    "height": img.shape[0],
                    "width": img.shape[1],
                    "mask": mask,
                }
            )

        if not targets:
            raise RuntimeError(f"No valid target images found in {self.target_dir}")

        return targets

    def _load_background_images(self) -> list[np.ndarray]:
        """
        Load background images.

        Returns:
            List of background images

        Raises:
            RuntimeError: If no valid background images found
        """
        backgrounds: list[np.ndarray] = []

        for img_path in self.background_dir.rglob("*"):
            if img_path.suffix not in IMAGE_EXTENSIONS:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Cannot read background image: {img_path}")
                continue
            backgrounds.append(img)

        if not backgrounds:
            raise RuntimeError(f"No valid background images found in {self.background_dir}")

        return backgrounds

    def _sample_rotation_angle(self) -> float:
        """
        Sample rotation angle from configured range.

        Returns:
            Random angle within self.rotation_range, avoiding near-zero angles (±5°) when possible
        """
        min_angle, max_angle = self.rotation_range
        max_attempts = 50

        for _ in range(max_attempts):
            angle = random.uniform(min_angle, max_angle)
            # Avoid near-zero angles if the range allows it
            if abs(angle) >= 5 or (min_angle < -5 or max_angle > 5):
                return angle

        # If we can't avoid near-zero after max_attempts, just return a random angle
        return random.uniform(min_angle, max_angle)

    def _rotate_with_padding(
        self, image: np.ndarray, mask: np.ndarray, angle: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rotate image and mask with padding to avoid cropping.

        Args:
            image: Input image
            mask: Input mask
            angle: Rotation angle in degrees

        Returns:
            Tuple of (rotated_image, rotated_mask, transform_matrix)
        """
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new bounds
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        bound_w = max(1, int(math.ceil(h * sin + w * cos)))
        bound_h = max(1, int(math.ceil(h * cos + w * sin)))

        # Add padding
        padding = 4
        bound_w += padding * 2
        bound_h += padding * 2

        # Adjust translation
        M[0, 2] += bound_w / 2.0 - center[0]
        M[1, 2] += bound_h / 2.0 - center[1]

        # Rotate image and mask
        rotated = cv2.warpAffine(
            image,
            M,
            (bound_w, bound_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        rotated_mask = cv2.warpAffine(
            mask,
            M,
            (bound_w, bound_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Create valid region mask
        valid_src = np.ones((h, w), dtype=np.uint8) * 255
        valid_mask = cv2.warpAffine(
            valid_src,
            M,
            (bound_w, bound_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        rotated_mask = cv2.bitwise_and(rotated_mask, valid_mask)

        # Smooth mask edges
        soft = cv2.GaussianBlur(rotated_mask, (5, 5), 0)
        _, core = cv2.threshold(rotated_mask, 200, 255, cv2.THRESH_BINARY)
        rotated_mask = np.where(core > 0, 255, soft).astype(np.uint8)

        return rotated, rotated_mask, M

    def _resize_and_rotate_target(
        self,
        target_img: np.ndarray,
        target_annotations: list[dict],
        bg_width: int,
        bg_height: int,
        target_mask: np.ndarray | None = None,
        desired_short_side: int | None = None,
    ) -> tuple[np.ndarray, list[dict], np.ndarray]:
        """
        Resize and rotate target with proper annotation transformation.

        Args:
            target_img: Target image
            target_annotations: List of annotations
            bg_width: Background width
            bg_height: Background height
            target_mask: Optional mask
            desired_short_side: Desired short side length

        Returns:
            Tuple of (transformed_image, transformed_annotations, mask)
        """
        th, tw = target_img.shape[:2]

        # Ensure mask matches image size
        if target_mask is not None and target_mask.shape[:2] != (th, tw):
            target_mask = cv2.resize(target_mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # IMPORTANT: Limit maximum size before rotation to prevent memory issues
        max_dimension = 2000  # Maximum width or height before rotation
        if max(th, tw) > max_dimension:
            scale_limit = max_dimension / max(th, tw)
            new_w = max(1, int(round(tw * scale_limit)))
            new_h = max(1, int(round(th * scale_limit)))
            target_img = cv2.resize(target_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if target_mask is not None:
                target_mask = cv2.resize(
                    target_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
                )
            th, tw = new_h, new_w

        # Scale down to desired short side
        base_short = max(1, min(th, tw))
        if desired_short_side is not None:
            scale = min(1.0, float(desired_short_side) / float(base_short))
        else:
            scale = 1.0

        new_w = max(1, int(round(tw * scale)))
        new_h = max(1, int(round(th * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(target_img, (new_w, new_h), interpolation=interp)

        # Process mask
        if target_mask is not None:
            fg_mask = cv2.resize(target_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            if fg_mask.ndim == 3:
                fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
            fg_mask = np.where(fg_mask > 0, 255, 0).astype(np.uint8)

            # Check mask coverage
            coverage = float(np.count_nonzero(fg_mask)) / max(1, fg_mask.size)
            if coverage < 0.02:
                fg_mask = np.ones((new_h, new_w), dtype=np.uint8) * 255
        else:
            fg_mask = np.ones((new_h, new_w), dtype=np.uint8) * 255

        # Rotate
        angle = self._sample_rotation_angle()
        rotated, rotated_mask, transform = self._rotate_with_padding(resized, fg_mask, angle)

        # Scale to fit background
        rot_h, rot_w = rotated_mask.shape[:2]
        scale_w = bg_width / max(1, rot_w)
        scale_h = bg_height / max(1, rot_h)
        fit_scale = min(1.0, scale_w, scale_h)

        if fit_scale < 1.0:
            final_w = max(1, int(round(rot_w * fit_scale)))
            final_h = max(1, int(round(rot_h * fit_scale)))
            rotated = cv2.resize(rotated, (final_w, final_h), interpolation=cv2.INTER_AREA)
            rotated_mask = cv2.resize(
                rotated_mask, (final_w, final_h), interpolation=cv2.INTER_NEAREST
            )
        else:
            final_w, final_h = rot_w, rot_h

        # Scale by area ratio
        area_ratio = random.uniform(*self.target_area_ratio)
        bg_area = float(max(1, bg_width * bg_height))
        desired_area = float(area_ratio) * bg_area
        current_area = float(max(1, int(np.count_nonzero(rotated_mask > 0))))
        area_scale = math.sqrt(desired_area / current_area)

        if not np.isfinite(area_scale):
            area_scale = 1.0
        area_scale = float(max(0.0, min(1.0, area_scale)))

        if area_scale < 1.0:
            new_w2 = max(1, int(round(final_w * area_scale)))
            new_h2 = max(1, int(round(final_h * area_scale)))
            if new_w2 != final_w or new_h2 != final_h:
                rotated = cv2.resize(rotated, (new_w2, new_h2), interpolation=cv2.INTER_AREA)
                rotated_mask = cv2.resize(
                    rotated_mask, (new_w2, new_h2), interpolation=cv2.INTER_NEAREST
                )
                final_w, final_h = new_w2, new_h2

        # Transform annotations
        rotated_annotations: list[dict] = []
        total_scale = 1.0
        if fit_scale < 1.0:
            total_scale *= fit_scale
        if area_scale < 1.0:
            total_scale *= area_scale

        for ann in target_annotations:
            pts = np.array(
                [
                    [0.0 * new_w, 0.0 * new_h],
                    [1.0 * new_w, 0.0 * new_h],
                    [1.0 * new_w, 1.0 * new_h],
                    [0.0 * new_w, 1.0 * new_h],
                ],
                dtype=np.float32,
            )

            pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
            pts_rot = (pts_h @ transform.T).astype(np.float32)

            if total_scale != 1.0:
                pts_rot *= total_scale

            norm_pts = [(p[0] / final_w, p[1] / final_h) for p in pts_rot]
            rotated_annotations.append({"class_id": ann["class_id"], "points": norm_pts})

        return rotated, rotated_annotations, rotated_mask

    def _overlaps_too_much(self, placed: list[dict], new_obbs: list[dict]) -> bool:
        """
        Check if new OBBs overlap too much with existing objects.

        Args:
            placed: List of already placed objects
            new_obbs: List of new OBBs to check

        Returns:
            True if overlap exceeds threshold
        """
        for existing in placed:
            for existing_obb in existing["obbs"]:
                poly1 = np.array(existing_obb["points"], dtype=np.float32)
                area1 = abs(cv2.contourArea(poly1))
                if area1 <= 0:
                    continue

                for obb in new_obbs:
                    poly2 = np.array(obb["points"], dtype=np.float32)
                    area2 = abs(cv2.contourArea(poly2))
                    if area2 <= 0:
                        continue

                    try:
                        ret, inter = cv2.intersectConvexConvex(poly1, poly2)
                    except cv2.error:
                        ret, inter = -1, None

                    if ret <= 0 or inter is None:
                        continue

                    inter_area = abs(cv2.contourArea(inter))

                    # Check overlap ratio for both objects
                    if inter_area / area1 > self.max_overlap_ratio:
                        return True
                    if inter_area / area2 > self.max_overlap_ratio:
                        return True

        return False

    def _place_target_on_background(
        self,
        background: np.ndarray,
        target_img: np.ndarray,
        target_annotations: list[dict],
        existing_objects: list[dict],
        target_mask: np.ndarray | None = None,
    ) -> dict | None:
        """
        Place target on background at random valid position.

        Args:
            background: Background image (modified in-place)
            target_img: Target image to place
            target_annotations: Target annotations
            existing_objects: Already placed objects
            target_mask: Optional mask for blending

        Returns:
            Placement info dict or None if placement failed
        """
        bg_h, bg_w = background.shape[:2]
        tgt_h, tgt_w = target_img.shape[:2]

        if tgt_w > bg_w or tgt_h > bg_h:
            return None

        # Try random placements
        for _ in range(80):
            x = random.randint(0, bg_w - tgt_w)
            y = random.randint(0, bg_h - tgt_h)

            # Transform annotations to absolute coordinates
            target_obbs = []
            for ann in target_annotations:
                points = []
                for px, py in ann["points"]:
                    abs_x = (px * tgt_w + x) / bg_w
                    abs_y = (py * tgt_h + y) / bg_h
                    points.append((abs_x, abs_y))
                target_obbs.append({"class_id": ann["class_id"], "points": points})

            # Check overlap
            if self._overlaps_too_much(existing_objects, target_obbs):
                continue

            # Place target on background
            roi = background[y : y + tgt_h, x : x + tgt_w]
            if target_mask is None:
                roi[:, :] = target_img
            else:
                if target_mask.ndim == 2:
                    alpha = target_mask[:, :, None].astype(np.float32) / 255.0
                else:
                    alpha = target_mask.astype(np.float32) / 255.0
                alpha = np.clip(alpha, 0.0, 1.0)

                blended = (
                    roi.astype(np.float32) * (1.0 - alpha) + target_img.astype(np.float32) * alpha
                )
                roi[:, :] = blended.astype(np.uint8)

            return {
                "x": x,
                "y": y,
                "width": tgt_w,
                "height": tgt_h,
                "obbs": target_obbs,
            }

        return None

    def _synthesize_single_image(self, background: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Synthesize a single image by placing multiple targets on background.

        Args:
            background: Background image

        Returns:
            Tuple of (synthesized_image, placed_objects)
        """
        bg_h, bg_w = background.shape[:2]
        synthesized = background.copy()

        desired_targets = random.randint(self.min_objects_per_image, self.max_objects_per_image)

        # Sample target indices
        if len(self.target_data) == 0:
            return synthesized, []

        indices = [random.randrange(len(self.target_data)) for _ in range(desired_targets)]

        # Calculate minimum short side
        candidate_shorts: list[int] = []
        for idx in indices:
            td = self.target_data[idx]
            candidate_shorts.append(int(max(1, min(td["height"], td["width"]))))
        desired_short_side = int(min(candidate_shorts)) if candidate_shorts else 1

        # Place objects
        placed_objects: list[dict] = []
        attempts = 0
        max_attempts = max(200, desired_targets * 60)

        while len(placed_objects) < desired_targets and attempts < max_attempts:
            attempts += 1
            idx = indices[attempts % len(indices)]
            target_data = self.target_data[idx]

            resized_img, resized_ann, rotated_mask = self._resize_and_rotate_target(
                target_data["image"],
                target_data["annotations"],
                bg_w,
                bg_h,
                target_data.get("mask"),
                desired_short_side=desired_short_side,
            )

            placement = self._place_target_on_background(
                synthesized,
                resized_img,
                resized_ann,
                placed_objects,
                rotated_mask,
            )

            if placement is not None:
                placed_objects.append(placement)

        return synthesized, placed_objects

    def _generate_yolo_annotations(self, placed_objects: list[dict]) -> list[str]:
        """Generate YOLO format annotations (OBB or HBB)"""
        annotations: list[str] = []
        for obj in placed_objects:
            for obb in obj["obbs"]:
                class_id = obb["class_id"]
                points = obb["points"]

                if self.annotation_format == "obb":
                    # OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    line = str(class_id)
                    for px, py in points:
                        line += f" {px:.6f} {py:.6f}"
                    annotations.append(line)
                else:  # hbb
                    # HBB format: class_id center_x center_y width height
                    # Convert OBB points to bounding box
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    center_x = (x_min + x_max) / 2.0
                    center_y = (y_min + y_max) / 2.0
                    width = x_max - x_min
                    height = y_max - y_min

                    line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    annotations.append(line)
        return annotations

    def _create_data_yaml(self, class_names: dict[int, str]) -> None:
        """Create data.yaml configuration file"""
        yaml_content = {
            "names": class_names,
            "path": "./",
            "train": "images/train",
        }

        # Only include val field if split_mode is "trainval"
        if self.split_mode == "trainval":
            yaml_content["val"] = "images/val"

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    def synthesize_dataset(
        self, num_images: int, class_names: dict[int, str] | None = None
    ) -> dict[str, int]:
        """
        Synthesize complete dataset with train/val splits.

        Args:
            num_images: Total number of images to generate
            class_names: Optional class name mapping (uses self.class_names if None)

        Returns:
            Dictionary with synthesis statistics

        Examples:
            >>> synthesizer = DatasetSynthesizer(...)
            >>> stats = synthesizer.synthesize_dataset(1000)
            >>> print(f"Generated {stats['train_count']} train, {stats['val_count']} val")
        """
        if class_names is None:
            class_names = self.class_names

        logger.info(f"Synthesizing {num_images} images with split mode: {self.split_mode}")

        if self.split_mode == "trainval":
            num_train = int(num_images * self.train_ratio)
            num_val = num_images - num_train
            logger.info(f"Train: {num_train}, Val: {num_val}")
        else:
            num_train = num_images
            num_val = 0
            logger.info(f"Train only: {num_train}")

        # Generate training set
        for i in tqdm(range(num_train), desc="Synthesizing train", unit="img"):
            background = random.choice(self.background_images)
            synthesized, placed_objects = self._synthesize_single_image(background)
            annotations = self._generate_yolo_annotations(placed_objects)

            img_path = self.output_dir / "images" / "train" / f"train_{i:06d}.jpg"
            label_path = self.output_dir / "labels" / "train" / f"train_{i:06d}.txt"

            cv2.imwrite(str(img_path), synthesized)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(annotations))

        # Generate validation set only if split_mode is "trainval"
        if self.split_mode == "trainval" and num_val > 0:
            for i in tqdm(range(num_val), desc="Synthesizing val", unit="img"):
                background = random.choice(self.background_images)
                synthesized, placed_objects = self._synthesize_single_image(background)
                annotations = self._generate_yolo_annotations(placed_objects)

                img_path = self.output_dir / "images" / "val" / f"val_{i:06d}.jpg"
                label_path = self.output_dir / "labels" / "val" / f"val_{i:06d}.txt"

                cv2.imwrite(str(img_path), synthesized)
                with open(label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(annotations))

        self._create_data_yaml(class_names)
        logger.info(f"Dataset synthesis complete! Output: {self.output_dir}")

        return {
            "train_count": num_train,
            "val_count": num_val,
            "output_dir": str(self.output_dir),
        }

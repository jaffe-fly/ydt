"""
数据集结构处理模块

提供YOLO数据集目录结构的创建和管理功能。
"""

import shutil
from datetime import datetime
from pathlib import Path

from ydt.core.logger import get_logger

logger = get_logger(__name__)


class DatasetStructure:
    """YOLO数据集结构管理器"""

    def __init__(self, output_dir: str | Path | None = None):
        """
        初始化数据集结构管理器

        Args:
            output_dir: 输出目录，如果为None则自动创建
        """
        if output_dir is None:
            # 自动创建带时间戳的输出目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path.cwd() / f"auto_label_{timestamp}"
        else:
            self.output_dir = Path(output_dir)

        # 创建目录结构
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"

        logger.info(f"输出目录: {self.output_dir}")

    def create_structure(self) -> None:
        """创建YOLO数据集目录结构"""
        try:
            # 创建输出目录
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(exist_ok=True)
            self.labels_dir.mkdir(exist_ok=True)

            logger.info("已创建数据集结构:")
            logger.info(f"  输出目录: {self.output_dir}")
            logger.info(f"  图片目录: {self.images_dir}")
            logger.info(f"  标签目录: {self.labels_dir}")

        except Exception as e:
            logger.error(f"创建数据集结构失败: {e}")
            raise

    def copy_image(self, image_path: str | Path) -> Path:
        """
        复制图片到数据集目录

        Args:
            image_path: 原图片路径

        Returns:
            Path: 复制后的图片路径
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        target_path = self.images_dir / image_path.name

        try:
            shutil.copy2(image_path, target_path)
            logger.debug(f"已复制图片: {image_path.name}")
            return target_path

        except Exception as e:
            logger.error(f"复制图片失败 {image_path}: {e}")
            raise

    def save_label(self, image_name: str, detections: list[dict], format_type: str) -> Path:
        """
        保存标签文件

        Args:
            image_name: 图片名称
            detections: 检测结果列表
            format_type: 格式类型 ("bbox" 或 "obb")

        Returns:
            Path: 标签文件路径
        """
        # 生成标签文件名
        label_name = Path(image_name).stem + ".txt"
        label_path = self.labels_dir / label_name

        try:
            with open(label_path, "w", encoding="utf-8") as f:
                for detection in detections:
                    class_id = detection["class_id"]
                    coords = detection["coordinates"]

                    if format_type == "obb":
                        # OBB格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
                        if len(coords) == 8:
                            line = f"{class_id} " + " ".join([f"{c:.6f}" for c in coords]) + "\n"
                        else:
                            logger.warning(f"OBB坐标数量不正确: {len(coords)}")
                            continue
                    else:
                        # BBox格式: class_id x_center y_center width height
                        if len(coords) == 4:
                            line = f"{class_id} " + " ".join([f"{c:.6f}" for c in coords]) + "\n"
                        else:
                            logger.warning(f"BBox坐标数量不正确: {len(coords)}")
                            continue

                    f.write(line)

            logger.debug(f"已保存标签: {label_name}")
            return label_path

        except Exception as e:
            logger.error(f"保存标签失败 {label_path}: {e}")
            raise

    def get_stats(self) -> dict:
        """获取数据集统计信息"""
        try:
            image_count = len(list(self.images_dir.glob("*"))) if self.images_dir.exists() else 0
            label_count = (
                len(list(self.labels_dir.glob("*.txt"))) if self.labels_dir.exists() else 0
            )

            return {
                "output_dir": str(self.output_dir),
                "image_count": image_count,
                "label_count": label_count,
                "images_dir": str(self.images_dir),
                "labels_dir": str(self.labels_dir),
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    def clear_empty_files(self) -> tuple[int, int]:
        """
        清理空文件

        Returns:
            Tuple[int, int]: (清理的图片数量, 清理的标签数量)
        """
        cleaned_images = 0
        cleaned_labels = 0

        try:
            # 清理空的标签文件
            if self.labels_dir.exists():
                for label_file in self.labels_dir.glob("*.txt"):
                    if label_file.stat().st_size == 0:
                        label_file.unlink()
                        cleaned_labels += 1
                        logger.debug(f"删除空标签文件: {label_file.name}")

            # 清理没有对应标签的图片
            if self.images_dir.exists():
                for image_file in self.images_dir.glob("*"):
                    if image_file.is_file():
                        label_file = self.labels_dir / (image_file.stem + ".txt")
                        if not label_file.exists():
                            image_file.unlink()
                            cleaned_images += 1
                            logger.debug(f"删除无标签图片: {image_file.name}")

            if cleaned_images > 0 or cleaned_labels > 0:
                logger.info(f"清理完成: {cleaned_images} 张图片, {cleaned_labels} 个标签文件")

            return cleaned_images, cleaned_labels

        except Exception as e:
            logger.error(f"清理空文件失败: {e}")
            return 0, 0

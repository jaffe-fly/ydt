"""
模型预测工具模块

提供YOLO模型的加载和预测功能。
"""

from pathlib import Path

import numpy as np

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

from ydt.core.logger import get_logger

logger = get_logger(__name__)


class ModelPredictor:
    """YOLO模型预测器"""

    def __init__(self, model_path: str | Path, device: int | str = 0):
        """
        初始化模型预测器

        Args:
            model_path: YOLO模型路径
            device: 设备ID，0为GPU，"cpu"为CPU
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics未安装，请安装: pip install ultralytics")

        self.model_path = Path(model_path)
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        logger.info(f"正在加载模型: {model_path}")
        self.model = YOLO(str(model_path))

        # 检测模型类型
        self.is_obb_model = self._detect_obb_model()
        logger.info(f"模型类型: {'OBB' if self.is_obb_model else 'BBox'}")

    def _detect_obb_model(self) -> bool:
        """检测是否为OBB模型"""
        try:
            # 通过模型名称或任务类型判断
            if hasattr(self.model.model, "names"):
                # 尝试进行一次预测来判断
                dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
                results = self.model(dummy_input, verbose=False)

                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, "obb") and result.obb is not None:
                        return True
                    elif hasattr(result, "boxes") and result.boxes is not None:
                        return False

            # 通过模型文件名判断
            model_name = self.model_path.name.lower()
            if "obb" in model_name:
                return True

            return False
        except Exception as e:
            logger.warning(f"检测模型类型失败，默认使用BBox: {e}")
            return False

    def predict(
        self, image_path: str | Path, conf_threshold: float = 0.25, iou_threshold: float = 0.7
    ) -> tuple[list[dict], str]:
        """
        对单张图片进行预测

        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            Tuple[List[dict], str]: (检测结果列表, 格式类型)
            结果格式: [{"class_id": int, "coordinates": list, "confidence": float}]
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        try:
            # 进行预测
            results = self.model(
                str(image_path), conf=conf_threshold, iou=iou_threshold, verbose=False
            )

            if not results:
                return [], "bbox"

            result = results[0]
            detections = []

            if self.is_obb_model and hasattr(result, "obb") and result.obb is not None:
                # OBB格式结果
                boxes = result.obb
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())
                        # OBB坐标: 4个点的x,y坐标
                        coords = box.xyxyxyxyn[0].cpu().numpy().tolist()
                        detections.append(
                            {"class_id": class_id, "coordinates": coords, "confidence": confidence}
                        )
                format_type = "obb"
            else:
                # BBox格式结果
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())
                        # BBox坐标: x_center, y_center, width, height (归一化)
                        xywhn = box.xywhn[0].cpu().numpy()
                        coords = xywhn.tolist()
                        detections.append(
                            {"class_id": class_id, "coordinates": coords, "confidence": confidence}
                        )
                format_type = "bbox"

            return detections, format_type

        except Exception as e:
            logger.error(f"预测失败 {image_path}: {e}")
            raise

    def get_class_names(self) -> list[str]:
        """获取模型类别名称"""
        if hasattr(self.model.model, "names"):
            return list(self.model.model.names.values())
        return []

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_path": str(self.model_path),
            "is_obb": self.is_obb_model,
            "device": self.device,
            "class_names": self.get_class_names(),
        }

"""
自动标注核心功能模块

提供完整的自动标注流程，包括模型预测、数据集结构创建和标签生成。
"""

from pathlib import Path

import yaml

from ydt.core import IMAGE_EXTENSIONS
from ydt.core.logger import get_logger

from .model_utils import ModelPredictor
from .structure import DatasetStructure

logger = get_logger(__name__)


def create_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    """
    创建data.yaml配置文件

    Args:
        output_dir: 输出目录
        class_names: 类别名称列表

    Returns:
        Path: yaml文件路径
    """
    yaml_path = output_dir / "data.yaml"

    try:
        # 创建精简的data.yaml，只包含必要字段
        data = {
            "names": dict(enumerate(class_names)),
            "path": str(output_dir.absolute()),
            "train": "images",
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"已创建data.yaml文件: {yaml_path}")
        logger.info(f"包含 {len(class_names)} 个类别")

        return yaml_path

    except Exception as e:
        logger.error(f"创建data.yaml失败: {e}")
        raise


def validate_format(format_type: str) -> str:
    """
    验证并标准化格式类型

    Args:
        format_type: 输入的格式类型

    Returns:
        str: 标准化的格式类型 ("bbox" 或 "obb")

    Raises:
        ValueError: 格式类型不支持
    """
    format_type = format_type.lower().strip()
    if format_type not in ["bbox", "obb"]:
        raise ValueError(f"不支持的格式类型: {format_type}，请使用 'bbox' 或 'obb'")
    return format_type


def auto_label_dataset(
    input_dir: str | Path,
    model_path: str | Path,
    format_type: str,
    output_dir: str | Path | None = None,
    device: int | str = 0,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    dry_run: bool = False,
) -> dict:
    """
    自动标注数据集

    Args:
        input_dir: 输入图片目录
        model_path: YOLO模型路径
        format_type: 输出格式 ("bbox" 或 "obb")
        output_dir: 输出目录（可选）
        device: 设备ID
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
        dry_run: 预览模式

    Returns:
        dict: 处理结果统计

    Raises:
        ValueError: 参数验证失败
        FileNotFoundError: 文件不存在
        ImportError: 缺少依赖
    """
    # 参数验证
    input_dir = Path(input_dir)
    model_path = Path(model_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"输入路径不是目录: {input_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    format_type = validate_format(format_type)

    logger.info("=" * 60)
    logger.info("开始自动标注")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"模型路径: {model_path}")
    logger.info(f"输出格式: {format_type}")
    logger.info(f"设备: {device}")
    logger.info(f"置信度阈值: {conf_threshold}")
    logger.info(f"IOU阈值: {iou_threshold}")
    logger.info(f"预览模式: {dry_run}")
    logger.info("=" * 60)

    # 创建数据集结构
    dataset_structure = DatasetStructure(output_dir)

    if dry_run:
        logger.info("[预览模式] 不会实际创建文件和复制图片")
        logger.info(f"输出目录将是: {dataset_structure.output_dir}")
    else:
        dataset_structure.create_structure()

    # 加载模型
    try:
        predictor = ModelPredictor(model_path, device)
        model_info = predictor.get_model_info()
        logger.info(f"模型加载成功: {model_info}")

        class_names = model_info["class_names"]
        if not class_names:
            raise ValueError("无法获取模型类别信息")

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

    # 扫描图片文件
    image_files = []

    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    # 递归搜索子目录
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_dir.rglob(f"*{ext}"))
        image_files.extend(input_dir.rglob(f"*{ext.upper()}"))

    if not image_files:
        logger.warning(f"在目录 {input_dir} 中未找到图片文件")
        return {"success": False, "message": "未找到图片文件", "stats": {}}

    logger.info(f"找到 {len(image_files)} 张图片")

    # 检查格式兼容性
    if format_type == "obb" and not model_info["is_obb"]:
        logger.warning("模型不是OBB类型，但要求输出OBB格式，结果可能不准确")
    elif format_type == "bbox" and model_info["is_obb"]:
        logger.info("OBB模型输出BBox格式，将自动转换")

    # 处理每张图片
    processed_count = 0
    skipped_count = 0
    total_detections = 0

    for i, image_path in enumerate(image_files, 1):
        try:
            logger.info(f"处理 [{i}/{len(image_files)}]: {image_path.name}")

            # 进行预测
            detections, detected_format = predictor.predict(
                image_path, conf_threshold, iou_threshold
            )

            if not detections:
                logger.debug(f"未检测到目标: {image_path.name}")
                skipped_count += 1
                continue

            total_detections += len(detections)

            if not dry_run:
                # 复制图片
                dataset_structure.copy_image(image_path)

                # 保存标签
                dataset_structure.save_label(image_path.name, detections, format_type)

            processed_count += 1
            logger.debug(f"检测到 {len(detections)} 个目标")

        except Exception as e:
            logger.error(f"处理图片失败 {image_path}: {e}")
            skipped_count += 1
            continue

    # 创建data.yaml
    yaml_path = None
    if not dry_run and processed_count > 0:
        yaml_path = create_data_yaml(dataset_structure.output_dir, class_names)

        # 清理空文件
        cleaned_images, cleaned_labels = dataset_structure.clear_empty_files()

        if cleaned_images > 0 or cleaned_labels > 0:
            logger.info(f"清理了 {cleaned_images} 张无标签图片和 {cleaned_labels} 个空标签文件")

    # 获取最终统计
    stats = {}
    if not dry_run:
        stats = dataset_structure.get_stats()

    # 输出结果
    logger.info("=" * 60)
    logger.info("自动标注完成")
    logger.info(f"处理成功: {processed_count} 张图片")
    logger.info(f"跳过: {skipped_count} 张图片")
    logger.info(f"总检测数: {total_detections} 个目标")

    if not dry_run and processed_count > 0:
        logger.info(f"输出目录: {dataset_structure.output_dir}")
        logger.info(f"图片数量: {stats.get('image_count', 0)}")
        logger.info(f"标签数量: {stats.get('label_count', 0)}")

    logger.info("=" * 60)

    return {
        "success": True,
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "total_detections": total_detections,
        "output_dir": str(dataset_structure.output_dir) if not dry_run else None,
        "yaml_path": str(yaml_path) if yaml_path else None,
        "stats": stats,
        "model_info": model_info,
    }

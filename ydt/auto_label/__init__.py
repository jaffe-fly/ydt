"""
YDT自动标注模块

提供基于YOLO模型的自动标注功能，支持OBB和BBox两种格式。
"""

from .core import auto_label_dataset
from .model_utils import ModelPredictor
from .structure import DatasetStructure

__all__ = [
    "auto_label_dataset",
    "ModelPredictor",
    "DatasetStructure"
]
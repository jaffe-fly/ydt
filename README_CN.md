<div align="center">

### 🎯 YDT - YOLO数据集工具

[![Python版本](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)[![许可证](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)[![代码风格: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)[![类型检查](https://img.shields.io/badge/type--checked-mypy-informational.svg)](https://mypy.readthedocs.io/)

[English](README.md) | [简体中文](README_CN.md)

---


</div>

#### 特性

- 自动检测和处理 OBB（9个值：`class_id x1 y1 x2 y2 x3 y3 x4 y4`）和 BBox（5个值：`class_id x_center y_center width height`）两种格式
- 基于 SAHI 的智能切片，支持水平/网格模式和可配置重叠率
- 旋转增强with自动 OBB 坐标变换
- 多方法 resize（scale & crop），支持自定义插值（linear/lanczos4），支持单图和数据集
- 基于坐标的精确裁剪
- 支持从模型推理或数据集标签裁剪对象，支持padding和尺寸过滤
- 视频切帧支持并行处理
- 智能训练/验证集划分，类别平衡
- 多数据集合并
- 按类别 ID 提取数据，支持标签过滤和 ID 重映射
- 合成数据集生成，可配置每张图物体数量、旋转范围和类别均衡采样
- YOLO 自动标注，支持 BBox/OBB 格式

**可视化**
- 交互式数据集浏览with键盘控制（n/p/q）
- 类别过滤和 letterbox 预览
- 增强效果预览

#### 安装

```bash
pip install yolodt
```

#### 使用方法

```bash
ydt --help

usage: ydt [-h] [--version] [-v]
           {slice,augment,video,crop-coords,resize,concat,split,merge,extract,synthesize,auto-label,analyze,visualize,viz-letterbox}
           ...

YOLO Dataset Tools - Process and manage YOLO format datasets

positional arguments:
  {slice,augment,video,crop-coords,crop,resize,concat,split,merge,extract,synthesize,auto-label,analyze,visualize,viz-letterbox}
                        Available commands
    slice               切片大图为小块
    augment             数据增强（旋转）
    video               从视频提取帧
    crop-coords         按坐标裁剪图片
    crop                从模型或数据集标签裁剪对象
    resize              调整图片或YOLO数据集大小
    concat              拼接两张图片
    split               划分训练/验证集
    merge               合并多个数据集
    extract             提取类别、图片或标签
    synthesize          生成合成数据集
    auto-label          使用YOLO模型自动标注
    analyze             分析数据集统计信息
    visualize           交互式可视化YOLO数据集
    viz-letterbox       可视化letterbox变换

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -v, --verbose         Verbose output
```

#### 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO框架
- [SAHI](https://github.com/obss/sahi) - 切片辅助超级推理
- [Albumentations](https://github.com/albumentations-team/albumentations) - 图像增强

---


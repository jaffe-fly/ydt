<div align="center">

# 🎯 YDT - YOLO数据集工具

**专业的YOLO数据集处理工具包**

[![Python版本](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![许可证](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![代码风格: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![类型检查](https://img.shields.io/badge/type--checked-mypy-informational.svg)](https://mypy.readthedocs.io/)

[English](README.md) | [简体中文](README_CN.md)

---

**智能 • 强大 • 易用**

</div>

## ✨ 特性

<table>
<tr>
<td width="50%">

### 🖼️ **图像处理**
- 📐 SAHI智能切片
- 🔄 旋转增强（OBB坐标转换）
- 📏 多方法缩放（scale & crop）
- 📍 坐标精确裁剪
- 🎨 HSV颜色增强
- 🎬 视频切帧
- 🚀 多线程视频处理

</td>
<td width="50%">

### 📊 **数据集操作**
- ✂️ 智能训练/验证集划分
- 🔗 多数据集合并
- 🎲 合成数据生成
- 🤖 YOLO自动标注
- 📦 批量处理

</td>
</tr>
<tr>
<td width="50%">

### 👁️ **可视化**
- 🖼️ 交互式数据集浏览
- ⌨️ 键盘控制 (n/p/q)
- 🎯 类别过滤
- 📸 Letterbox预览
- 🎨 增强效果预览

</td>
<td width="50%">

### 🎯 **格式支持**
- 📐 OBB（旋转边界框）
- 📦 标准BBox
- 🔄 自动格式检测
- ✨ 无缝格式转换

</td>
</tr>
</table>

## 🚀 快速开始

```bash
# 克隆并安装
git clone https://github.com/yourusername/ydt.git
cd ydt/package
uv venv && uv pip install -e .

# 试试看！
ydt image slice -i ./images -o ./output
ydt image slice -i ./images -o ./output -c 3 -d 2
ydt viz dataset -i ./dataset
```

## 💻 使用方法

### 命令行

```bash
# 图像处理
ydt image slice -i ./imgs -o ./out -c 3
ydt image slice -i ./imgs -o ./out -c 3 -d 2 -r 0.1 --overlap-vertical 0.05
ydt image resize -i ./images -o ./resized -s 640 800 1024
ydt image concat img1.jpg img2.jpg -o output.jpg -d horizontal -a center
ydt image augment -i data.yaml -o ./aug
ydt image video -i ./videos -o ./frames -s 30
ydt image video -i ./videos -o ./frames --parallel -w 4
ydt image crop-coords -i ./images -o ./cropped -c "100,50,600,400"

# 数据集操作
ydt dataset split -i data.yaml -o ./split -r 0.8
ydt dataset merge -i ./ds1 ./ds2 -o ./merged
ydt dataset synthesize -t ./targets -b ./backgrounds -o ./synthetic --objects-per-image 2-5 --split train --train-ratio 0.8
ydt dataset auto-label -i ./images -m ./yolo11n.pt --format bbox -o ./labeled

# 可视化
ydt viz dataset -i ./dataset
ydt viz letterbox -i ./image.jpg
ydt viz augment -i ./image.jpg
```

### Python API

```python
from ydt.image import (
    slice_dataset,
    augment_dataset,
    extract_frames,
    process_images_multi_method,
    concat_images_horizontally,
    concat_images_vertically
)
from ydt.dataset import split_dataset, DatasetSynthesizer, auto_label_dataset
from ydt.visual import visualize_dataset, visualize_letterbox

# 切片大图
slice_dataset("./dataset", "./sliced", horizontal_count=3)

# 网格切片（2×3 = 6块）
slice_dataset("./dataset", "./sliced", horizontal_count=2, vertical_count=3)

# 多方法缩放图像
process_images_multi_method("./images", "./resized", target_sizes=[640, 800, 1024])

# 拼接图像
concat_images_horizontally("img1.jpg", "img2.jpg", "output.jpg", alignment="center")

# 视频切帧
extract_frames("./videos", "./frames", step=30)

# 划分数据集
split_dataset("./data.yaml", "./split", train_ratio=0.8)

# 自动标注图像
result = auto_label_dataset(
    input_dir="./images",
    model_path="./yolo11n.pt",
    format_type="bbox",
    output_dir="./labeled"
)

# 生成合成数据集
synthesizer = DatasetSynthesizer(
    target_dir="./targets",
    background_dir="./backgrounds",
    output_dir="./synthetic",
    objects_per_image=(2, 5),  # 每张图2-5个物体
    split_mode="trainval",      # 生成训练+验证集
    train_ratio=0.8            # 80%训练，20%验证
)
stats = synthesizer.synthesize_dataset(num_images=1000)

# 可视化数据集
visualize_dataset("./dataset", scan_train=True)
visualize_letterbox("./image.jpg", output_dir="./output")
```

## 📦 模块结构

```
ydt/
├── 🖼️  image/       # 图像处理
├── 📊  dataset/     # 数据集操作
├── 👁️  visual/      # 可视化
├── 🛠️  core/        # 核心工具
├── 🤖  auto_label/  # 自动标注
└── ⚡  cli/         # 命令行接口
```

## 🎯 核心功能

### 双格式支持

自动检测并处理两种格式：

| 格式 | 值数量 | 描述 |
|------|--------|------|
| **OBB** | 9个值 | `class_id x1 y1 x2 y2 x3 y3 x4 y4` |
| **BBox** | 5个值 | `class_id x_center y_center width height` |

### 智能切片

基于SAHI的智能切片，支持水平和网格切片，并保持标注准确性。

```bash
# 水平切片（默认）
ydt image slice -i ./images -o ./sliced -c 3 -r 0.1

# 网格切片（3×2 = 6块）
ydt image slice -i ./images -o ./sliced -c 3 -d 2 -r 0.1 --overlap-vertical 0.05

# 精细网格切片
ydt image slice -i ./images -o ./sliced -c 4 -d 3 -r 0.05 --overlap-vertical 0.02
```

### 视频切帧

从视频文件提取帧用于数据集创建，支持顺序和并行处理。

```bash
# 顺序处理（默认）
ydt image video -i ./videos -o ./frames -s 30

# 并行处理多个视频
ydt image video -i ./videos -o ./frames --parallel -w 4
```

**特性：**
- 🎯 智能工作线程数自动检测
- ⚡ 并发视频解码
- 📊 每个视频的进度跟踪
- 🔄 单视频自动回退到顺序处理

### 自动标注

使用YOLO模型自动标注图像，支持BBox和OBB格式：

```bash
ydt dataset auto-label -i ./images -m ./yolo11n.pt --format bbox -o ./labeled
```

**特性：**
- 🎯 支持BBox和OBB格式
- 🤖 自动格式检测和转换
- 📁 清晰的输出目录结构
- ⚙️ 可配置的置信度和IOU阈值
- 🔍 使用`--dry-run`预览模式

### 交互式可视化

使用键盘控制浏览数据集：
- `n` - 下一张
- `p` - 上一张
- `q` - 退出

```bash
ydt viz dataset -i ./dataset
```

## 📚 文档

详细文档位于 [docs](docs/) 文件夹：

| 文档 | 说明 |
|------|------|
| 📖 [安装指南](docs/installation.md) | 详细安装说明 |
| 🎯 [使用教程](docs/usage.md) | 使用示例和教程 |
| 📘 [API参考](docs/api-reference.md) | 完整API文档 |
| 📦 [发布指南](docs/publishing.md) | 打包发布步骤 |

## 🛠️ 依赖要求

- Python >= 3.8
- OpenCV >= 4.5.0
- Ultralytics >= 8.0.0
- SAHI >= 0.11.0

完整依赖列表见 [pyproject.toml](pyproject.toml)。

## 🤝 贡献

欢迎贡献！请查看[贡献指南](CONTRIBUTING.md)。

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 开启 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO框架
- [SAHI](https://github.com/obss/sahi) - 切片辅助超级推理
- [Albumentations](https://github.com/albumentations-team/albumentations) - 图像增强

## 📮 联系方式

- 📧 邮箱：your.email@example.com
- 🐛 问题反馈：[GitHub Issues](https://github.com/yourusername/ydt/issues)

---

<div align="center">

**用 ❤️ 为计算机视觉社区打造**

⭐ **给我们点个Star吧 — 这对我们是很大的鼓励！**

[⬆ 回到顶部](#-ydt---yolo数据集工具)

</div>

# GeoTIFF_GeoJSON

## 项目概述

这是一个简化的端到端AI系统，用于从遥感影像（GeoTIFF）自动提取道路与路口矢量数据（GeoJSON）。基于多模态大模型，实现遥感影像到结构化地理数据的自动生成。

## 核心功能

### 1. 多模态特征提取
- 对GeoTIFF影像进行patch切分
- 利用**DINOv3**提取密集视觉特征
- 作为道路/路口结构的语义表征

### 2. 大模型序列生成
- 基于Qwen2.5构建序列到序列模型
- 将视觉特征解码为结构化的GeoJSON矢量要素
- 支持道路中心线和路口几何的生成

### 3. 地理空间处理
- 实现地理坐标系（CRS84）与影像坐标系的双向投影变换
- 确保标注与影像的精确对齐
- 使用rasterio和pyproj进行坐标转换

### 4. 训练约束机制
- 引入mask（人工审核区）约束训练与评估
- 仅在有效区域采样，保证结果可靠性
- 支持tile-based训练，避免无效区域干扰

## 项目结构

```
simplified_geomapgen/
├── scripts/               # 启动脚本（Python）
│   ├── train.py           # 训练脚本
│   ├── infer.py           # 推理脚本
│   └── preprocess.py      # 数据预处理脚本
├── unimapgen/             # 核心模块
│   ├── data/              # 数据处理
│   │   ├── dataset.py     # 数据集类
│   │   └── tokenizer.py   # 序列化tokenizer
│   ├── models/            # 模型定义
│   │   └── qwen_geo_generator.py  # Qwen地理生成器
│   └── geo/               # 地理空间处理
│       ├── io.py          # 影像和矢量IO
│       └── geometry.py    # 几何变换
├── configs/               # 配置文件
│   └── default.yaml       # 默认配置
├── requirements.txt       # 依赖列表
└── README.md              # 项目说明
```

## 快速开始

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 数据准备
- 准备GeoTIFF影像文件
- 准备对应的GeoJSON标注文件（Lane.geojson, Intersection.geojson）
- 准备mask文件（审核区域）

### 3. 数据预处理
```bash
python scripts/preprocess.py
```

### 4. 训练模型
```bash
python scripts/train.py
```

### 5. 执行推理
```bash
python scripts/infer.py
```
**支持批量处理**：将测试影像放在 `data.image_root` 目录下，脚本会自动处理所有.tif/.tiff文件

## 技术栈

- **深度学习框架**: PyTorch, Transformers
- **多模态模型**: Qwen2.5-VL, **DINOv3**
- **地理空间库**: rasterio, pyproj, shapely
- **配置管理**: PyYAML

## 关键特性

- **端到端流程**: 从原始影像到最终GeoJSON的一站式处理
- **高精度对齐**: 精确的坐标系变换确保地理准确性
- **约束训练**: mask机制保证只在有效区域学习
- **可扩展性**: 支持不同分辨率和区域的遥感数据

## 注意事项

- 确保CUDA环境与PyTorch版本匹配
- 数据路径需在配置文件中正确设置
- 训练前验证mask文件与影像的对齐

# GeoTIFF_GeoJSON:基于多模态大模型的遥感影像自动提取系统

## 项目简介
本项目基于DINO+Qwen设计了一套遥感影像自动提取系统，用于从卫星影像 GeoTIFF 文件中生成 GeoJSON 文件。

- 本地 DINO 负责提取视觉特征
- 本地 Qwen 负责生成几何 token 并还原成 geojson 文件

## 目录结构

```text
GeoTIFF_GeoJSON/
  configs/default.yaml
  scripts/
    check.py
    extract_features.py
    extract_tokens.py
    train.py
    infer.py
    evaluate.py
  unimapgen/
    data/
    geo/
    models/
    utils/
```

## 数据集目录：

```text
data/
  train/
    images/
    masks/
    labels/
    feature/
    token/
  val/
    images/
    masks/
    labels/
    feature/
    token/
  test/
    images/
    labels/
    feature/
```

其中：

- `images/` 保存原始的遥感卫星影像（.tif/.tiff）
- `labels/` 保存卫星影像对应的 geojson 文件（.geojson）
- `masks/` 保存卫星影像对应的人工mask区域（可选）
- `feature/<image_stem>.pt` 保存该影像全部 patch 的 DINO 特征
- `token/<image_stem>.pt` 保存该影像全部 patch 的 GeoJSON token

## 环境依赖
请先安装Anaconda/Miniconda，推荐Python 3.12+。

```bash
conda create -n dino_qwen python=3.12
conda activate dino_qwen
pip install -r requirement.txt
```

## 项目流程

### 1. 配置config/default.yaml

```text
config/default.yaml中已注释各个参数的含义
```

### 2. 检查目录、权重、样本和 patch 配置

```bash
python scripts/check.py
```

### 3. 提取 train / val / test 的 DINO 特征

```bash
python scripts/extract_features.py
```

### 4. 提取 train / val 的 GeoJSON token

```bash
python scripts/extract_tokens.py
```

### 5. 训练 Qwen

```bash
python scripts/train.py
```

### 6. 测试集推理

```bash
python scripts/infer.py
```

### 7. 用测试集真值评估

```bash
python scripts/evaluate.py

```

## 致谢

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [Qwen3](https://github.com/QwenLM/Qwen3)

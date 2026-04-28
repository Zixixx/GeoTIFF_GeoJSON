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
    preprocess.py
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

推荐数据目录：

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

### 2. 检查目录、样本和 patch 配置
```bash
python scripts/preprocess.py
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

## 默认训练配置

```yaml
model:
  qwen_tuning_mode: lora
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_ensure_weight_tying: true
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
  lora_modules_to_save:
    - embed_tokens
    - lm_head
  embedding_mean_resizing: false
  attn_implementation: flash_attention_2

training:
  batch_size: 1
  val_batch_size: 1
  feature_batch_size: 1
  dataset_cache_files: 1
  optimizer: adamw
  mixed_precision: auto
  gradient_checkpointing: true
  save_every_n_steps: 0
```

## checkpoint 保存策略

- 每个 epoch 结束后，如果 `val_loss` 刷新最好值，覆盖保存一次 `best_model`
- 所有 epoch 完成后保存一次 `final_model`

另外支持按 step 保存：

```yaml
training:
  save_every_n_steps: 0
```

- `0`：关闭
- `>0`：每若干个 optimizer step 覆盖保存一次 `checkpoints/latest_step`

## 主要依赖

- `PyTorch`：用于模型训练、推理以及混合精度、多卡相关能力。
- `transformers`：用于加载和调用 Qwen 基座模型。
- `peft`：用于 LoRA 微调、适配器保存与加载。
- `safetensors`：用于加载本地 Qwen 权重分片。
- `rasterio`：用于读取 GeoTIFF/TIFF 影像和地理变换信息。
- `shapely`：用于 GeoJSON 几何裁剪、合并和距离计算。
- `numpy`：用于数值计算与特征张量处理。
- `pyyaml`：用于读取 `configs/default.yaml` 配置文件。
- `affine`：用于像素坐标和地理坐标之间的仿射变换。

## 致谢

- 感谢 Qwen 系列模型及其开源生态，为几何序列生成提供了稳定的语言模型基础。
- 感谢 DINO 相关工作，为遥感影像视觉特征提取提供了强大的视觉 backbone。

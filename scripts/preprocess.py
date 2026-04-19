#!/usr/bin/env python3
"""
数据预处理脚本：遥感影像自动提取系统数据预处理
"""

import os
import sys
import yaml
from pathlib import Path
from glob import glob

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unimapgen.geo.io import GeoIO
from unimapgen.data.dataset import GeoMapDataset

def main():
    print("Starting data preprocessing...")

    # 加载配置
    config_path = project_root / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    geo_io = GeoIO()

    # 数据路径
    image_root = Path(config['data']['image_root'])
    label_root = Path(config['data']['label_root'])
    mask_root = Path(config['data'].get('mask_root', ''))

    # 检查目录是否存在
    if not image_root.exists():
        print(f"Warning: Image root directory does not exist: {image_root}")
        return
    if not label_root.exists():
        print(f"Warning: Label root directory does not exist: {label_root}")
        return

    # 查找所有影像和标注文件
    image_files = list(image_root.glob("*.tif")) + list(image_root.glob("*.tiff"))
    label_files = list(label_root.glob("*.geojson"))
    mask_files = list(mask_root.glob("*.tif")) if mask_root.exists() else []

    print(f"Found {len(image_files)} image files")
    print(f"Found {len(label_files)} label files")
    print(f"Found {len(mask_files)} mask files")

    # 验证数据对齐
    validation_results = []
    for i, (img_path, label_path) in enumerate(zip(image_files, label_files)):
        try:
            is_aligned = geo_io.validate_alignment(str(img_path), str(label_path))
            validation_results.append((img_path.name, is_aligned))
            print(f"Validated {i+1}/{len(image_files)}: {img_path.name} - {'OK' if is_aligned else 'Warning'}")
        except Exception as e:
            print(f"Error validating {img_path.name}: {e}")
            validation_results.append((img_path.name, False))

    # 创建数据集验证
    try:
        dataset = GeoMapDataset(
            image_paths=[str(p) for p in image_files[:1]],  # 测试前1个
            label_paths=[str(p) for p in label_files[:1]],
            mask_paths=[str(p) for p in mask_files[:1]] if mask_files else None,
            tile_size=config['data']['tile_size_px']
        )

        # 测试数据加载
        sample = dataset[0]
        print(f"Dataset sample keys: {sample.keys()}")
        print(f"Number of patches: {len(sample['patches'])}")

    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # 保存预处理结果
    preprocess_info = {
        "image_count": len(image_files),
        "label_count": len(label_files),
        "mask_count": len(mask_files),
        "validation_results": validation_results,
        "tile_size": config['data']['tile_size_px'],
        "min_mask_ratio": config['data']['min_mask_ratio']
    }

    output_path = project_root / "preprocess_info.json"
    with open(output_path, 'w') as f:
        import json
        json.dump(preprocess_info, f, indent=2)

    print(f"Data preprocessing completed! Info saved to {output_path}")
    print(f"Validation summary: {sum(1 for _, ok in validation_results if ok)}/{len(validation_results)} files OK")

if __name__ == "__main__":
    main()
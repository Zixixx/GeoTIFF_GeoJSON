#!/usr/bin/env python3
"""
离线提取 DINO patch 特征。

当前脚本会对 train / val / test 三个 split 按影像逐张提取特征，
并按文件名保存到：
- data/train/feature/<image_stem>.pt
- data/val/feature/<image_stem>.pt
- data/test/feature/<image_stem>.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unimapgen.utils.utils import (
    configure_runtime,
    extract_and_cache_split_features,
    feature_root_for_split,
    load_config,
)


def main():
    print("开始提取 DINO 特征...")
    config = load_config()
    device = configure_runtime(config)
    print(f"使用设备: {device}")

    train_feature_size = extract_and_cache_split_features(config, "train", device)
    val_feature_size = extract_and_cache_split_features(config, "val", device)
    test_feature_size = extract_and_cache_split_features(config, "test", device)
    if not (train_feature_size == val_feature_size == test_feature_size):
        raise RuntimeError(
            "train/val/test 特征维度不一致: "
            f"train={train_feature_size}, val={val_feature_size}, test={test_feature_size}"
        )

    print("DINO 特征提取完成。")
    print(f"train 特征目录: {feature_root_for_split(config, 'train')}")
    print(f"val 特征目录: {feature_root_for_split(config, 'val')}")
    print(f"test 特征目录: {feature_root_for_split(config, 'test')}")
    print(f"特征维度: {train_feature_size}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
离线提取 train/val patch GeoJSON 的 token 序列。

当前脚本会对 train / val 两个 split 按影像逐张提取 token，并按文件名保存：
- data/train/token/<image_stem>.pt
- data/val/token/<image_stem>.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unimapgen.utils.utils import build_tokenizer, extract_and_cache_split_tokens, load_config, token_root_for_split


def main():
    print("开始提取离线 token...")
    config = load_config()
    tokenizer = build_tokenizer(config)

    train_patch_count = extract_and_cache_split_tokens(config, "train", tokenizer)
    val_patch_count = extract_and_cache_split_tokens(config, "val", tokenizer)

    print("离线 token 提取完成。")
    print(f"train token 目录: {token_root_for_split(config, 'train')}")
    print(f"val token 目录: {token_root_for_split(config, 'val')}")
    print(f"train patch token 数: {train_patch_count}")
    print(f"val patch token 数: {val_patch_count}")


if __name__ == "__main__":
    main()

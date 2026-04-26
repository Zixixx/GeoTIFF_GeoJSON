#!/usr/bin/env python3
"""
数据检查脚本。

脚本会读取 data.train、data.val、data.test 三个 split，检查影像、
标签和掩码文件是否齐全，并预览 train/val 的 patch 切分结果。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unimapgen.data.dataset import GeoMapDataset
from unimapgen.geo.io import GeoIO
from unimapgen.utils.utils import (
    collect_image_files,
    collect_masks,
    find_same_stem_raster,
    load_config,
    split_config,
)


def inspect_split(split_name, config, geo_io):
    """检查单个 split，并返回统计结果。"""
    image_root, label_root, mask_root, _, _ = split_config(config, split_name)
    if split_name not in {"train", "val"}:
        mask_root = None

    if image_root is None or not image_root.exists():
        raise FileNotFoundError(f"{split_name} 影像目录不存在: {image_root}")

    image_files = collect_image_files(image_root)
    label_files = sorted(label_root.glob("*.geojson")) if label_root and label_root.exists() else []
    mask_files = collect_masks(mask_root) if split_name in {"train", "val"} else []
    print(f"[{split_name}] images={len(image_files)}, labels={len(label_files)}, masks={len(mask_files)}")

    validation_results = []
    if label_root and label_root.exists():
        for image_path in image_files:
            label_path = label_root / f"{image_path.stem}.geojson"
            if not label_path.exists():
                validation_results.append((image_path.name, False, "missing_label"))
                print(f"[{split_name}] 缺少标签: {image_path.name}")
                continue
            try:
                aligned = geo_io.validate_alignment(str(image_path), str(label_path))
                validation_results.append((image_path.name, bool(aligned), "aligned" if aligned else "crs_mismatch"))
            except Exception as exc:
                validation_results.append((image_path.name, False, str(exc)))
                print(f"[{split_name}] 校验失败 {image_path.name}: {exc}")

    # train/val 需要预览带标签的数据集，test 可能没有标签。
    if split_name in {"train", "val"} and image_files and label_root and label_root.exists():
        first_pair = next(
            (
                (image_path, label_root / f"{image_path.stem}.geojson")
                for image_path in image_files
                if (label_root / f"{image_path.stem}.geojson").exists()
            ),
            None,
        )
        if first_pair:
            preview_mask = find_same_stem_raster(mask_root, first_pair[0].stem)
            dataset = GeoMapDataset(
                image_paths=[str(first_pair[0])],
                label_paths=[str(first_pair[1])],
                mask_paths=[str(preview_mask)] if preview_mask else None,
                tile_size=config["data"]["tile_size_px"],
                overlap=config["data"].get("overlap_px", 0),
                min_mask_ratio=config["data"].get("min_mask_ratio", 0.02),
            )
            print(f"[{split_name}] 预览 patch 数: {len(dataset)}")

    return {
        "image_count": len(image_files),
        "label_count": len(label_files),
        "mask_count": len(mask_files),
        "validation_results": validation_results,
    }


def main():
    print("开始数据检查...")
    config = load_config()

    geo_io = GeoIO()
    results = {}
    for split_name in ("train", "val", "test"):
        split_cfg = config["data"].get(split_name)
        if not isinstance(split_cfg, dict):
            raise ValueError(f"配置缺少 data.{split_name}")
        results[split_name] = inspect_split(split_name, config, geo_io)

    output_path = project_root / "preprocess_info.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"数据检查完成，结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

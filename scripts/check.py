#!/usr/bin/env python3
"""
启动前检查脚本。

脚本会检查：
1. Qwen 模型目录、DINO 权重文件、DINO 仓库目录是否存在；
2. train / val / test 三个 split 的影像、标签、掩码文件是否齐全；
3. train / val 的 patch 切分是否能正常生成。
"""

from __future__ import annotations

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
    resolve_path,
    split_config,
)


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def inspect_model_assets(config) -> bool:
    """检查模型相关本地路径是否存在。"""
    model_cfg = config.get("model", {})
    qwen_root = resolve_path(model_cfg.get("name"))
    dino_weight = resolve_path(model_cfg.get("backbone"))
    dino_repo = resolve_path(model_cfg.get("dino_repo_or_dir"))

    all_ok = True

    print_header("模型路径检查")

    if qwen_root and qwen_root.is_dir():
        print(f"[OK] Qwen 模型目录: {qwen_root}")
    else:
        print(f"[FAIL] Qwen 模型目录不存在: {qwen_root}")
        all_ok = False

    if dino_weight and dino_weight.is_file():
        print(f"[OK] DINO 权重文件: {dino_weight}")
    else:
        print(f"[FAIL] DINO 权重文件不存在: {dino_weight}")
        all_ok = False

    if dino_repo and dino_repo.is_dir():
        print(f"[OK] DINO 仓库目录: {dino_repo}")
        hubconf = dino_repo / "hubconf.py"
        if hubconf.is_file():
            print(f"[OK] DINO hubconf.py: {hubconf}")
        else:
            print(f"[FAIL] DINO 仓库目录缺少 hubconf.py: {hubconf}")
            all_ok = False
    else:
        print(f"[FAIL] DINO 仓库目录不存在: {dino_repo}")
        all_ok = False

    dino_model_name = model_cfg.get("dino_model_name")
    if dino_model_name:
        print(f"[OK] DINO 模型名: {dino_model_name}")
    else:
        print("[FAIL] model.dino_model_name 未填写")
        all_ok = False

    return all_ok


def inspect_split(split_name, config, geo_io):
    """检查单个 split，并打印统计结果。"""
    image_root, label_root, mask_root, _, _ = split_config(config, split_name)
    if split_name not in {"train", "val"}:
        mask_root = None

    if image_root is None or not image_root.exists():
        raise FileNotFoundError(f"{split_name} 影像目录不存在: {image_root}")

    print_header(f"{split_name} 数据检查")

    image_files = collect_image_files(image_root)
    label_files = sorted(label_root.glob("*.geojson")) if label_root and label_root.exists() else []
    mask_files = collect_masks(mask_root) if split_name in {"train", "val"} else []
    print(f"[{split_name}] images={len(image_files)}, labels={len(label_files)}, masks={len(mask_files)}")

    missing_label_count = 0
    aligned_count = 0
    failed_alignment_count = 0

    if label_root and label_root.exists():
        for image_path in image_files:
            label_path = label_root / f"{image_path.stem}.geojson"
            if not label_path.exists():
                missing_label_count += 1
                print(f"[{split_name}] 缺少标签: {image_path.name}")
                continue
            try:
                aligned = geo_io.validate_alignment(str(image_path), str(label_path))
                if aligned:
                    aligned_count += 1
                else:
                    failed_alignment_count += 1
                    print(f"[{split_name}] 坐标未对齐: {image_path.name}")
            except Exception as exc:
                failed_alignment_count += 1
                print(f"[{split_name}] 校验失败 {image_path.name}: {exc}")

    print(
        f"[{split_name}] 标签检查总结: aligned={aligned_count}, "
        f"missing_label={missing_label_count}, alignment_failed={failed_alignment_count}"
    )

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


def main():
    print("开始执行项目预检查...")
    config = load_config()

    model_ok = inspect_model_assets(config)

    geo_io = GeoIO()
    for split_name in ("train", "val", "test"):
        split_cfg = config["data"].get(split_name)
        if not isinstance(split_cfg, dict):
            raise ValueError(f"配置缺少 data.{split_name}")
        inspect_split(split_name, config, geo_io)

    print_header("检查完成")
    if model_ok:
        print("模型路径检查通过，数据集检查已输出到屏幕。")
    else:
        print("模型路径检查未完全通过，请先修正上述 FAIL 项后再继续。")


if __name__ == "__main__":
    main()

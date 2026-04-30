#!/usr/bin/env python3
"""
测试集推理脚本。

流程为：
1. 先运行 scripts/extract_features.py，为 test 数据集提取离线 DINO 特征
2. 本脚本只读取测试影像、patch offset 和对应的 visual_features
3. 使用 Qwen 生成几何 token，再恢复成整图像素坐标或地理坐标
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unimapgen.data.tokenizer import GeoTokenizer
from unimapgen.geo.io import GeoIO
from unimapgen.models.qwen_geo_generator import QwenGeoGenerator
from unimapgen.utils.utils import (
    collect_image_files,
    create_test_patches,
    feature_path_for_image,
    load_config,
    resolve_path,
)


def decode_patch_prediction(tokenizer, generated_ids, patch_offset, transform, geo_processor):
    """解码单个 patch 的预测结果，并恢复到整图坐标或地理坐标。"""
    patch_geojson = tokenizer.decode(generated_ids)
    features = []
    for feature in patch_geojson.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "LineString":
            continue

        transformed_coords = []
        for x, y in geometry.get("coordinates", []):
            pixel_x = x + patch_offset[0]
            pixel_y = y + patch_offset[1]
            if transform is not None:
                geo_x, geo_y = geo_processor.pixel_to_geo([(pixel_x, pixel_y)], transform)[0]
                transformed_coords.append([geo_x, geo_y])
            else:
                transformed_coords.append([pixel_x, pixel_y])

        if len(transformed_coords) >= 2:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": transformed_coords},
                    "properties": feature.get("properties", {}),
                }
            )
    return features


def merge_patch_results(features):
    """删除完全重复的 LineString，避免重叠窗口带来的重复输出。"""
    dedup = []
    seen = set()
    for feature in features:
        key = tuple(tuple(coord) for coord in feature["geometry"]["coordinates"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(feature)
    return {"type": "FeatureCollection", "features": dedup}


def load_tokenizer(checkpoint_config):
    """根据 checkpoint 中保存的 tokenizer 元数据重建 GeoTokenizer。"""
    tokenizer_cfg = checkpoint_config.get("tokenizer", {})
    training_cfg = checkpoint_config.get("training_config", {})
    model_cfg = training_cfg.get("model", {})
    data_cfg = training_cfg.get("data", {})

    max_features = tokenizer_cfg.get("max_features", model_cfg.get("max_features"))
    max_points = tokenizer_cfg.get("max_points", model_cfg.get("max_points_per_feature"))
    max_coord = tokenizer_cfg.get("max_coord")
    if max_coord is None:
        tile_size = data_cfg.get("tile_size_px")
        if tile_size is None:
            raise KeyError("checkpoint config 缺少 tokenizer.max_coord，且 training_config.data.tile_size_px 不存在。")
        max_coord = int(tile_size) - 1

    if max_features is None or max_points is None:
        raise KeyError("checkpoint config 缺少 tokenizer 或 training_config 中的关键字段，无法重建 GeoTokenizer。")

    return GeoTokenizer(
        max_features=int(max_features),
        max_points=int(max_points),
        max_coord=int(max_coord),
    )


def test_image_root(config):
    """读取测试集影像目录配置。"""
    test_cfg = config["data"].get("test")
    if not isinstance(test_cfg, dict) or not test_cfg.get("image_root"):
        raise ValueError("配置缺少 data.test.image_root。")
    return resolve_path(test_cfg["image_root"])


def resolve_qwen_override(config, checkpoint_config):
    """按训练模式决定推理时是否需要外部 Qwen 基座目录。"""
    mode = str(checkpoint_config.get("qwen_tuning_mode", "lora")).lower()
    runtime_qwen_root = config.get("model", {}).get("name")

    if mode == "full":
        print("检测到 full 模式 checkpoint，将优先使用 checkpoint 内的 qwen_finetuned。")
        return None

    if mode in {"lora", "frozen"}:
        if runtime_qwen_root:
            print(f"检测到 {mode} 模式 checkpoint，将使用 default.yaml 中的 model.name 作为 Qwen 基座目录。")
        else:
            print(f"检测到 {mode} 模式 checkpoint，但 default.yaml 未填写 model.name；将仅尝试使用 checkpoint 中记录的路径。")
        return runtime_qwen_root

    print(f"未识别的 qwen_tuning_mode={mode}，将按默认配置尝试加载。")
    return runtime_qwen_root


def main():
    print("开始测试集推理...")
    config = load_config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    checkpoint_path = resolve_path(config["inference"].get("checkpoint_path"))
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")

    with open(checkpoint_path / "config.json", encoding="utf-8") as file:
        checkpoint_config = json.load(file)

    tokenizer = load_tokenizer(checkpoint_config)
    qwen_model_path_override = resolve_qwen_override(config, checkpoint_config)
    model = QwenGeoGenerator.from_pretrained(
        str(checkpoint_path),
        qwen_model_path_override=qwen_model_path_override,
    ).to(device)
    model.eval()

    image_dir = test_image_root(config)
    if not image_dir.exists():
        raise FileNotFoundError(f"测试集影像目录不存在: {image_dir}")

    output_dir = resolve_path(config["inference"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = collect_image_files(image_dir)
    if not test_images:
        print(f"测试集影像目录中没有 tif/tiff 文件: {image_dir}")
        return

    geo_io = GeoIO()
    geo_processor = geo_io.processor
    tile_size = int(config["data"]["tile_size_px"])
    overlap = int(config["data"].get("overlap_px", 0))
    infer_batch_size = int(config["inference"].get("batch_size", 1))
    default_max_length = 2 * tokenizer.max_points * tokenizer.max_features + 2
    infer_max_length = int(config["inference"].get("max_length", default_max_length))
    generate_log_interval = int(config["inference"].get("generate_log_interval", 0))

    for image_path in test_images:
        print(f"处理测试影像: {image_path.name}")
        image, _, transform, _ = geo_io.load_image_with_geo(str(image_path))
        patches = create_test_patches(image, tile_size=tile_size, overlap=overlap)
        print(
            f"{image_path.name}: patches={len(patches)}, batch_size={infer_batch_size}, max_length={infer_max_length}",
            flush=True,
        )

        feature_file = feature_path_for_image(config, "test", image_path.stem)
        if not feature_file.exists():
            raise FileNotFoundError(
                f"未找到测试集feature文件: {feature_file}。请先运行 scripts/extract_features.py。"
            )
        cached_features = torch.load(feature_file, map_location="cpu")
        if int(cached_features.shape[0]) != len(patches):
            raise RuntimeError(
                f"测试影像 patch 数与特征数不一致: image={image_path.name}, "
                f"patches={len(patches)}, features={cached_features.shape[0]}"
            )

        all_features = []
        for start in range(0, len(patches), infer_batch_size):
            batch_patches = patches[start : start + infer_batch_size]
            batch_features = cached_features[start : start + infer_batch_size].to(device)
            batch_end = start + len(batch_patches)
            print(f"[{image_path.stem}] 开始生成 patch {start + 1}-{batch_end}/{len(patches)}", flush=True)
            with torch.no_grad():
                generated_ids = model.generate(
                    visual_features=batch_features,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=infer_max_length,
                    log_interval=generate_log_interval,
                    log_prefix=f"[{image_path.stem} {start + 1}-{batch_end}] ",
                )
            print(f"[{image_path.stem}] 完成生成 patch {start + 1}-{batch_end}/{len(patches)}", flush=True)

            for local_idx, patch in enumerate(batch_patches):
                all_features.extend(
                    decode_patch_prediction(
                        tokenizer,
                        generated_ids[local_idx].tolist(),
                        patch["offset"],
                        transform,
                        geo_processor,
                    )
                )

        output_path = output_dir / f"{image_path.stem}_result.geojson"
        geo_io.save_geojson(merge_patch_results(all_features), str(output_path))
        print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()

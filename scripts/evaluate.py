#!/usr/bin/env python3
"""
测试集矢量结果评估脚本。

脚本读取 data.test.label_root 中的真值 GeoJSON，以及 inference/evaluation
输出目录中的预测 GeoJSON。评估前会把真值和预测统一转换到影像像素坐标，
然后基于线几何 Hausdorff 距离做一对一匹配，并计算 Precision、Recall 和 F1。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import rasterio
from affine import Affine
from shapely.geometry import LineString, MultiLineString, shape

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unimapgen.geo.geometry import GeoProcessor
from unimapgen.utils.utils import collect_image_files, load_config, resolve_path

geo_processor = GeoProcessor()


def is_identity_transform(transform) -> bool:
    """判断影像 transform 是否等同于单位矩阵。"""
    return transform == Affine.identity()


def auto_coordinate_mode(transform) -> str:
    """auto 模式：有非单位 transform 时认为坐标是地理坐标，否则认为是像素坐标。"""
    return "pixel" if transform is None or is_identity_transform(transform) else "geo"


def to_pixel_coords(coords: Iterable, transform, coordinate_mode: str) -> List[Tuple[float, float]]:
    """把一串坐标转换为像素坐标，供距离计算使用。"""
    mode = auto_coordinate_mode(transform) if coordinate_mode == "auto" else coordinate_mode

    # 真值或预测如果是地理坐标，需要用 GeoTIFF transform 的逆变换转回影像像素坐标。
    pixels = []
    for x, y in coords:
        if mode == "geo" and transform is not None:
            px, py = geo_processor.geo_to_pixel([(x, y)], transform)[0]
        else:
            px, py = float(x), float(y)
        pixels.append((px, py))
    return pixels


def geometry_to_lines(geometry, transform, coordinate_mode: str) -> List[LineString]:
    """把 GeoJSON geometry 转成像素坐标 LineString 列表。"""
    if not geometry:
        return []

    geom = shape(geometry)
    raw_lines = []
    if isinstance(geom, LineString):
        raw_lines = [list(geom.coords)]
    elif isinstance(geom, MultiLineString):
        raw_lines = [list(line.coords) for line in geom.geoms]
    elif hasattr(geom, "geoms"):
        for sub_geom in geom.geoms:
            if isinstance(sub_geom, LineString):
                raw_lines.append(list(sub_geom.coords))

    lines = []
    for coords in raw_lines:
        pixel_coords = to_pixel_coords(coords, transform, coordinate_mode)
        if len(pixel_coords) >= 2:
            line = LineString(pixel_coords)
            if not line.is_empty and line.length > 0:
                lines.append(line)
    return lines


def load_geojson_lines(path: Path, transform, coordinate_mode: str) -> List[LineString]:
    """读取 GeoJSON 文件中的线几何，并统一转换到像素坐标。"""
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    lines = []
    for feature in data.get("features", []):
        lines.extend(geometry_to_lines(feature.get("geometry"), transform, coordinate_mode))
    return lines


def greedy_match(pred_lines: List[LineString], gt_lines: List[LineString], threshold_px: float):
    """基于 Hausdorff 距离做一对一贪心匹配。"""
    candidates = []
    for pred_idx, pred in enumerate(pred_lines):
        for gt_idx, gt in enumerate(gt_lines):
            distance = float(pred.hausdorff_distance(gt))
            if math.isfinite(distance):
                candidates.append((distance, pred_idx, gt_idx))
    candidates.sort(key=lambda item: item[0])

    matched_preds = set()
    matched_gts = set()
    matches = []
    for distance, pred_idx, gt_idx in candidates:
        # 距离超过阈值的线对不计为匹配；阈值单位是像素。
        if distance > threshold_px:
            break
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)
        matches.append({"pred_idx": pred_idx, "gt_idx": gt_idx, "hausdorff_px": distance})
    return matches


def compute_metrics(pred_lines: List[LineString], gt_lines: List[LineString], threshold_px: float) -> Dict:
    """计算单张影像的 Precision、Recall、F1 和匹配线段的平均 Hausdorff 距离。"""
    matches = greedy_match(pred_lines, gt_lines, threshold_px)
    tp = len(matches)
    fp = max(0, len(pred_lines) - tp)
    fn = max(0, len(gt_lines) - tp)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall <= 1e-12 else 2.0 * precision * recall / (precision + recall)

    matched_distances = [item["hausdorff_px"] for item in matches]
    mean_hausdorff = sum(matched_distances) / len(matched_distances) if matched_distances else None
    return {
        "pred_count": len(pred_lines),
        "gt_count": len(gt_lines),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_hausdorff_px": mean_hausdorff,
        "matches": matches,
    }


def prediction_path_for(prediction_dir: Path, stem: str) -> Path:
    """按 infer.py 的命名规则查找预测 GeoJSON。"""
    return prediction_dir / f"{stem}_result.geojson"


def main():
    print("开始评估测试集预测结果...")
    with open(project_root / "configs" / "default.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    test_cfg = config["data"].get("test", {})
    image_root = resolve_path(test_cfg.get("image_root"))
    label_root = resolve_path(test_cfg.get("label_root"))
    if image_root is None or not image_root.exists():
        raise FileNotFoundError(f"测试集影像目录不存在: {image_root}")
    if label_root is None or not label_root.exists():
        raise FileNotFoundError(f"测试集标签目录不存在: {label_root}")

    eval_cfg = config.get("evaluation", {})
    prediction_dir = resolve_path(eval_cfg.get("prediction_dir") or config["inference"]["output_dir"])
    output_path = resolve_path(eval_cfg.get("output_path") or "./outputs/test_metrics.json")
    threshold_px = float(eval_cfg.get("distance_threshold_px", 5.0))
    label_mode = str(eval_cfg.get("label_coordinate_mode", "pixel"))
    pred_mode = str(eval_cfg.get("prediction_coordinate_mode", "auto"))

    sample_metrics = []
    total_tp = total_fp = total_fn = 0
    all_matched_distances = []

    for image_path in collect_image_files(image_root):
        # 每张影像都用自己的 transform，因为不同 GeoTIFF 的地理位置可能不同。
        with rasterio.open(image_path) as src:
            transform = src.transform

        gt_path = label_root / f"{image_path.stem}.geojson"
        pred_path = prediction_path_for(prediction_dir, image_path.stem)

        # 评估前统一转成像素坐标：你的真值通常是 pixel，infer 输出通常是 geo/auto。
        gt_lines = load_geojson_lines(gt_path, transform, label_mode)
        pred_lines = load_geojson_lines(pred_path, transform, pred_mode)
        metrics = compute_metrics(pred_lines, gt_lines, threshold_px)
        metrics.update(
            {
                "image": image_path.name,
                "gt_path": str(gt_path),
                "prediction_path": str(pred_path),
                "label_coordinate_mode": label_mode,
                "prediction_coordinate_mode": pred_mode,
            }
        )
        sample_metrics.append(metrics)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        all_matched_distances.extend(item["hausdorff_px"] for item in metrics["matches"])
        print(
            f"{image_path.name}: P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, "
            f"TP/FP/FN={metrics['tp']}/{metrics['fp']}/{metrics['fn']}"
        )

    # micro 指标：先汇总全部测试图的 TP/FP/FN，再统一计算 Precision/Recall/F1。
    micro_precision = total_tp / max(1, total_tp + total_fp)
    micro_recall = total_tp / max(1, total_tp + total_fn)
    micro_f1 = 0.0 if micro_precision + micro_recall <= 1e-12 else 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
    result = {
        "threshold_px": threshold_px,
        "micro": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "mean_matched_hausdorff_px": (
                sum(all_matched_distances) / len(all_matched_distances) if all_matched_distances else None
            ),
        },
        "samples": sample_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"评估完成，结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

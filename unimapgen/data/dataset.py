from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import rasterio
import torch
from shapely.affinity import translate
from shapely.geometry import box, shape
from torch.utils.data import Dataset

from .patching import generate_patch_windows


class GeoMapDataset(Dataset):
    """
    面向“图像 patch -> 几何序列”训练的数据集。

    标签默认是整图像素坐标 GeoJSON。可选 mask 只决定哪些 patch 被保留。
    feature_paths 用于加载离线提取的视觉特征。
    token_paths 用于加载离线编码好的几何 token。
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        label_paths: Sequence[str],
        mask_paths: Optional[Sequence[str]] = None,
        feature_paths: Optional[Sequence[str]] = None,
        token_paths: Optional[Sequence[str]] = None,
        tile_size: int = 896,
        overlap: int = 0,
        tokenizer=None,
        min_mask_ratio: float = 0.02,
        cache_file_limit: int = 1,
    ):
        self.image_paths = list(image_paths)
        self.label_paths = list(label_paths)
        self.mask_paths = list(mask_paths) if mask_paths else None
        self.feature_paths = list(feature_paths) if feature_paths else None
        self.token_paths = list(token_paths) if token_paths else None
        self.tile_size = tile_size
        self.overlap = overlap
        self.tokenizer = tokenizer
        self.min_mask_ratio = min_mask_ratio
        self.cache_file_limit = max(0, int(cache_file_limit))

        self.samples: List[Dict] = []
        self._feature_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._token_cache: "OrderedDict[str, List[List[int]]]" = OrderedDict()
        self._prepare_samples()

    def _prepare_samples(self) -> None:
        """读取每组影像/标签，并展开成 patch 样本。"""
        for idx, image_path in enumerate(self.image_paths):
            with rasterio.open(image_path) as src:
                image = self._read_rgb(src)
                transform = src.transform
                crs = src.crs

            with open(self.label_paths[idx], encoding="utf-8") as file:
                labels = json.load(file)

            mask = None
            if self.mask_paths and idx < len(self.mask_paths):
                mask_path = self.mask_paths[idx]
                try:
                    with rasterio.open(mask_path) as src:
                        mask = src.read(1)
                    if mask.shape != image.shape[1:]:
                        raise ValueError(
                            f"Mask shape {mask.shape} does not match image shape {image.shape[1:]}: {mask_path}"
                        )
                except Exception:
                    mask = None

            feature_path = None
            if self.feature_paths and idx < len(self.feature_paths):
                feature_path = self.feature_paths[idx]

            token_path = None
            if self.token_paths and idx < len(self.token_paths):
                token_path = self.token_paths[idx]

            patches = self._tile_image(image, mask, self.tile_size, self.overlap)
            for patch_idx, patch_data in enumerate(patches):
                patch_labels = self._extract_patch_labels(
                    labels=labels,
                    offset=patch_data["offset"],
                    patch_size=(patch_data["width"], patch_data["height"]),
                )
                sample = {
                    "image": patch_data["image"],
                    "labels": patch_labels,
                    "transform": transform,
                    "crs": crs,
                    "patch_offset": patch_data["offset"],
                }
                if feature_path:
                    sample["feature_path"] = feature_path
                    sample["feature_index"] = patch_idx
                if token_path:
                    sample["token_path"] = token_path
                    sample["token_index"] = patch_idx
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cached_tensor(self, cache: "OrderedDict[str, torch.Tensor]", path: str) -> torch.Tensor:
        if path in cache:
            value = cache.pop(path)
            cache[path] = value
            return value
        value = torch.load(path, map_location="cpu")
        cache[path] = value
        while self.cache_file_limit and len(cache) > self.cache_file_limit:
            cache.popitem(last=False)
        if self.cache_file_limit == 0:
            cache.pop(path, None)
        return value

    def _get_cached_tokens(self, path: str) -> List[List[int]]:
        if path in self._token_cache:
            value = self._token_cache.pop(path)
            self._token_cache[path] = value
            return value
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            sequences = payload.get("encoded_sequences")
        else:
            sequences = payload
        if sequences is None:
            raise ValueError(f"Invalid token cache file: {path}")
        self._token_cache[path] = sequences
        while self.cache_file_limit and len(self._token_cache) > self.cache_file_limit:
            self._token_cache.popitem(last=False)
        if self.cache_file_limit == 0:
            self._token_cache.pop(path, None)
        return sequences

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image_tensor = torch.from_numpy(self._normalize_image(sample["image"])).float()

        item = {
            "image": image_tensor,
            "labels_geojson": sample["labels"],
            "patch_offset": sample["patch_offset"],
        }
        if "feature_path" in sample:
            feature_path = str(sample["feature_path"])
            feature_bank = self._get_cached_tensor(self._feature_cache, feature_path)
            item["visual_features"] = feature_bank[int(sample["feature_index"])]
        elif "visual_features" in sample:
            item["visual_features"] = sample["visual_features"]

        if "token_path" in sample:
            token_path = str(sample["token_path"])
            encoded_sequences = self._get_cached_tokens(token_path)
            encoded_labels = encoded_sequences[int(sample["token_index"])]
        elif self.tokenizer:
            encoded_labels = self.tokenizer.encode(sample["labels"])
        else:
            encoded_labels = None

        if encoded_labels is not None:
            input_ids = torch.tensor(encoded_labels[:-1], dtype=torch.long)
            labels = torch.tensor(encoded_labels[1:], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            item["input_ids"] = input_ids
            item["labels"] = labels
            item["attention_mask"] = attention_mask

        return item

    @staticmethod
    def _read_rgb(src) -> np.ndarray:
        """读取最多三个波段，并把单/双波段影像适配成 RGB。"""
        if src.count >= 3:
            band_indices = [1, 2, 3]
        elif src.count == 1:
            band_indices = [1]
        else:
            band_indices = list(range(1, src.count + 1))

        image = src.read(band_indices).astype(np.float32)
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        elif image.shape[0] == 2:
            image = np.concatenate([image, image[:1]], axis=0)
        return image[:3]

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        """归一化常见栅格取值范围，同时保留已是浮点影像的语义。"""
        image = image.astype(np.float32, copy=False)
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return np.zeros_like(image, dtype=np.float32)

        max_value = float(finite.max())
        min_value = float(finite.min())
        if max_value <= 1.0 and min_value >= 0.0:
            return np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
        if max_value <= 255.0:
            return (np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0) / 255.0).clip(0.0, 1.0)
        if max_value <= 65535.0:
            return (np.nan_to_num(image, nan=0.0, posinf=65535.0, neginf=0.0) / 65535.0).clip(0.0, 1.0)

        p1, p99 = np.percentile(finite, [1, 99])
        denom = max(float(p99 - p1), 1e-6)
        return ((np.nan_to_num(image, nan=float(p1)) - float(p1)) / denom).clip(0.0, 1.0)

    @staticmethod
    def tile_image_array(
        image: np.ndarray,
        mask: Optional[np.ndarray],
        tile_size: int,
        overlap: int = 0,
        min_mask_ratio: float = 0.02,
    ) -> List[Dict]:
        """创建固定尺寸 padding patch，并可按 mask 覆盖率筛选。"""
        h, w = image.shape[1:]
        patches: List[Dict] = []

        for window in generate_patch_windows(h, w, tile_size, overlap):
            x, y = window["offset"]
            patch = image[:, y : y + tile_size, x : x + tile_size]
            patch_h = patch.shape[1]
            patch_w = patch.shape[2]

            padded_patch = np.zeros((image.shape[0], tile_size, tile_size), dtype=image.dtype)
            padded_patch[:, :patch_h, :patch_w] = patch

            patch_info = {
                "image": padded_patch,
                "offset": (x, y),
                "width": patch_w,
                "height": patch_h,
            }

            if mask is not None:
                patch_mask = mask[y : y + tile_size, x : x + tile_size]
                if np.mean(patch_mask > 0) >= min_mask_ratio:
                    patches.append(patch_info)
            else:
                patches.append(patch_info)

        return patches

    def _tile_image(self, image: np.ndarray, mask: Optional[np.ndarray], tile_size: int, overlap: int = 0) -> List[Dict]:
        """创建固定尺寸 padding patch，并可按 mask 覆盖率筛选。"""
        return self.tile_image_array(
            image=image,
            mask=mask,
            tile_size=tile_size,
            overlap=overlap,
            min_mask_ratio=self.min_mask_ratio,
        )

    def _extract_patch_labels(self, labels: Dict, offset, patch_size) -> Dict:
        """把整图像素 GeoJSON 裁剪到一个 patch，并平移坐标。"""
        x_offset, y_offset = offset
        patch_width, patch_height = patch_size
        patch_bounds = box(x_offset, y_offset, x_offset + patch_width, y_offset + patch_height)

        patch_features = []
        for feature in labels.get("features", []):
            geometry = feature.get("geometry")
            if not geometry:
                continue

            geom = shape(geometry)
            if geom.is_empty or not geom.intersects(patch_bounds):
                continue

            clipped = geom.intersection(patch_bounds)
            if clipped.is_empty:
                continue

            clipped = translate(clipped, xoff=-x_offset, yoff=-y_offset)
            patch_features.append(
                {
                    "type": "Feature",
                    "geometry": clipped.__geo_interface__,
                    "properties": feature.get("properties", {}),
                }
            )

        return {"type": "FeatureCollection", "features": patch_features}

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """为 DataLoader batch 填充不同长度的 token 序列。"""
        images = torch.stack([item["image"] for item in batch], dim=0)
        result = {"image": images}

        if "input_ids" not in batch[0]:
            return result

        pad_token_id = 0
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )

        result["input_ids"] = input_ids
        result["labels"] = labels
        result["attention_mask"] = attention_mask
        result["labels_geojson"] = [item["labels_geojson"] for item in batch]
        result["patch_offset"] = [item["patch_offset"] for item in batch]
        if "visual_features" in batch[0]:
            result["visual_features"] = torch.stack([item["visual_features"] for item in batch], dim=0)
        return result

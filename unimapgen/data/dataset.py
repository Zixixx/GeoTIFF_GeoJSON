import torch
from torch.utils.data import Dataset
import rasterio
import json
from pathlib import Path
from shapely.geometry import shape
import numpy as np

class GeoMapDataset(Dataset):
    """地理地图数据集类"""

    def __init__(self, image_paths, label_paths, mask_paths=None, tile_size=896, tokenizer=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mask_paths = mask_paths
        self.tile_size = tile_size
        self.tokenizer = tokenizer

        # 收集所有训练样本
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        """准备训练样本"""
        for idx in range(len(self.image_paths)):
            # 读取影像
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read([1, 2, 3])  # RGB bands
                transform = src.transform
                crs = src.crs

            # 读取标注
            with open(self.label_paths[idx]) as f:
                labels = json.load(f)

            # 读取mask（如果有）
            mask = None
            if self.mask_paths and idx < len(self.mask_paths):
                try:
                    with rasterio.open(self.mask_paths[idx]) as src:
                        mask = src.read(1)
                except:
                    mask = None

            # 切patch并创建训练样本
            patches = self._tile_image(image, mask, self.tile_size)

            for patch_data in patches:
                sample = {
                    'image': patch_data['image'],
                    'labels': labels,  # 可以使用patch内的标签或者全局标签
                    'transform': transform,
                    'crs': crs,
                    'patch_offset': patch_data['offset']
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 如果有tokenizer，编码标签
        if self.tokenizer:
            # 这里可以根据需要编码地理标签
            encoded_labels = self.tokenizer.encode(sample['labels'])
            sample['input_ids'] = encoded_labels
            sample['attention_mask'] = [1] * len(encoded_labels)

        return sample

    def _tile_image(self, image, mask, tile_size):
        """切patch并返回patch信息"""
        h, w = image.shape[1:]
        patches = []

        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                patch = image[:, y:y+tile_size, x:x+tile_size]
                patch_info = {
                    'image': patch,
                    'offset': (x, y)
                }

                if mask is not None:
                    patch_mask = mask[y:y+tile_size, x:x+tile_size]
                    # 只保留有效区域
                    if np.mean(patch_mask > 0) > 0.02:  # min_mask_ratio
                        patches.append(patch_info)
                else:
                    patches.append(patch_info)

        return patches
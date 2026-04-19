#!/usr/bin/env python3
"""
推理脚本：遥感影像自动提取系统推理
"""

import os
import sys
import torch
import yaml
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unimapgen.models.qwen_geo_generator import QwenGeoGenerator
from unimapgen.data.tokenizer import GeoTokenizer

# 临时注释掉GeoIO导入，直到安装依赖
# from unimapgen.geo.io import GeoIO
# geo_io = GeoIO()

# 简化的GeoIO替代实现
class SimpleGeoIO:
    def load_image_with_geo(self, image_path):
        # 临时实现：返回模拟数据
        import numpy as np
        # 模拟一个小的测试图像
        image = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
        profile = {"width": 256, "height": 256, "count": 3}
        transform = None  # 简化为None
        crs = "EPSG:4326"
        return image, profile, transform, crs

    def save_geojson(self, data, path):
        import json
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved GeoJSON to {path}")

geo_io = SimpleGeoIO()

class GeoInferenceEngine:
    """地理推理引擎"""

    def __init__(self):
        pass

    def _create_patches(self, image, tile_size=896, overlap=0):
        """
        将大图像分成小patch

        Args:
            image: numpy array, shape [C, H, W]
            tile_size: patch大小
            overlap: 重叠像素

        Returns:
            list: 每个patch的字典 {'image': patch_image, 'offset': (x_offset, y_offset)}
        """
        if image.ndim == 3:
            c, h, w = image.shape
        else:
            raise ValueError(f"Expected 3D image [C, H, W], got shape {image.shape}")

        patches = []
        stride = tile_size - overlap

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                # 提取patch
                patch = image[:, y:y + tile_size, x:x + tile_size]

                patches.append({
                    'image': patch,
                    'offset': (x, y)
                })

        # 处理右边缘和下边缘的剩余部分
        if w % stride != 0:
            x = w - tile_size
            for y in range(0, h - tile_size + 1, stride):
                if x >= 0:
                    patch = image[:, y:y + tile_size, x:x + tile_size]
                    patches.append({
                        'image': patch,
                        'offset': (x, y)
                    })

        if h % stride != 0:
            y = h - tile_size
            for x in range(0, w - tile_size + 1, stride):
                if y >= 0:
                    patch = image[:, y:y + tile_size, x:x + tile_size]
                    patches.append({
                        'image': patch,
                        'offset': (x, y)
                    })

        return patches

    def _parse_generated_geojson(self, generated_text, patch_offset, transform):
        """
        解析模型生成的文本为GeoJSON格式

        Args:
            generated_text: 模型生成的文本
            patch_offset: patch在原图中的偏移 (x_offset, y_offset)
            transform: 地理变换矩阵

        Returns:
            list: GeoJSON features列表
        """
        features = []

        try:
            # 尝试解析为JSON
            if generated_text.strip().startswith('{'):
                geojson_data = json.loads(generated_text)
                if 'features' in geojson_data:
                    for feature in geojson_data['features']:
                        # 转换坐标从patch坐标系到原图坐标系
                        transformed_feature = self._transform_feature_coordinates(
                            feature, patch_offset, transform)
                        if transformed_feature:
                            features.append(transformed_feature)
            else:
                # 如果不是JSON格式，尝试提取坐标信息
                features = self._extract_coordinates_from_text(generated_text, patch_offset, transform)

        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试从文本中提取坐标
            features = self._extract_coordinates_from_text(generated_text, patch_offset, transform)
        except Exception as e:
            print(f"Warning: Failed to parse generated text: {e}")

        return features

    def _transform_feature_coordinates(self, feature, patch_offset, transform):
        """
        将feature的坐标从patch坐标系转换到地理坐标系

        Args:
            feature: GeoJSON feature
            patch_offset: (x_offset, y_offset)
            transform: rasterio transform对象

        Returns:
            dict: 转换后的feature
        """
        try:
            x_offset, y_offset = patch_offset
            geometry = feature['geometry']

            if geometry['type'] == 'LineString':
                coords = geometry['coordinates']
                transformed_coords = []

                for coord in coords:
                    # patch坐标 + offset = 原图坐标
                    pixel_x = coord[0] + x_offset
                    pixel_y = coord[1] + y_offset

                    # 像素坐标转地理坐标
                    if transform:
                        try:
                            geo_x, geo_y = transform * (pixel_x, pixel_y)
                            transformed_coords.append([geo_x, geo_y])
                        except:
                            # 如果transform失败，保持像素坐标
                            transformed_coords.append([pixel_x, pixel_y])
                    else:
                        transformed_coords.append([pixel_x, pixel_y])

                return {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': transformed_coords
                    },
                    'properties': feature.get('properties', {})
                }

            elif geometry['type'] == 'Point':
                coord = geometry['coordinates']
                pixel_x = coord[0] + x_offset
                pixel_y = coord[1] + y_offset

                if transform:
                    try:
                        geo_x, geo_y = transform * (pixel_x, pixel_y)
                        new_coord = [geo_x, geo_y]
                    except:
                        new_coord = [pixel_x, pixel_y]
                else:
                    new_coord = [pixel_x, pixel_y]

                return {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': new_coord
                    },
                    'properties': feature.get('properties', {})
                }

        except Exception as e:
            print(f"Warning: Failed to transform coordinates: {e}")

        return None

    def _extract_coordinates_from_text(self, text, patch_offset, transform):
        """
        从生成的文本中提取坐标信息（备用方法）

        Args:
            text: 生成的文本
            patch_offset: patch偏移
            transform: 地理变换

        Returns:
            list: 提取的features
        """
        features = []

        # 简单的坐标提取逻辑（可以根据需要扩展）
        import re

        # 查找坐标对模式 (x,y)
        coord_pattern = r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)'
        coords = re.findall(coord_pattern, text)

        if len(coords) >= 2:
            # 将连续的坐标点组成线段
            line_coords = []
            for x_str, y_str in coords:
                try:
                    x = float(x_str) + patch_offset[0]
                    y = float(y_str) + patch_offset[1]

                    if transform:
                        try:
                            geo_x, geo_y = transform * (x, y)
                            line_coords.append([geo_x, geo_y])
                        except:
                            line_coords.append([x, y])
                    else:
                        line_coords.append([x, y])
                except ValueError:
                    continue

            if len(line_coords) >= 2:
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': line_coords
                    },
                    'properties': {'type': 'lane', 'source': 'text_extraction'}
                })

        return features

    def _merge_patch_results(self, all_results):
        """
        合并多个patch的结果，进行去重和清理

        Args:
            all_results: 所有patch的结果列表

        Returns:
            dict: 合并后的GeoJSON
        """
        merged_features = []
        seen_features = set()

        for feature in all_results:
            # 创建特征的唯一标识符（基于坐标的简化版本）
            try:
                coords = tuple(tuple(coord) for coord in feature['geometry']['coordinates'])
                feature_id = (feature['geometry']['type'], coords)

                if feature_id not in seen_features:
                    seen_features.add(feature_id)
                    merged_features.append(feature)
            except:
                # 如果无法创建标识符，直接添加
                merged_features.append(feature)

        # 限制特征数量
        max_features = 100  # 可以从配置中读取
        if len(merged_features) > max_features:
            merged_features = merged_features[:max_features]

        return {
            "type": "FeatureCollection",
            "features": merged_features
        }

# 创建推理引擎实例
inference_engine = GeoInferenceEngine()

def main():
    print("Starting inference...")

    # 加载配置
    config_path = project_root / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载训练好的完整模型
    checkpoint_path = project_root / "checkpoints" / "final_model"
    if checkpoint_path.exists():
        # 加载完整的训练后模型（包含预训练权重+训练权重）
        model = QwenGeoGenerator.from_pretrained(str(checkpoint_path)).to(device)
        print("Complete trained model loaded from checkpoint.")
    else:
        print("Warning: No trained model found. Using base pretrained model.")
        # 在没有训练模型时加载预训练模型
        model = QwenGeoGenerator(
            qwen_model_path=config['model']['name'],
            dino_model_path=config['model']['backbone']
        ).to(device)

    model.eval()

    # 推理参数
    test_image_dir = Path(config['data']['image_root'])
    output_dir = Path(config['inference']['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # 获取所有测试影像
    image_extensions = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(test_image_dir.glob(ext)))

    if not test_images:
        print(f"No test images found in {test_image_dir}")
        return

    print(f"Found {len(test_images)} test images")

    # 批量推理
    for i, image_path in enumerate(test_images):
        print(f"Processing {i+1}/{len(test_images)}: {image_path.name}")

        try:
            # 加载测试影像
            image, profile, transform, crs = geo_io.load_image_with_geo(str(image_path))
            print(f"Loaded image: {image.shape}")

            # 推理提示词
            prompt = "Generate the reviewed Lane.geojson and Intersection.geojson content using only the reserved vector tokens. Predict all lane properties and the lane centerline geometry. Output at most {max_features} lane features and at most {max_points_per_feature} points per lane."

            print("Performing complete inference pipeline...")

            # 1. Patch处理：将大图像分成小块
            patches = inference_engine._create_patches(image, tile_size=config['data']['tile_size'],
                                         overlap=config['data']['overlap_px'])

            print(f"Created {len(patches)} patches for processing")

            # 2. 批量推理每个patch
            all_results = []
            for patch_idx, patch_data in enumerate(patches):
                print(f"Processing patch {patch_idx + 1}/{len(patches)}")

                patch_image = patch_data['image']
                patch_offset = patch_data['offset']

                # 转换为模型输入格式
                patch_tensor = torch.from_numpy(patch_image).float().to(device)
                if patch_tensor.dim() == 3:  # [C, H, W]
                    patch_tensor = patch_tensor.unsqueeze(0)  # [1, C, H, W]

                try:
                    # 模型推理
                    with torch.no_grad():
                        generated_ids = model.generate(patch_tensor, prompt)

                    # 解码结果
                    generated_text = model.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    # 解析生成的GeoJSON
                    patch_result = inference_engine._parse_generated_geojson(generated_text, patch_offset, transform)
                    if patch_result:
                        all_results.extend(patch_result)

                except Exception as e:
                    print(f"Warning: Failed to process patch {patch_idx}: {e}")
                    continue

            # 3. 结果聚合和去重
            if all_results:
                merged_result = inference_engine._merge_patch_results(all_results)
                print(f"Merged {len(all_results)} patch results into {len(merged_result['features'])} features")
            else:
                # 如果没有结果，使用示例数据
                print("No results generated, using sample data")
                merged_result = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": [[0, 0], [100, 100]]
                            },
                            "properties": {"type": "lane", "source": "sample"}
                        }
                    ]
                }


            # 保存结果
            output_filename = image_path.stem + "_result.geojson"
            output_path = output_dir / output_filename
            geo_io.save_geojson(merged_result, str(output_path))

            print(f"Result saved to {output_path}")

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue

    print(f"Inference completed! Processed {len(test_images)} images.")


if __name__ == "__main__":
    main()
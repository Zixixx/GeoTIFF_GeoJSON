import json
from pathlib import Path
from .geometry import GeoProcessor

class GeoIO:
    """地理数据IO类"""

    def __init__(self):
        self.processor = GeoProcessor()

    def load_geojson(self, path):
        """加载GeoJSON文件"""
        with open(path) as f:
            data = json.load(f)
        return data

    def save_geojson(self, data, path):
        """保存GeoJSON文件"""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_image_with_geo(self, image_path):
        """加载影像及其地理信息"""
        return self.processor.load_geotiff(image_path)

    def validate_alignment(self, image_path, geojson_path):
        """验证影像与标注的对齐"""
        image, profile, transform, crs = self.load_image_with_geo(image_path)
        geojson = self.load_geojson(geojson_path)

        # 检查CRS是否匹配
        if geojson.get('crs', {}).get('properties', {}).get('name') != str(crs):
            print(f"Warning: CRS mismatch between image ({crs}) and GeoJSON")

        return True
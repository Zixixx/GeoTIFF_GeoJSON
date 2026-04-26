from __future__ import annotations

import json

from .geometry import GeoProcessor


class GeoIO:
    """用于读取和验证栅格/矢量地理数据的辅助类。"""

    def __init__(self):
        self.processor = GeoProcessor()

    def load_geojson(self, path):
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    def save_geojson(self, data, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def load_image_with_geo(self, image_path):
        return self.processor.load_geotiff(image_path)

    def validate_alignment(self, image_path, geojson_path):
        """
        验证栅格 CRS 和 GeoJSON CRS 元数据是否基本一致。

        返回：
            当 CRS 一致或 GeoJSON 省略 CRS 元数据时返回 True。
        """
        _, _, _, crs = self.load_image_with_geo(image_path)
        geojson = self.load_geojson(geojson_path)

        geojson_crs = geojson.get("crs", {}).get("properties", {}).get("name")
        if geojson_crs is None:
            return True

        if str(geojson_crs) != str(crs):
            print(f"Warning: CRS mismatch between image ({crs}) and GeoJSON ({geojson_crs})")
            return False

        return True

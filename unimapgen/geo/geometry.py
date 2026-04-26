from __future__ import annotations

import json

import rasterio
from shapely.geometry import mapping, shape


class GeoProcessor:
    """用于影像读取和像素/地理坐标转换的辅助工具。"""

    def __init__(self, src_crs="EPSG:4326", dst_crs=None):
        self.src_crs = src_crs
        self.dst_crs = dst_crs

    def pixel_to_geo(self, pixel_coords, transform):
        """把像素坐标 [(x, y), ...] 转成 GeoTIFF transform 对应的地理坐标。"""
        geo_coords = []
        for x, y in pixel_coords:
            geo_x, geo_y = transform * (float(x), float(y))
            geo_coords.append((geo_x, geo_y))
        return geo_coords

    def geo_to_pixel(self, geo_coords, transform):
        """把地理坐标 [(x, y), ...] 反算成影像像素坐标。"""
        inv_transform = ~transform
        pixel_coords = []
        for x, y in geo_coords:
            pixel_x, pixel_y = inv_transform * (float(x), float(y))
            pixel_coords.append((pixel_x, pixel_y))
        return pixel_coords

    def transform_geometry(self, geometry, transform, direction="pixel_to_geo"):
        """在像素坐标和地理坐标之间转换简单 shapely 几何或 GeoJSON geometry。"""
        geom = shape(geometry) if isinstance(geometry, dict) else geometry

        if not hasattr(geom, "coords"):
            raise ValueError("Only simple coordinate geometries are supported.")

        coords = list(geom.coords)
        if direction == "pixel_to_geo":
            transformed_coords = self.pixel_to_geo(coords, transform)
        elif direction == "geo_to_pixel":
            transformed_coords = self.geo_to_pixel(coords, transform)
        else:
            raise ValueError(f"Unsupported direction: {direction}")

        transformed_geometry = type(geom)(transformed_coords)
        if isinstance(geometry, dict):
            return mapping(transformed_geometry)
        return transformed_geometry

    def load_geotiff(self, path):
        """读取 GeoTIFF 影像数组及其 profile、transform 和 CRS。"""
        with rasterio.open(path) as src:
            image = src.read()
            profile = src.profile
            transform = src.transform
            crs = src.crs
        return image, profile, transform, crs

    def save_geojson(self, features, path):
        """把 feature 列表保存为 GeoJSON FeatureCollection。"""
        geojson = {"type": "FeatureCollection", "features": features}
        with open(path, "w", encoding="utf-8") as file:
            json.dump(geojson, file, indent=2, ensure_ascii=False)

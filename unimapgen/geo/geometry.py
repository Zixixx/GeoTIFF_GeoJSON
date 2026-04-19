import rasterio
from pyproj import Transformer
from shapely.geometry import shape
import numpy as np

class GeoProcessor:
    """地理空间处理类"""

    def __init__(self, src_crs='EPSG:4326', dst_crs=None):
        self.src_crs = src_crs
        self.dst_crs = dst_crs
        self.transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True) if dst_crs else None

    def pixel_to_geo(self, pixel_coords, transform):
        """像素坐标转地理坐标"""
        geo_coords = []
        for x, y in pixel_coords:
            lon, lat = rasterio.transform.xy(transform, y, x)
            geo_coords.append((lon, lat))
        return geo_coords

    def geo_to_pixel(self, geo_coords, transform):
        """地理坐标转像素坐标"""
        pixel_coords = []
        for lon, lat in geo_coords:
            x, y = rasterio.transform.rowcol(transform, lon, lat)
            pixel_coords.append((x, y))
        return pixel_coords

    def transform_geometry(self, geometry, direction='pixel_to_geo'):
        """变换几何对象"""
        if direction == 'pixel_to_geo':
            coords = list(geometry.coords)
            transformed_coords = self.pixel_to_geo(coords, self.transform)
            return type(geometry)(transformed_coords)
        else:
            coords = list(geometry.coords)
            transformed_coords = self.geo_to_pixel(coords, self.transform)
            return type(geometry)(transformed_coords)

    def load_geotiff(self, path):
        """加载GeoTIFF"""
        with rasterio.open(path) as src:
            image = src.read()
            profile = src.profile
            transform = src.transform
            crs = src.crs
        return image, profile, transform, crs

    def save_geojson(self, features, path):
        """保存GeoJSON"""
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        with open(path, 'w') as f:
            json.dump(geojson, f)
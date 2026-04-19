import json
from shapely.geometry import shape
import numpy as np

class GeoTokenizer:
    """地理数据tokenizer"""

    def __init__(self, max_features=50, max_points=100):
        self.max_features = max_features
        self.max_points = max_points

    def encode(self, geojson_data):
        """将GeoJSON编码为序列"""
        tokens = []

        # 处理Lane
        if 'features' in geojson_data:
            for feature in geojson_data['features'][:self.max_features]:
                geom = shape(feature['geometry'])
                coords = list(geom.coords)[:self.max_points]

                # 简化为坐标序列
                for coord in coords:
                    tokens.extend([coord[0], coord[1]])

                tokens.append('[SEP]')  # 分隔符

        return tokens

    def decode(self, tokens):
        """将序列解码为GeoJSON"""
        # 将坐标对解析为线段
        features = []
        coords = []

        i = 0
        while i < len(tokens):
            if tokens[i] == '[SEP]':
                if coords:
                    features.append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'LineString',
                            'coordinates': coords
                        },
                        'properties': {}
                    })
                    coords = []
                i += 1
            else:
                coords.append([tokens[i], tokens[i+1]])
                i += 2

        return {
            'type': 'FeatureCollection',
            'features': features
        }
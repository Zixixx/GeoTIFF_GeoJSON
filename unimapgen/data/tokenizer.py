from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from shapely.geometry import LineString, Point, shape


class GeoTokenizer:
    """
    把 GeoJSON 几何转换为离散 token 序列。

    tokenizer 刻意保持简单：坐标变成整数 id，feature 边界用特殊 token 表示。
    """

    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    SEP_TOKEN = "[SEP]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, max_features: int = 50, max_points: int = 100, max_coord: int = 895):
        self.max_features = max_features
        self.max_points = max_points
        self.max_coord = max(1, int(max_coord))

        self.special_tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.SEP_TOKEN,
            self.UNK_TOKEN,
        ]
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.sep_token_id = 3
        self.unk_token_id = 4
        self.coord_token_offset = len(self.special_tokens)
        self.vocab_size = self.coord_token_offset + self.max_coord + 1

    def encode(self, geojson_data: Optional[Dict]) -> List[int]:
        """把 GeoJSON FeatureCollection 编码成离散 token id。"""
        tokens = [self.bos_token_id]

        if not geojson_data or "features" not in geojson_data:
            tokens.append(self.eos_token_id)
            return tokens

        feature_count = 0
        for feature in geojson_data["features"]:
            if feature_count >= self.max_features:
                break

            coords = self._extract_feature_coords(feature)
            if len(coords) < 2:
                continue

            for x, y in coords[: self.max_points]:
                tokens.append(self._encode_coord(x))
                tokens.append(self._encode_coord(y))

            tokens.append(self.sep_token_id)
            feature_count += 1

        if tokens[-1] == self.sep_token_id:
            tokens[-1] = self.eos_token_id
        else:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids: Iterable[int]) -> Dict:
        """把离散 token id 解码回简单的 GeoJSON FeatureCollection。"""
        features = []
        coords = []

        for token_id in token_ids:
            if token_id in (self.pad_token_id, self.bos_token_id):
                continue

            if token_id in (self.sep_token_id, self.eos_token_id):
                if len(coords) >= 2:
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "LineString",
                                "coordinates": coords,
                            },
                            "properties": {},
                        }
                    )
                coords = []
                if token_id == self.eos_token_id:
                    break
                continue

            value = self._decode_coord(token_id)
            if value is None:
                continue

            if len(coords) == 0 or len(coords[-1]) == 2:
                coords.append([value])
            else:
                coords[-1].append(value)

        valid_features = []
        for feature in features:
            line_coords = [coord for coord in feature["geometry"]["coordinates"] if len(coord) == 2]
            if len(line_coords) >= 2:
                feature["geometry"]["coordinates"] = line_coords
                valid_features.append(feature)

        return {"type": "FeatureCollection", "features": valid_features}

    def _extract_feature_coords(self, feature: Dict) -> List[List[float]]:
        """从单个 GeoJSON feature 中提取线状坐标。"""
        geometry = feature.get("geometry")
        if not geometry:
            return []

        geom = shape(geometry)
        if isinstance(geom, LineString):
            return [[float(x), float(y)] for x, y in geom.coords]
        if isinstance(geom, Point):
            return [[float(geom.x), float(geom.y)]]
        if hasattr(geom, "geoms"):
            coords: List[List[float]] = []
            for sub_geom in geom.geoms:
                if isinstance(sub_geom, LineString):
                    coords.extend([[float(x), float(y)] for x, y in sub_geom.coords])
            return coords
        return []

    def _encode_coord(self, value: float) -> int:
        """把单个坐标量化到 tokenizer 词表中。"""
        clipped = min(max(int(round(float(value))), 0), self.max_coord)
        return self.coord_token_offset + clipped

    def _decode_coord(self, token_id: int) -> Optional[int]:
        """把单个坐标 token id 映射回整数坐标。"""
        value = int(token_id) - self.coord_token_offset
        if 0 <= value <= self.max_coord:
            return value
        return None

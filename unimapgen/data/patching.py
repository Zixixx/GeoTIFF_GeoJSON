from __future__ import annotations

from typing import Dict, List


def generate_patch_windows(height: int, width: int, tile_size: int, overlap: int = 0) -> List[Dict]:
    """
    生成训练和推理共用的 patch 窗口。

    overlap 会让相邻窗口共享一部分区域，主要用于减少目标落在 patch 边界时被截断或漏检的风险。
    返回的 width/height 表示该窗口在原图中的真实有效尺寸，右边界和下边界不足 tile_size 时由调用方 padding。
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"影像尺寸必须为正数，当前为 height={height}, width={width}")
    if tile_size <= 0:
        raise ValueError(f"tile_size 必须为正数，当前为 {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap 不能为负数，当前为 {overlap}")
    if overlap >= tile_size:
        raise ValueError(f"overlap 必须小于 tile_size，当前 overlap={overlap}, tile_size={tile_size}")

    stride = tile_size - overlap
    y_positions = list(range(0, max(height - tile_size + 1, 1), stride))
    x_positions = list(range(0, max(width - tile_size + 1, 1), stride))

    last_y = max(0, height - tile_size)
    last_x = max(0, width - tile_size)
    if not y_positions or y_positions[-1] != last_y:
        y_positions.append(last_y)
    if not x_positions or x_positions[-1] != last_x:
        x_positions.append(last_x)

    windows = []
    seen_offsets = set()
    for y in y_positions:
        for x in x_positions:
            if (x, y) in seen_offsets:
                continue
            seen_offsets.add((x, y))
            windows.append(
                {
                    "offset": (x, y),
                    "width": min(tile_size, width - x),
                    "height": min(tile_size, height - y),
                }
            )
    return windows

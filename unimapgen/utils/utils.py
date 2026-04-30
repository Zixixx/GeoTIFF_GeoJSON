from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
import torch
import yaml
import json

from unimapgen.data.dataset import GeoMapDataset
from unimapgen.data.tokenizer import GeoTokenizer
from unimapgen.models.qwen_geo_generator import DinoEncoder, QwenGeoGenerator


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_path(path_value):
    """把配置路径解析为绝对路径；相对路径以项目根目录为根。"""
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Optional[str] = None) -> Dict:
    """读取 YAML 配置；未指定时默认使用 configs/default.yaml。"""
    final_path = resolve_path(config_path) if config_path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(final_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_tokenizer(config: Dict) -> GeoTokenizer:
    """按配置构造几何 tokenizer。"""
    return GeoTokenizer(
        max_features=config["model"]["max_features"],
        max_points=config["model"]["max_points_per_feature"],
        max_coord=config["data"]["tile_size_px"] - 1,
    )


def collect_pairs(image_root: Path, label_root: Path):
    """按文件 stem 匹配影像与 GeoJSON；同名时优先选择 .tiff。"""
    image_files = {}
    for extension in ("*.tif", "*.TIF", "*.tiff", "*.TIFF"):
        for path in image_root.glob(extension):
            current = image_files.get(path.stem)
            if current is None or path.suffix.lower() == ".tiff":
                image_files[path.stem] = path
    label_files = {path.stem: path for path in label_root.glob("*.geojson")}
    shared_stems = sorted(set(image_files) & set(label_files))
    return [(image_files[stem], label_files[stem]) for stem in shared_stems]


def collect_image_files(image_root: Path):
    """收集 tif/tiff 影像；同名时优先选择 .tiff。"""
    by_stem = {}
    for extension in ("*.tif", "*.TIF", "*.tiff", "*.TIFF"):
        for path in image_root.glob(extension):
            current = by_stem.get(path.stem)
            if current is None or path.suffix.lower() == ".tiff":
                by_stem[path.stem] = path
    return [by_stem[stem] for stem in sorted(by_stem)]



def collect_masks(mask_root: Optional[Path]):
    """收集目录中的 tif/tiff 掩码文件。"""
    if not mask_root or not mask_root.exists():
        return []
    masks = []
    for extension in ("*.tif", "*.TIF", "*.tiff", "*.TIFF"):
        masks.extend(sorted(mask_root.glob(extension)))
    return masks

def split_config(config: Dict, split: str) -> Tuple[Path, Optional[Path], Optional[Path], Path, Path]:
    """读取指定 split 的路径配置。"""
    split_cfg = config["data"].get(split)
    if not isinstance(split_cfg, dict):
        raise ValueError(f"配置缺少 data.{split}")
    image_root = resolve_path(split_cfg.get("image_root"))
    label_root = resolve_path(split_cfg.get("label_root"))
    mask_root = resolve_path(split_cfg.get("mask_root"))
    feature_root = resolve_path(split_cfg.get("feature_root"))
    token_root = resolve_path(split_cfg.get("token_root"))
    if image_root is None:
        raise ValueError(f"配置缺少 data.{split}.image_root")
    if feature_root is None:
        feature_root = image_root.parent / "feature"
    if token_root is None:
        token_root = image_root.parent / "token"
    return image_root, label_root, mask_root, feature_root, token_root


def find_same_stem_raster(root: Optional[Path], stem: str) -> Optional[Path]:
    """在指定目录中查找与给定 stem 同名的 tif/tiff 文件。"""
    if not root or not root.exists():
        return None
    for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def build_mask_paths(pairs, mask_root: Optional[Path]):
    """按影像 stem 收集可选的 mask 路径。"""
    if not mask_root or not mask_root.exists():
        return None
    mask_paths = []
    for image_path, _ in pairs:
        found = find_same_stem_raster(mask_root, image_path.stem)
        mask_paths.append(str(found) if found else "")
    return mask_paths if any(mask_paths) else None


def build_feature_paths_for_pairs(pairs, feature_root: Optional[Path]):
    """按影像 stem 收集逐图缓存的特征路径。"""
    if not feature_root:
        return None
    return [str(feature_root / f"{image_path.stem}.pt") for image_path, _ in pairs]


def build_token_paths_for_pairs(pairs, token_root: Optional[Path]):
    """按影像 stem 收集逐图缓存的 token 路径。"""
    if not token_root:
        return None
    return [str(token_root / f"{image_path.stem}.pt") for image_path, _ in pairs]


def build_dataset(config: Dict, split: str, tokenizer: GeoTokenizer):
    """构造指定 split 的 patch 数据集。"""
    image_root, label_root, mask_root, feature_root, token_root = split_config(config, split)
    if not image_root.exists():
        raise FileNotFoundError(f"{split} 影像目录不存在: {image_root}")
    if label_root is None or not label_root.exists():
        raise FileNotFoundError(f"{split} 标签目录不存在: {label_root}")

    pairs = collect_pairs(image_root, label_root)
    if not pairs:
        raise RuntimeError(f"{split} 中没有找到同名影像/标签对: {image_root} 和 {label_root}")

    feature_paths = build_feature_paths_for_pairs(pairs, feature_root)
    token_paths = build_token_paths_for_pairs(pairs, token_root)

    missing_features = [path for path in (feature_paths or []) if not Path(path).exists()]
    missing_tokens = [path for path in (token_paths or []) if not Path(path).exists()]
    if missing_features:
        raise FileNotFoundError(f"{split} 缺少离线特征文件，例如: {missing_features[0]}")
    if missing_tokens:
        raise FileNotFoundError(f"{split} 缺少离线 token 文件，例如: {missing_tokens[0]}")

    dataset = GeoMapDataset(
        image_paths=[str(image_path) for image_path, _ in pairs],
        label_paths=[str(label_path) for _, label_path in pairs],
        mask_paths=build_mask_paths(pairs, mask_root),
        feature_paths=feature_paths,
        token_paths=token_paths,
        tile_size=config["data"]["tile_size_px"],
        overlap=config["data"].get("overlap_px", 0),
        tokenizer=tokenizer,
        min_mask_ratio=config["data"].get("min_mask_ratio", 0.02),
        cache_file_limit=int(config.get("training", {}).get("dataset_cache_files", 1)),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"{split} 数据集没有生成有效 patch，请检查 tile_size 或 mask 阈值。")
    return dataset, len(pairs)


def maybe_wrap_data_parallel(module, config: Dict):
    """按配置把模块包成 DataParallel。"""
    training_cfg = config.get("training", {})
    use_dp = bool(training_cfg.get("use_data_parallel", False))
    if not use_dp or not torch.cuda.is_available():
        return module
    if torch.cuda.device_count() < 2:
        print("已请求 DataParallel，但当前只有 1 张 CUDA 设备。", flush=True)
        return module
    device_ids = training_cfg.get("device_ids")
    device_ids = [int(x) for x in device_ids] if device_ids else list(range(torch.cuda.device_count()))
    print(f"启用 DataParallel，device_ids={device_ids}", flush=True)
    return torch.nn.DataParallel(module, device_ids=device_ids)


def unwrap_model(module):
    """从 DataParallel 中取回原始模型对象。"""
    return module.module if isinstance(module, torch.nn.DataParallel) else module


def get_amp_settings(config: Dict):
    """根据配置决定是否启用 AMP，以及使用 bf16 还是 fp16。"""
    if not torch.cuda.is_available():
        return False, None, False
    mode = str(config.get("training", {}).get("mixed_precision", "auto")).lower()
    if mode in {"none", "false", "off"}:
        return False, None, False
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("已请求 mixed_precision=bf16，但当前 CUDA 设备不支持 bf16。")
        return True, torch.bfloat16, False
    if mode == "fp16":
        return True, torch.float16, True
    if mode == "auto":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        return True, torch.float16, True
    raise ValueError("training.mixed_precision 必须是 auto、bf16、fp16 或 none。")


def build_optimizer(model, config: Dict):
    """构建训练优化器；当前仅支持 AdamW。"""
    training_cfg = config.get("training", {})
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("没有可训练参数，请检查 qwen_tuning_mode / LoRA 配置。")
    if optimizer_name != "adamw":
        raise ValueError("当前仅支持 optimizer=adamw。")
    betas = training_cfg.get("adam_betas", [0.9, 0.999])
    if len(betas) != 2:
        raise ValueError("training.adam_betas 必须包含两个数值。")
    return torch.optim.AdamW(
        trainable_parameters,
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(training_cfg.get("adam_epsilon", 1e-8)),
    )


def build_visual_encoder(config: Dict, device: torch.device):
    """单独构造 DINO 编码器，用于离线提取视觉特征。"""
    model_cfg = config["model"]
    encoder = DinoEncoder(
        dino_model_path=model_cfg.get("backbone"),
        dino_arch=model_cfg.get("dino_arch", "vit_b_16"),
        dino_hub_model=model_cfg.get("dino_hub_model"),
        local_files_only=model_cfg.get("local_files_only", True),
        freeze=True,
    )
    return encoder.to(device)


def feature_root_for_split(config: Dict, split: str) -> Path:
    """返回某个 split 的逐图特征目录。"""
    _, _, _, feature_root, _ = split_config(config, split)
    feature_root.mkdir(parents=True, exist_ok=True)
    return feature_root


def token_root_for_split(config: Dict, split: str) -> Path:
    """返回某个 split 的逐图 token 目录。"""
    _, _, _, _, token_root = split_config(config, split)
    token_root.mkdir(parents=True, exist_ok=True)
    return token_root


def feature_path_for_image(config: Dict, split: str, image_stem: str) -> Path:
    """返回某个影像对应的特征路径。"""
    return feature_root_for_split(config, split) / f"{image_stem}.pt"


def token_path_for_image(config: Dict, split: str, image_stem: str) -> Path:
    """返回某个影像对应的 token 路径。"""
    return token_root_for_split(config, split) / f"{image_stem}.pt"


def normalize_image(image):
    """把常见遥感栅格值域归一化到 [0, 1]。"""
    image = np.asarray(image, dtype=np.float32)
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
    return ((np.nan_to_num(image, nan=float(p1)) - float(p1)) / max(float(p99 - p1), 1e-6)).clip(0.0, 1.0)


def create_test_patches(image, tile_size=896, overlap=0):
    """把 CHW 影像切成带 padding 的推理 patch，并记录 offset。"""
    if image.ndim != 3:
        raise ValueError(f"期望影像形状为 [C, H, W]，实际为 {image.shape}")

    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    elif image.shape[0] == 2:
        image = np.concatenate([image, image[:1]], axis=0)
    elif image.shape[0] > 3:
        image = image[:3]
    image = normalize_image(image)

    return GeoMapDataset.tile_image_array(
        image=image,
        mask=None,
        tile_size=tile_size,
        overlap=overlap,
        min_mask_ratio=0.0,
    )


def load_patches_for_image(config: Dict, image_path: Path, mask_path: Optional[Path]) -> list[Dict]:
    """读取单张影像并按训练规则切 patch。"""
    with rasterio.open(image_path) as src:
        image = GeoMapDataset._read_rgb(src)

    mask = None
    if mask_path and mask_path.exists():
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        if mask.shape != image.shape[1:]:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape[1:]}: {mask_path}")

    return GeoMapDataset.tile_image_array(
        image=image,
        mask=mask,
        tile_size=int(config["data"]["tile_size_px"]),
        overlap=int(config["data"].get("overlap_px", 0)),
        min_mask_ratio=float(config["data"].get("min_mask_ratio", 0.02)),
    )


def extract_and_cache_image_features(
    config: Dict,
    split: str,
    image_path: Path,
    mask_path: Optional[Path],
    device: torch.device,
    encoder=None,
) -> int:
    """为单张影像提取 DINO 特征并按文件名缓存。"""
    own_encoder = encoder is None
    if own_encoder:
        encoder = build_visual_encoder(config, device)
        encoder = maybe_wrap_data_parallel(encoder, config)
        encoder.eval()

    patches = load_patches_for_image(config, image_path, mask_path)
    batch_size = int(config.get("training", {}).get("feature_batch_size", 1))
    amp_enabled, amp_dtype, _ = get_amp_settings(config)
    feature_dtype_name = str(config.get("data", {}).get("feature_dtype", "float16")).lower()
    if feature_dtype_name in {"fp16", "float16", "half"}:
        feature_dtype = torch.float16
    elif feature_dtype_name in {"fp32", "float32", "float"}:
        feature_dtype = torch.float32
    else:
        raise ValueError(f"不支持的数据特征保存精度: {feature_dtype_name}，可选值为 float16 或 float32。")
    chunks = []

    for start in range(0, len(patches), batch_size):
        batch_tensor = torch.stack(
            [torch.from_numpy(GeoMapDataset._normalize_image(patch["image"])).float() for patch in patches[start : start + batch_size]],
            dim=0,
        ).to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                visual_features = encoder(batch_tensor)
        chunks.append(visual_features.detach().to(dtype=feature_dtype, device="cpu"))
        del batch_tensor, visual_features

    features = torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0), dtype=feature_dtype)
    cache_file = feature_path_for_image(config, split, image_path.stem)
    torch.save(features, cache_file)
    print(f"[{split}] 已缓存 {image_path.name}: {cache_file}", flush=True)

    del patches, chunks, features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if own_encoder:
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return int(torch.load(cache_file, map_location="cpu").shape[-1]) if cache_file.exists() else 0


def extract_and_cache_split_features(config: Dict, split: str, device: torch.device) -> int:
    """按影像逐张提取某个 split 的 DINO 特征并缓存。"""
    image_root, label_root, mask_root, _, _ = split_config(config, split)
    if not image_root.exists():
        raise FileNotFoundError(f"{split} 影像目录不存在: {image_root}")

    if split in {"train", "val"}:
        if label_root is None or not label_root.exists():
            raise FileNotFoundError(f"{split} 标签目录不存在: {label_root}")
        image_items = [image_path for image_path, _ in collect_pairs(image_root, label_root)]
    else:
        image_items = collect_image_files(image_root)

    if not image_items:
        raise RuntimeError(f"{split} 中没有可处理的影像: {image_root}")

    encoder = build_visual_encoder(config, device)
    encoder = maybe_wrap_data_parallel(encoder, config)
    encoder.eval()
    feature_size = None

    for image_path in image_items:
        mask_path = find_same_stem_raster(mask_root, image_path.stem) if split in {"train", "val"} else None
        current_size = extract_and_cache_image_features(
            config=config,
            split=split,
            image_path=image_path,
            mask_path=mask_path,
            device=device,
            encoder=encoder,
        )
        if feature_size is None:
            feature_size = current_size
        elif current_size != feature_size:
            raise RuntimeError(
                f"{split} 特征维度不一致: previous={feature_size}, current={current_size}, image={image_path.name}"
            )

    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return int(feature_size or 0)


def extract_and_cache_image_tokens(
    config: Dict,
    split: str,
    image_path: Path,
    label_path: Path,
    mask_path: Optional[Path],
    tokenizer: GeoTokenizer,
) -> int:
    """为单张影像离线提取 patch GeoJSON 的 token 序列。"""
    with rasterio.open(image_path) as src:
        image = GeoMapDataset._read_rgb(src)

    with open(label_path, encoding="utf-8") as file:
        labels = json.load(file)

    mask = None
    if mask_path and mask_path.exists():
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        if mask.shape != image.shape[1:]:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape[1:]}: {mask_path}")

    helper = GeoMapDataset(
        image_paths=[],
        label_paths=[],
        tile_size=int(config["data"]["tile_size_px"]),
        overlap=int(config["data"].get("overlap_px", 0)),
        min_mask_ratio=float(config["data"].get("min_mask_ratio", 0.02)),
    )
    patches = helper._tile_image(image, mask, helper.tile_size, helper.overlap)
    encoded_sequences = []
    for patch in patches:
        patch_labels = helper._extract_patch_labels(
            labels=labels,
            offset=patch["offset"],
            patch_size=(patch["width"], patch["height"]),
        )
        encoded_sequences.append(tokenizer.encode(patch_labels))

    cache_file = token_path_for_image(config, split, image_path.stem)
    torch.save({"encoded_sequences": encoded_sequences}, cache_file)
    print(f"[{split}] 已缓存 token {image_path.name}: {cache_file}", flush=True)

    del image, labels, mask, patches, encoded_sequences, helper
    gc.collect()
    return len(torch.load(cache_file, map_location="cpu")["encoded_sequences"])


def extract_and_cache_split_tokens(config: Dict, split: str, tokenizer: GeoTokenizer) -> int:
    """按影像逐张提取 train/val 的离线 token。"""
    image_root, label_root, mask_root, _, _ = split_config(config, split)
    if not image_root.exists():
        raise FileNotFoundError(f"{split} 影像目录不存在: {image_root}")
    if label_root is None or not label_root.exists():
        raise FileNotFoundError(f"{split} 标签目录不存在: {label_root}")

    pairs = collect_pairs(image_root, label_root)
    if not pairs:
        raise RuntimeError(f"{split} 中没有可处理的影像/标签对: {image_root} 和 {label_root}")

    patch_count = 0
    for image_path, label_path in pairs:
        mask_path = find_same_stem_raster(mask_root, image_path.stem)
        patch_count += extract_and_cache_image_tokens(
            config=config,
            split=split,
            image_path=image_path,
            label_path=label_path,
            mask_path=mask_path,
            tokenizer=tokenizer,
        )
    return patch_count


def build_qwen_only_model(config: Dict, tokenizer: GeoTokenizer, visual_feature_size: int, device: torch.device):
    """构造只接收预提取视觉特征的 Qwen 模型。"""
    model_cfg = config["model"]
    model = QwenGeoGenerator(
        qwen_model_path=model_cfg.get("name"),
        dino_model_path=model_cfg.get("backbone"),
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_cfg.get("hidden_size", 256),
        pad_token_id=tokenizer.pad_token_id,
        dino_arch=model_cfg.get("dino_arch", "vit_b_16"),
        dino_hub_model=model_cfg.get("dino_hub_model"),
        local_files_only=model_cfg.get("local_files_only", True),
        qwen_tuning_mode=model_cfg.get("qwen_tuning_mode", "lora"),
        lora_r=model_cfg.get("lora_r", 16),
        lora_alpha=model_cfg.get("lora_alpha", 32),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        lora_target_modules=model_cfg.get("lora_target_modules"),
        lora_modules_to_save=model_cfg.get("lora_modules_to_save"),
        lora_ensure_weight_tying=model_cfg.get("lora_ensure_weight_tying", True),
        build_visual_encoder=False,
        visual_feature_size=visual_feature_size,
        embedding_mean_resizing=model_cfg.get("embedding_mean_resizing", False),
        gradient_checkpointing=config.get("training", {}).get("gradient_checkpointing", False),
    )
    return model.to(device)


def configure_runtime(config: Dict) -> torch.device:
    """配置训练/提特征时的运行环境并返回设备。"""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if bool(config.get("training", {}).get("allow_tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

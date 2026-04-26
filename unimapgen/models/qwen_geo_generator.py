from __future__ import annotations

import json
import inspect
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoModelForCausalLM

    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  
    AutoModel = None
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = exc

try:
    from peft import LoraConfig, PeftModel, get_peft_model

    _PEFT_IMPORT_ERROR = None
except Exception as exc:  
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
    _PEFT_IMPORT_ERROR = exc


def _as_model_root(path: Optional[str]) -> Optional[str]:
    """把单个 safetensors 文件路径规整为 HuggingFace 模型目录。"""
    if not path:
        return None
    model_path = Path(path)
    if model_path.is_file() and model_path.suffix == ".safetensors":
        return str(model_path.parent)
    return str(model_path)


def _find_first_file(root: Path, suffixes) -> Optional[Path]:
    """只在指定目录本身查找第一个指定后缀文件。"""
    for suffix in suffixes:
        found = sorted(root.glob(f"*{suffix}"))
        if found:
            return found[0]
    return None


def _infer_dino_hub_model(path: Path, fallback: Optional[str]) -> str:
    """从权重文件名推断 DINO hub 模型名，失败时使用 fallback。"""
    if fallback and str(fallback).lower() not in {"none", "null"}:
        return str(fallback)
    match = re.search(r"(dinov3_vit[^_\.]+)", path.name)
    if match:
        return match.group(1)
    return "vit_b_16"


def _resolve_dino_paths(dino_model_path: str, dino_arch: Optional[str], dino_hub_model: Optional[str]):
    """只支持 DINO 权重文件和 hubconf.py 放在同一个目录中。"""
    path = Path(dino_model_path)
    if path.is_dir():
        weight_path = _find_first_file(path, (".pth", ".pt"))
        if weight_path is None:
            return path, dino_arch or "vit_b_16", dino_hub_model, None
        repo_dir = path if (path / "hubconf.py").is_file() else None
        inferred = _infer_dino_hub_model(weight_path, dino_hub_model or dino_arch)
        return weight_path, inferred, inferred, repo_dir
    inferred = _infer_dino_hub_model(path, dino_hub_model or dino_arch)
    return path, inferred, inferred, None


def _extract_checkpoint_state_dict(checkpoint_obj):
    """从常见 checkpoint 容器中提取 state_dict。"""
    if isinstance(checkpoint_obj, dict):
        for key in ("model", "state_dict", "teacher"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                checkpoint_obj = value
                break
    if not isinstance(checkpoint_obj, dict):
        raise RuntimeError(f"无法从 checkpoint 中提取 state_dict，实际类型: {type(checkpoint_obj)!r}")

    cleaned = {}
    for key, value in checkpoint_obj.items():
        key = str(key)
        for prefix in ("module.", "backbone.", "model.", "teacher."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


class _GenericViTEncoder(nn.Module):
    """通过 timm 或 torchvision 加载普通 ViT .pth 权重。"""

    def __init__(self, weights_path: str, arch: str, out_hw=(8, 8)):
        super().__init__()
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"不支持的 DINO .pth 对象类型: {type(state_dict)!r}")

        cleaned = {}
        for key, value in state_dict.items():
            key = str(key)
            for prefix in ("module.", "backbone.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            if not key.startswith("heads."):
                cleaned[key] = value

        self.backend = "timm"
        try:
            import timm

            self.model = timm.create_model(str(arch), pretrained=False, num_classes=0)
            missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
            self.hidden_size = int(getattr(self.model, "num_features", 768))
            print(f"[DINO] 已加载到 timm {arch}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        except Exception:
            from torchvision.models import get_model

            self.backend = "torchvision"
            self.model = get_model(str(arch), weights=None)
            missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
            hidden_size = getattr(self.model, "hidden_dim", None)
            if hidden_size is None and hasattr(self.model, "heads"):
                hidden_size = getattr(getattr(self.model.heads, "head", None), "in_features", None)
            self.hidden_size = int(hidden_size or 768)
            print(f"[DINO] 已加载到 torchvision {arch}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        self.out_hw = tuple(out_hw) if out_hw else None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.backend == "timm":
            out = self.model.forward_features(images)
            if isinstance(out, dict):
                out = out.get("x", out.get("last_hidden_state"))
            if out.ndim == 4:
                out = out.flatten(2).transpose(1, 2).contiguous()
            if out.ndim == 2:
                out = out.unsqueeze(1)
            tokens = out[:, 1:, :] if out.ndim == 3 and out.shape[1] > 1 else out
            return self._pool(tokens)

        x = self.model._process_input(images)
        batch_size = x.shape[0]
        cls = self.model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.model.encoder(x)
        return self._pool(x[:, 1:, :])

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """把任意数量 patch token 池化到固定 token 网格，控制 Qwen 前缀长度。"""
        if self.out_hw is None:
            return tokens
        batch_size = tokens.shape[0]
        side = max(1, int(round(tokens.shape[1] ** 0.5)))
        usable = min(tokens.shape[1], side * side)
        tokens = tokens[:, :usable, :]
        feat = tokens.reshape(batch_size, side, side, -1).permute(0, 3, 1, 2).contiguous()
        feat = F.adaptive_avg_pool2d(feat, self.out_hw)
        return feat.flatten(2).transpose(1, 2).contiguous()


class DinoEncoder(nn.Module):
    """把本地 DINO 权重统一封装成视觉 token 编码器。"""

    def __init__(
        self,
        dino_model_path: str,
        dino_arch: str = "vit_b_16",
        dino_hub_model: Optional[str] = None,
        local_files_only: bool = True,
        freeze: bool = True,
        out_hw=(8, 8),
    ):
        super().__init__()
        if not dino_model_path:
            raise ValueError("model.backbone 必须指向本地 DINO 目录或 .pth/.pt 权重文件。")

        self.image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        path, dino_arch, dino_hub_model, resolved_repo_dir = _resolve_dino_paths(
            dino_model_path=dino_model_path,
            dino_arch=dino_arch,
            dino_hub_model=dino_hub_model,
        )
        repo_dir = str(resolved_repo_dir) if resolved_repo_dir else None

        if path.is_file() and path.suffix.lower() in {".pt", ".pth"}:
            if str(dino_hub_model or dino_arch).startswith("dinov3_"):
                self.model = self._load_dinov3_hub_model(
                    repo_dir,
                    dino_hub_model or dino_arch,
                    str(path),
                    local_files_only=bool(local_files_only),
                )
                self.hidden_size = int(getattr(self.model, "embed_dim", getattr(self.model, "num_features", 4096)))
                self.mode = "dinov3_hub"
                print(f"[DINO] 已加载 {dino_hub_model or dino_arch}: hidden={self.hidden_size}", flush=True)
            else:
                self.model = _GenericViTEncoder(str(path), arch=dino_arch, out_hw=out_hw)
                self.hidden_size = int(self.model.hidden_size)
                self.mode = "generic_vit"
        else:
            if AutoModel is None:
                raise RuntimeError(f"transformers.AutoModel 不可用，原始错误: {_TRANSFORMERS_IMPORT_ERROR!r}")
            model_root = _as_model_root(str(path))
            self.model = AutoModel.from_pretrained(
                model_root,
                local_files_only=bool(local_files_only),
                trust_remote_code=True,
                torch_dtype="auto",
            )
            self.hidden_size = int(getattr(self.model.config, "hidden_size", 768))
            self.mode = "hf"
            print(f"[DINO] 已加载 HuggingFace 模型: {model_root}", flush=True)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    @staticmethod
    def _load_dinov3_hub_model(
        repo_dir: Optional[str],
        hub_model: str,
        weights_path: str,
        local_files_only: bool = True,
    ):
        """通过本地 DINO 仓库或 torch hub 加载官方 DINO 单文件权重。"""
        if repo_dir:
            # 对本地 repo 直接手动 load_state_dict
            try:
                model = torch.hub.load(str(repo_dir), str(hub_model), source="local", pretrained=False)
                checkpoint = torch.load(weights_path, map_location="cpu")
                state_dict = _extract_checkpoint_state_dict(checkpoint)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(
                    f"[DINO] 本地权重已加载: missing={len(missing)} unexpected={len(unexpected)}",
                    flush=True,
                )
                return model
            except Exception as exc:
                print(f"[DINO] 本地直载失败，回退到 hub weights 参数: {exc}", flush=True)
                return torch.hub.load(str(repo_dir), str(hub_model), source="local", weights=str(weights_path))
        if local_files_only:
            raise FileNotFoundError(
                "DINO 离线加载需要在 model.backbone 指向的目录中同时放置 .pth 权重和官方源码根目录内容，"
                "其中必须包含 hubconf.py。"
            )
        return torch.hub.load("facebookresearch/dinov3", str(hub_model), weights=str(weights_path))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        x = (images - mean) / std.clamp_min(1e-6)

        if self.mode == "hf":
            out = self.model(pixel_values=x)
            tokens = out.last_hidden_state
            return tokens[:, 1:, :] if tokens.shape[1] > 1 else tokens

        if self.mode == "dinov3_hub":
            return self._forward_dinov3(x)

        return self.model(x)

    def _forward_dinov3(self, images: torch.Tensor) -> torch.Tensor:
        """处理 DINO forward_features 的不同返回结构。"""
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(images)
        else:
            out = self.model(images)
        if isinstance(out, dict):
            for key in ("x_norm_patchtokens", "patch_tokens", "tokens", "last_hidden_state", "x"):
                value = out.get(key)
                if value is not None:
                    out = value
                    break
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 4:
            out = out.flatten(2).transpose(1, 2).contiguous()
        if out.ndim == 2:
            out = out.unsqueeze(1)
        return out[:, 1:, :] if out.ndim == 3 and out.shape[1] > 1 else out


class QwenGeoGenerator(nn.Module):
    """只保留 Qwen + DINO 的图像条件几何 token 生成器。"""

    def __init__(
        self,
        qwen_model_path: str,
        dino_model_path: str,
        vocab_size: int,
        hidden_size: int = 256,
        pad_token_id: int = 0,
        dino_arch: str = "vit_b_16",
        dino_hub_model: Optional[str] = None,
        local_files_only: bool = True,
        qwen_tuning_mode: str = "lora",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list] = None,
        lora_modules_to_save: Optional[list] = None,
        lora_ensure_weight_tying: bool = True,
        lora_adapter_path: Optional[str] = None,
        lora_adapter_trainable: bool = True,
        build_visual_encoder: bool = True,
        visual_feature_size: Optional[int] = None,
        embedding_mean_resizing: bool = False,
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        if AutoModelForCausalLM is None:
            raise RuntimeError(f"transformers.AutoModelForCausalLM 不可用，原始错误: {_TRANSFORMERS_IMPORT_ERROR!r}")
        if not qwen_model_path:
            raise ValueError("model.name 必须指向本地 Qwen 模型目录。")
        if not dino_model_path:
            raise ValueError("model.backbone 必须指向本地 DINO 权重。")

        self.qwen_model_path = qwen_model_path
        self.dino_model_path = dino_model_path
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.pad_token_id = int(pad_token_id)
        self.dino_arch = dino_arch
        self.dino_hub_model = dino_hub_model
        self.local_files_only = bool(local_files_only)
        self.qwen_tuning_mode = str(qwen_tuning_mode or "lora").lower()
        if self.qwen_tuning_mode not in {"full", "lora", "frozen"}:
            raise ValueError("qwen_tuning_mode 必须是 full、lora 或 frozen。")
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules = list(
            lora_target_modules
            or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.lora_modules_to_save = list(lora_modules_to_save or ["embed_tokens", "lm_head"])
        self.lora_ensure_weight_tying = bool(lora_ensure_weight_tying)
        self.lora_adapter_path = lora_adapter_path
        self.build_visual_encoder = bool(build_visual_encoder)
        self.visual_feature_size = None
        self.embedding_mean_resizing = bool(embedding_mean_resizing)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.attn_implementation = attn_implementation

        if self.build_visual_encoder:
            self.visual_encoder = DinoEncoder(
                dino_model_path=dino_model_path,
                dino_arch=dino_arch,
                dino_hub_model=dino_hub_model,
                local_files_only=local_files_only,
            )
            self.visual_feature_size = int(self.visual_encoder.hidden_size)
        else:
            if visual_feature_size is None:
                raise ValueError("禁用视觉编码器时必须提供 visual_feature_size。")
            self.visual_encoder = None
            self.visual_feature_size = int(visual_feature_size)

        qwen_root = _as_model_root(qwen_model_path)
        print(f"[Qwen] 正在加载本地模型: {qwen_root}", flush=True)
        llm_kwargs = dict(
            local_files_only=bool(local_files_only),
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        if self.attn_implementation:
            llm_kwargs["attn_implementation"] = self.attn_implementation
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(qwen_root, **llm_kwargs)
        except Exception as exc:
            if self.attn_implementation:
                print(
                    f"[Qwen] attn_implementation={self.attn_implementation} 加载失败，将回退到默认注意力实现: {exc}",
                    flush=True,
                )
                llm_kwargs.pop("attn_implementation", None)
                self.llm = AutoModelForCausalLM.from_pretrained(qwen_root, **llm_kwargs)
                self.attn_implementation = None
            else:
                raise
        self.base_vocab_size = int(self.llm.get_input_embeddings().num_embeddings)
        self.qwen_token_offset = self.base_vocab_size
        self.llm.resize_token_embeddings(
            self.base_vocab_size + self.vocab_size,
            mean_resizing=self.embedding_mean_resizing,
        )
        self.hidden_size = int(getattr(self.llm.config, "hidden_size", self.hidden_size))
        self.visual_proj = nn.Linear(int(self.visual_feature_size), self.hidden_size)
        self.llm_embed_dtype = self.llm.get_input_embeddings().weight.dtype

        if self.qwen_tuning_mode == "frozen":
            for param in self.llm.parameters():
                param.requires_grad = False
        elif self.qwen_tuning_mode == "lora":
            if PeftModel is None or LoraConfig is None or get_peft_model is None:
                raise RuntimeError(f"peft 不可用，无法启用 LoRA，原始错误: {_PEFT_IMPORT_ERROR!r}")
            if lora_adapter_path:
                self.llm = PeftModel.from_pretrained(
                    self.llm,
                    lora_adapter_path,
                    is_trainable=bool(lora_adapter_trainable),
                )
                print(f"[Qwen] 已加载 LoRA adapter: {lora_adapter_path}", flush=True)
            else:
                lora_kwargs = dict(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=self.lora_target_modules,
                    modules_to_save=self.lora_modules_to_save,
                )
                if "ensure_weight_tying" in inspect.signature(LoraConfig).parameters:
                    lora_kwargs["ensure_weight_tying"] = self.lora_ensure_weight_tying
                lora_config = LoraConfig(**lora_kwargs)
                self.llm = get_peft_model(self.llm, lora_config)
                print(
                    "[Qwen] 已启用 LoRA: "
                    f"r={self.lora_r}, alpha={self.lora_alpha}, targets={self.lora_target_modules}",
                    flush=True,
                )

        if self.gradient_checkpointing:
            self.enable_gradient_checkpointing()

    def enable_gradient_checkpointing(self) -> None:
        """启用 Qwen 梯度检查点，并关闭 use_cache 以降低训练显存。"""
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()
        if hasattr(self.llm, "config") and hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = False

    def _to_qwen_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """把自定义几何 token id 平移到 Qwen 扩展词表区间。"""
        mapped = ids.clone()
        ignore = mapped.lt(0)
        mapped = mapped.clamp_min(0) + int(self.qwen_token_offset)
        return mapped.masked_fill(ignore, -100)

    def _from_qwen_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """把 Qwen 扩展词表 id 转回自定义几何 token id。"""
        return ids - int(self.qwen_token_offset)

    def forward(self, images=None, visual_features=None, input_ids=None, attention_mask=None, labels=None):
        if input_ids is None:
            raise ValueError("训练前向传播必须提供 input_ids。")

        visual_tokens = self._encode_visual_inputs(images=images, visual_features=visual_features).to(dtype=self.llm_embed_dtype)
        batch_size = int(visual_tokens.shape[0])
        device = visual_tokens.device
        qwen_input_ids = self._to_qwen_ids(input_ids)
        token_embeddings = self.llm.get_input_embeddings()(qwen_input_ids)
        inputs_embeds = torch.cat([visual_tokens, token_embeddings], dim=1)

        visual_mask = torch.ones((batch_size, visual_tokens.shape[1]), device=device, dtype=torch.long)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        full_attention = torch.cat([visual_mask, attention_mask.long()], dim=1)

        qwen_labels = None
        if labels is not None:
            mapped_labels = self._to_qwen_ids(labels)
            prefix_labels = torch.full(
                (batch_size, visual_tokens.shape[1]),
                -100,
                device=device,
                dtype=torch.long,
            )
            qwen_labels = torch.cat([prefix_labels, mapped_labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            labels=qwen_labels,
            use_cache=False,
            return_dict=True,
        )
        return SimpleNamespace(logits=outputs.logits, loss=outputs.loss)

    @torch.no_grad()
    def generate(
        self,
        images=None,
        visual_features=None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_length: int = 256,
        log_interval: int = 0,
        log_prefix: str = "",
    ):
        if images is None and visual_features is None:
            raise ValueError("generate 至少需要 images 或 visual_features。")
        device = images.device if images is not None else visual_features.device
        batch_size = images.size(0) if images is not None else visual_features.size(0)
        visual_tokens = self._encode_visual_inputs(images=images, visual_features=visual_features).to(dtype=self.llm_embed_dtype)
        visual_mask = torch.ones((batch_size, visual_tokens.shape[1]), device=device, dtype=torch.long)
        generated_qwen = torch.full(
            (batch_size, 1),
            int(bos_token_id) + int(self.qwen_token_offset),
            dtype=torch.long,
            device=device,
        )
        allowed = torch.arange(
            int(self.qwen_token_offset),
            int(self.qwen_token_offset) + int(self.vocab_size),
            dtype=torch.long,
            device=device,
        )
        qwen_eos = int(eos_token_id) + int(self.qwen_token_offset)

        for step_idx in range(max_length - 1):
            token_embeddings = self.llm.get_input_embeddings()(generated_qwen).to(dtype=self.llm_embed_dtype)
            inputs_embeds = torch.cat([visual_tokens, token_embeddings], dim=1)
            gen_mask = torch.ones_like(generated_qwen, dtype=torch.long)
            attention_mask = torch.cat([visual_mask, gen_mask], dim=1)
            outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False, return_dict=True)
            logits = outputs.logits[:, -1, :].index_select(dim=1, index=allowed)
            next_custom = torch.argmax(logits, dim=-1, keepdim=True)
            next_qwen = next_custom + int(self.qwen_token_offset)
            generated_qwen = torch.cat([generated_qwen, next_qwen], dim=1)
            if log_interval > 0 and (step_idx + 1) % int(log_interval) == 0:
                print(
                    f"{log_prefix}生成进度: {step_idx + 1}/{max_length - 1} token steps",
                    flush=True,
                )
            if torch.all(next_qwen.squeeze(1) == qwen_eos):
                break
        return self._from_qwen_ids(generated_qwen)

    def _encode_visual_inputs(self, images=None, visual_features=None):
        if visual_features is None:
            if self.visual_encoder is None:
                raise ValueError("当前模型未构建视觉编码器，必须直接提供 visual_features。")
            if images is None:
                raise ValueError("未提供 images。")
            visual_features = self.visual_encoder(images)
        if visual_features.ndim == 2:
            visual_features = visual_features.unsqueeze(1)
        visual_features = visual_features.to(
            device=self.visual_proj.weight.device,
            dtype=self.visual_proj.weight.dtype,
        )
        return self.visual_proj(visual_features)

    def save_pretrained(self, model_path: str, extra_config: Optional[dict] = None) -> None:
        """保存当前训练得到的视觉投影层，并记录基础模型路径。"""
        os.makedirs(model_path, exist_ok=True)
        torch.save({"visual_proj": self.visual_proj.state_dict()}, os.path.join(model_path, "adapter_model.bin"))
        qwen_model_path = self.qwen_model_path
        lora_adapter_path = None
        if self.qwen_tuning_mode == "full":
            qwen_model_path = "qwen_finetuned"
            self.llm.save_pretrained(os.path.join(model_path, qwen_model_path))
        elif self.qwen_tuning_mode == "lora":
            lora_adapter_path = "qwen_lora"
            self.llm.save_pretrained(os.path.join(model_path, lora_adapter_path))
        config = {
            "qwen_model_path": qwen_model_path,
            "dino_model_path": self.dino_model_path,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "visual_feature_size": self.visual_feature_size,
            "pad_token_id": self.pad_token_id,
            "dino_arch": self.dino_arch,
            "dino_hub_model": self.dino_hub_model,
            "local_files_only": self.local_files_only,
            "qwen_tuning_mode": self.qwen_tuning_mode,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "lora_modules_to_save": self.lora_modules_to_save,
            "lora_ensure_weight_tying": self.lora_ensure_weight_tying,
            "lora_adapter_path": lora_adapter_path,
            "embedding_mean_resizing": self.embedding_mean_resizing,
            "gradient_checkpointing": self.gradient_checkpointing,
            "attn_implementation": self.attn_implementation,
        }
        if extra_config:
            config.update(extra_config)
        with open(os.path.join(model_path, "config.json"), "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, model_path, qwen_model_path_override: Optional[str] = None):
        config_file = os.path.join(model_path, "config.json")
        with open(config_file, encoding="utf-8") as file:
            config = json.load(file)

        qwen_tuning_mode = str(
            config.get(
                "qwen_tuning_mode",
                "full" if str(config.get("qwen_model_path", "")).endswith("qwen_finetuned") else "lora",
            )
        ).lower()

        qwen_model_path = config.get("qwen_model_path")
        if qwen_model_path and not os.path.isabs(qwen_model_path):
            qwen_model_path = os.path.join(model_path, qwen_model_path)

        override_root = _as_model_root(qwen_model_path_override) if qwen_model_path_override else None
        if qwen_tuning_mode in {"lora", "frozen"} and override_root:
            if qwen_model_path is None or not Path(qwen_model_path).exists():
                print(
                    f"[Qwen] checkpoint 内记录的基座路径不可用，将回退到当前配置的 model.name: {override_root}",
                    flush=True,
                )
                qwen_model_path = override_root
        elif qwen_tuning_mode == "full":
            print("[Qwen] 当前 checkpoint 为 full 模式，将直接使用 checkpoint 内保存的 qwen_finetuned。", flush=True)

        if qwen_model_path is None:
            raise FileNotFoundError("checkpoint 中未记录 qwen_model_path，且未提供可用的 model.name 覆盖路径。")

        if not Path(qwen_model_path).exists():
            if qwen_tuning_mode == "full":
                raise FileNotFoundError(
                    f"full 模式 checkpoint 需要的 Qwen 目录不存在: {qwen_model_path}。"
                    "这通常说明 qwen_finetuned 没有和 checkpoint 一起拷贝完整。"
                )
            raise FileNotFoundError(
                f"Qwen 基座模型目录不存在: {qwen_model_path}。"
                "如果 checkpoint 里只保存了 LoRA adapter，请在 default.yaml 的 model.name 中填写本地 Qwen 模型目录。"
            )

        lora_adapter_path = config.get("lora_adapter_path")
        if lora_adapter_path and not os.path.isabs(lora_adapter_path):
            lora_adapter_path = os.path.join(model_path, lora_adapter_path)

        instance = cls(
            qwen_model_path=qwen_model_path,
            dino_model_path=config.get("dino_model_path"),
            vocab_size=config.get("vocab_size", 1024),
            hidden_size=config.get("hidden_size", 256),
            visual_feature_size=config.get("visual_feature_size"),
            pad_token_id=config.get("pad_token_id", 0),
            dino_arch=config.get("dino_arch", "vit_b_16"),
            dino_hub_model=config.get("dino_hub_model"),
            local_files_only=config.get("local_files_only", True),
            qwen_tuning_mode=qwen_tuning_mode,
            lora_r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.05),
            lora_target_modules=config.get("lora_target_modules"),
            lora_modules_to_save=config.get("lora_modules_to_save"),
            lora_ensure_weight_tying=config.get("lora_ensure_weight_tying", True),
            lora_adapter_path=lora_adapter_path,
            lora_adapter_trainable=False,
            build_visual_encoder=False,
            embedding_mean_resizing=config.get("embedding_mean_resizing", False),
            gradient_checkpointing=config.get("gradient_checkpointing", False),
            attn_implementation=config.get("attn_implementation"),
        )
        adapter_file = os.path.join(model_path, "adapter_model.bin")
        if not os.path.exists(adapter_file):
            raise FileNotFoundError(f"未找到视觉投影权重: {adapter_file}")
        state = torch.load(adapter_file, map_location="cpu")
        instance.visual_proj.load_state_dict(state["visual_proj"])
        return instance

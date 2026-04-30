from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM

    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = exc

try:
    from peft import LoraConfig, PeftModel, get_peft_model

    _PEFT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    LoraConfig = None
    PeftModel = None
    get_peft_model = None
    _PEFT_IMPORT_ERROR = exc


def _as_model_root(path: Optional[str]) -> Optional[str]:
    """把单个 safetensors 文件路径规整为 Hugging Face 模型目录。"""
    if not path:
        return None
    model_path = Path(path)
    if model_path.is_file() and model_path.suffix == ".safetensors":
        return str(model_path.parent)
    return str(model_path)


def _normalize_optional_name(value: Optional[str]) -> Optional[str]:
    """把 null、none 和空字符串统一视为未填写。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


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


class DinoEncoder(nn.Module):
    """只支持本地 DINO 仓库 + 单个权重文件的视觉编码器。"""

    def __init__(
        self,
        dino_weight_path: str,
        dino_model_name: str,
        dino_repo_or_dir: str,
        freeze: bool = True,
    ):
        super().__init__()

        weight_path = Path(dino_weight_path)
        repo_dir = Path(dino_repo_or_dir)

        self.dino_weight_path = str(weight_path)
        self.dino_model_name = str(dino_model_name)
        self.dino_repo_or_dir = str(repo_dir)
        self.image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        self.model = torch.hub.load(str(repo_dir), self.dino_model_name, source="local", pretrained=False)
        checkpoint = torch.load(str(weight_path), map_location="cpu")
        state_dict = _extract_checkpoint_state_dict(checkpoint)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.hidden_size = int(getattr(self.model, "embed_dim", getattr(self.model, "num_features", 4096)))
        print(
            f"[DINO] 已加载 {self.dino_model_name}: hidden={self.hidden_size}, "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        x = (images - mean) / std.clamp_min(1e-6)

        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(x)
        else:
            out = self.model(x)

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
    """保留 Qwen 与视觉条件输入的几何 token 生成器。"""

    def __init__(
        self,
        qwen_model_path: str,
        dino_weight_path: str,
        dino_model_name: str,
        dino_repo_or_dir: str,
        vocab_size: int,
        hidden_size: int = 256,
        pad_token_id: int = 0,
        local_files_only: bool = True,
        freeze_dino: bool = True,
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
        if not dino_weight_path:
            raise ValueError("model.backbone 必须指向本地 DINO 权重文件。")
        if not dino_model_name:
            raise ValueError("model.dino_model_name 必须填写。")
        if not dino_repo_or_dir:
            raise ValueError("model.dino_repo_or_dir 必须指向本地 DINO 仓库目录。")

        self.qwen_model_path = qwen_model_path
        self.dino_weight_path = dino_weight_path
        self.dino_model_name = dino_model_name
        self.dino_repo_or_dir = dino_repo_or_dir
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.pad_token_id = int(pad_token_id)
        self.local_files_only = bool(local_files_only)
        self.freeze_dino = bool(freeze_dino)
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
                dino_weight_path=dino_weight_path,
                dino_model_name=dino_model_name,
                dino_repo_or_dir=dino_repo_or_dir,
                freeze=freeze_dino,
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
                print(f"{log_prefix}生成进度: {step_idx + 1}/{max_length - 1} token steps", flush=True)
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
            "dino_weight_path": self.dino_weight_path,
            "dino_model_name": self.dino_model_name,
            "dino_repo_or_dir": self.dino_repo_or_dir,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "visual_feature_size": self.visual_feature_size,
            "pad_token_id": self.pad_token_id,
            "local_files_only": self.local_files_only,
            "freeze_dino": self.freeze_dino,
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
            print("[Qwen] 当前 checkpoint 为 full 模式，将直接使用其中保存的 qwen_finetuned。", flush=True)

        if qwen_model_path is None:
            raise FileNotFoundError("checkpoint 中未记录 qwen_model_path，且未提供可用的 model.name 覆盖路径。")
        if not Path(qwen_model_path).exists():
            if qwen_tuning_mode == "full":
                raise FileNotFoundError(
                    f"full 模式 checkpoint 需要的 Qwen 目录不存在: {qwen_model_path}。"
                    "这通常说明 qwen_finetuned 没有和 checkpoint 一起完整拷贝。"
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
            dino_weight_path=config.get("dino_weight_path"),
            dino_model_name=config.get("dino_model_name"),
            dino_repo_or_dir=config.get("dino_repo_or_dir"),
            vocab_size=config.get("vocab_size", 1024),
            hidden_size=config.get("hidden_size", 256),
            visual_feature_size=config.get("visual_feature_size"),
            pad_token_id=config.get("pad_token_id", 0),
            local_files_only=config.get("local_files_only", True),
            freeze_dino=config.get("freeze_dino", True),
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

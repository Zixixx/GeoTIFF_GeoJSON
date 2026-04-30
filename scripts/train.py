#!/usr/bin/env python3
"""
训练脚本。

训练流程为：
1. 先运行 scripts/extract_features.py，按影像生成 train/val 特征文件
2. 再运行 scripts/extract_tokens.py，按影像生成 train/val token 文件
3. 最后运行本脚本，只加载 Qwen + visual_proj 训练
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unimapgen.data.dataset import GeoMapDataset
from unimapgen.utils.utils import (
    build_dataset,
    build_optimizer,
    build_qwen_only_model,
    build_tokenizer,
    configure_runtime,
    feature_root_for_split,
    get_amp_settings,
    load_config,
    maybe_wrap_data_parallel,
    token_root_for_split,
    unwrap_model,
)


def build_grad_scaler(enabled: bool):
    """通过PyTorch AMP API构造 GradScaler。"""
    enabled = bool(enabled and torch.cuda.is_available())
    return torch.amp.GradScaler("cuda", enabled=enabled)


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    amp_enabled: bool = False,
    amp_dtype=None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 0.0,
    global_step: int = 0,
    step_save_every: int = 0,
    step_save_callback=None,
):
    """执行一个训练或验证 epoch，支持 AMP、梯度累积、梯度裁剪和按 step 保存。"""
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        visual_features = batch["visual_features"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    visual_features=visual_features,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                raw_loss = outputs.loss
                loss = raw_loss / max(1, int(grad_accum_steps))

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                should_step = ((batch_idx + 1) % max(1, int(grad_accum_steps)) == 0) or ((batch_idx + 1) == len(loader))
                if should_step:
                    if max_grad_norm and max_grad_norm > 0:
                        if scaler is not None and scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if step_save_every > 0 and global_step % step_save_every == 0 and step_save_callback is not None:
                        step_save_callback(global_step)

        total_loss += float(raw_loss.item())
        if is_train and batch_idx % 10 == 0:
            print(f"训练 batch={batch_idx}, loss={raw_loss.item():.4f}")
    return total_loss / max(1, len(loader)), global_step


def save_checkpoint(model, checkpoint_dir: Path, config: Dict, tokenizer, summary: Dict):
    """保存模型、配置和训练摘要。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(
        str(checkpoint_dir),
        extra_config={
            "training_config": config,
            "tokenizer": {
                "max_features": tokenizer.max_features,
                "max_points": tokenizer.max_points,
                "max_coord": tokenizer.max_coord,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "sep_token_id": tokenizer.sep_token_id,
                "vocab_size": tokenizer.vocab_size,
            },
        },
    )
    with open(checkpoint_dir / "training_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def infer_feature_size(feature_root: Path) -> int:
    """从某个 split 的第一份特征文件推断特征维度。"""
    feature_files = sorted(feature_root.glob("*.pt"))
    if not feature_files:
        raise FileNotFoundError(f"feature 目录不存在或为空: {feature_root}")
    return int(torch.load(feature_files[0], map_location="cpu").shape[-1])


def count_feature_files(feature_root: Path, split: str) -> int:
    """统计某个 split 的特征文件数。"""
    feature_files = sorted(feature_root.glob("*.pt"))
    if not feature_files:
        raise FileNotFoundError(f"{split} feature 目录不存在或为空: {feature_root}")
    return len(feature_files)


def ensure_token_files_exist(token_root: Path, split: str) -> int:
    """确认某个 split 已经生成离线 token 文件。"""
    token_files = sorted(token_root.glob("*.pt"))
    if not token_files:
        raise FileNotFoundError(f"{split} token 目录不存在或为空: {token_root}")
    return len(token_files)


def main():
    print("开始训练 Qwen...")
    config = load_config()
    device = configure_runtime(config)
    print(f"使用设备: {device}")

    tokenizer = build_tokenizer(config)

    train_feature_root = feature_root_for_split(config, "train")
    val_feature_root = feature_root_for_split(config, "val")
    train_token_root = token_root_for_split(config, "train")
    val_token_root = token_root_for_split(config, "val")

    train_feature_size = infer_feature_size(train_feature_root)
    val_feature_size = infer_feature_size(val_feature_root)
    if train_feature_size != val_feature_size:
        raise RuntimeError(f"train/val 特征维度不一致: train={train_feature_size}, val={val_feature_size}")

    train_feature_files = count_feature_files(train_feature_root, "train")
    val_feature_files = count_feature_files(val_feature_root, "val")
    train_token_files = ensure_token_files_exist(train_token_root, "train")
    val_token_files = ensure_token_files_exist(val_token_root, "val")
    print(f"检测到 train feature 文件 {train_feature_files} 个，val feature 文件 {val_feature_files} 个")
    print(f"检测到 train token 文件 {train_token_files} 个，val token 文件 {val_token_files} 个")

    train_dataset, train_pair_count = build_dataset(config, "train", tokenizer)
    val_dataset, val_pair_count = build_dataset(config, "val", tokenizer)
    print(f"训练 patch 数: {len(train_dataset)}，验证 patch 数: {len(val_dataset)}")

    resume_checkpoint = resolve_path(config.get("training", {}).get("resume_checkpoint"))
    if resume_checkpoint:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"resume_checkpoint 不存在: {resume_checkpoint}")
        print(f"从已有模型参数继续训练: {resume_checkpoint}")
        model = QwenGeoGenerator.from_pretrained(
            str(resume_checkpoint),
            qwen_model_path_override=config["model"].get("name"),
        ).to(device)
    else:
        model = build_qwen_only_model(config, tokenizer, train_feature_size, device)
    model = maybe_wrap_data_parallel(model, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        collate_fn=GeoMapDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("val_batch_size", config["training"]["batch_size"])),
        shuffle=False,
        collate_fn=GeoMapDataset.collate_fn,
    )

    amp_enabled, amp_dtype, use_grad_scaler = get_amp_settings(config)
    scaler = build_grad_scaler(use_grad_scaler)
    optimizer = build_optimizer(model, config)
    grad_accum_steps = int(config.get("training", {}).get("grad_accum_steps", 1))
    max_grad_norm = float(config.get("training", {}).get("max_grad_norm", 0.0))
    step_save_every = int(config.get("training", {}).get("save_every_n_steps", 0))

    best_val_loss = float("inf")
    history = []
    global_step = 0
    num_epochs = int(config["training"]["epochs"])

    def save_latest_step_checkpoint(step: int):
        save_checkpoint(
            model,
            project_root / "checkpoints" / "latest_step",
            config,
            tokenizer,
            {
                "global_step": step,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "train_pairs": train_pair_count,
                "val_pairs": val_pair_count,
                "device": str(device),
                "mixed_precision": str(config.get("training", {}).get("mixed_precision", "auto")),
                "gradient_checkpointing": bool(config.get("training", {}).get("gradient_checkpointing", False)),
                "grad_accum_steps": grad_accum_steps,
            },
        )
        print(f"已保存 latest_step checkpoint，global_step={step}")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, global_step = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            global_step=global_step,
            step_save_every=step_save_every,
            step_save_callback=save_latest_step_checkpoint,
        )
        with torch.no_grad():
            val_loss, global_step = run_epoch(
                model,
                val_loader,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                global_step=global_step,
            )
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "global_step": global_step})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                project_root / "checkpoints" / "best_model",
                config,
                tokenizer,
                {
                    "best_epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "global_step": global_step,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "train_pairs": train_pair_count,
                    "val_pairs": val_pair_count,
                    "device": str(device),
                    "mixed_precision": str(config.get("training", {}).get("mixed_precision", "auto")),
                    "gradient_checkpointing": bool(config.get("training", {}).get("gradient_checkpointing", False)),
                    "grad_accum_steps": grad_accum_steps,
                },
            )
            print(f"已保存 best_model，val_loss={best_val_loss:.4f}")

    save_checkpoint(
        model,
        project_root / "checkpoints" / "final_model",
        config,
        tokenizer,
        {
            "best_val_loss": best_val_loss,
            "history": history,
            "global_step": global_step,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "train_pairs": train_pair_count,
            "val_pairs": val_pair_count,
            "device": str(device),
            "epochs": num_epochs,
            "mixed_precision": str(config.get("training", {}).get("mixed_precision", "auto")),
            "gradient_checkpointing": bool(config.get("training", {}).get("gradient_checkpointing", False)),
            "grad_accum_steps": grad_accum_steps,
        },
    )
    print("训练完成，已保存 final_model。")


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoProcessor

class QwenGeoGenerator(nn.Module):
    """基于Qwen2.5的地理生成器"""

    def __init__(self, qwen_model_path, dino_model_path):
        super().__init__()

        # 加载Qwen2.5-VL
        self.qwen_model = AutoModel.from_pretrained(qwen_model_path)
        self.processor = AutoProcessor.from_pretrained(qwen_model_path)

        # 加载DINOv3作为视觉编码器
        self.dino_model = AutoModel.from_pretrained(dino_model_path)
        

        # 获取视觉特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.dino_model(dummy_input)
            if hasattr(dummy_output, 'last_hidden_state'):
                visual_dim = dummy_output.last_hidden_state.shape[-1]
            else:
                visual_dim = dummy_output.pooler_output.shape[-1]

        # 投影层
        self.visual_proj = nn.Linear(visual_dim, self.qwen_model.config.hidden_size)

    def forward(self, images, input_ids=None, attention_mask=None, labels=None):
        # DINO特征提取
        with torch.no_grad():
            visual_features = self.dino_model(images).last_hidden_state

        # 投影到Qwen维度
        visual_features = self.visual_proj(visual_features)

        # 纯视觉生成模式：只用视觉特征生成文本
        if input_ids is None:
            outputs = self.qwen_model(
                inputs_embeds=visual_features,
                attention_mask=attention_mask,
                labels=labels  # 用于计算损失
            )
        else:
            # 条件生成模式：视觉特征 + 文本输入
            outputs = self.qwen_model(
                inputs_embeds=visual_features,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        return outputs

    def generate(self, images, prompt):
        """生成地理序列"""
        # 编码视觉特征
        with torch.no_grad():
            visual_features = self.dino_model(images).last_hidden_state

        visual_features = self.visual_proj(visual_features)

        # 编码文本prompt
        inputs = self.processor(text=prompt, return_tensors="pt")

        # 生成
        generated_ids = self.qwen_model.generate(
            inputs_embeds=visual_features,
            input_ids=inputs['input_ids'],
            max_length=512
        )

        return generated_ids

    @classmethod
    def from_pretrained(cls, model_path):
        """
        直接加载完整的训练后模型

        Args:
            model_path: 训练后模型的目录路径

        Returns:
            QwenGeoGenerator: 加载了训练权重的模型实例
        """
        # 加载完整的模型状态字典
        model_file = os.path.join(model_path, "pytorch_model.bin")
        config_file = os.path.join(model_path, "config.json")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # 首先需要创建模型实例，但这里有个问题：
        # 我们需要预训练模型路径来初始化，但训练后模型应该已经包含所有权重
        # 这里需要配置文件或者其他方式获取原始模型路径

        # 临时解决方案：从config.json读取原始模型路径
        if os.path.exists(config_file):
            import json
            with open(config_file) as f:
                config = json.load(f)
            qwen_path = config.get('qwen_model_path', 'Qwen/Qwen2.5-VL-7B-Instruct')
            dino_path = config.get('dino_model_path', 'facebook/dinov2-base')
        else:
            # 默认使用HuggingFace模型
            qwen_path = 'Qwen/Qwen2.5-VL-7B-Instruct'
            dino_path = 'facebook/dinov2-base'

        # 创建模型实例
        instance = cls(qwen_path, dino_path)

        # 加载训练后的权重
        state_dict = torch.load(model_file, map_location='cpu')
        instance.load_state_dict(state_dict)

        return instance
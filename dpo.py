#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from transformers.trainer_callback import TrainerCallback

# 导入你的自定义模块，如果有的话
# from utils import MyTrainerCallback

# ============================
# DPOArguments 数据类定义
# ============================

@dataclass
class DPOArguments:
    """
    DPO 训练的参数配置
    """
    # SFT模型的检查点目录
    model_checkpoint: str = field(default="model_save/sft/")
    # DPO训练数据文件
    train_file: str = field(default="./data/dpo_train.json")
    # 可选的验证集数据文件
    eval_file: Optional[str] = field(default=None)
    # DPO训练后的模型保存目录
    output_dir: str = field(default="model_save/dpo/")
    # 日志目录
    logging_dir: str = field(default="./logs_dpo/")
    # 训练epoch数
    num_train_epochs: int = field(default=3)
    # 每个设备的训练batch大小
    per_device_train_batch_size: int = field(default=4)
    # 梯度累积步数
    gradient_accumulation_steps: int = field(default=8)
    # 学习率
    learning_rate: float = field(default=5e-5)
    # 最大序列长度
    max_seq_length: int = field(default=1024)
    # 是否使用 bf16
    bf16: bool = field(default=True)
    # DPO的超参数，可以根据需要调整
    dpo_beta: float = field(default=1.0)
    # 保存策略
    save_steps: int = field(default=500)
    # 日志记录步数
    logging_steps: int = field(default=100)

# ============================
# DPO Dataset 定义
# ============================

class DPODataset(Dataset):
    """
    DPO 数据集，假设每个样本包含两个候选回答及其偏好关系
    """
    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = []
        if data_path and os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    example = json.loads(line)
                    self.data.append(example)
        else:
            raise FileNotFoundError(f"DPO training data file {data_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        example = self.data[idx]
        instruction = example.get("instruction", "")
        choice_a = example.get("choice_a", "")
        choice_b = example.get("choice_b", "")
        preferred = example.get("preferred", "A")  # "A" 或 "B"

        # 根据偏好关系确定哪些回答被视为更好
        if preferred == "A":
            preferred_choice = choice_a
            other_choice = choice_b
        elif preferred == "B":
            preferred_choice = choice_b
            other_choice = choice_a
        else:
            raise ValueError(f"Invalid preferred value: {preferred}")

        # 构建 prompt
        prompt = f"用户：{instruction}\n\n助手："

        # 对两个候选回答进行编码
        # 选择更好的回答作为正样本，其他的作为负样本
        # 这里采用排序的方式进行简单实现
        # 具体的DPO损失函数需要根据论文或实现细节进行调整

        # 编码输入
        encoded_prompt = self.tokenizer(
            prompt,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        # 编码选择A
        encoded_a = self.tokenizer(
            preferred_choice,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # 编码选择B
        encoded_b = self.tokenizer(
            other_choice,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # 拼接 prompt + choice
        input_ids_a = torch.cat([encoded_prompt["input_ids"], encoded_a["input_ids"]], dim=1).squeeze(0)
        attention_mask_a = torch.cat([encoded_prompt["attention_mask"], encoded_a["attention_mask"]], dim=1).squeeze(0)

        input_ids_b = torch.cat([encoded_prompt["input_ids"], encoded_b["input_ids"]], dim=1).squeeze(0)
        attention_mask_b = torch.cat([encoded_prompt["attention_mask"], encoded_b["attention_mask"]], dim=1).squeeze(0)

        return {
            "input_ids_a": input_ids_a,
            "attention_mask_a": attention_mask_a,
            "input_ids_b": input_ids_b,
            "attention_mask_b": attention_mask_b,
            "preferred": preferred,  # "A" 或 "B"
        }

# ============================
# DPO Loss 定义
# ============================

class DPOLoss(torch.nn.Module):
    """
    简单的 DPO 损失函数实现
    这里假设更好的选择的 logit 应该更高
    实际的 DPO 损失可能更复杂，需参考相关论文或实现
    """
    def __init__(self, model, tokenizer, dpo_beta: float = 1.0):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dpo_beta = dpo_beta

    def forward(self, batch):
        # 获取输入
        input_ids_a = batch["input_ids_a"].to(self.model.device)
        attention_mask_a = batch["attention_mask_a"].to(self.model.device)
        input_ids_b = batch["input_ids_b"].to(self.model.device)
        attention_mask_b = batch["attention_mask_b"].to(self.model.device)
        preferred = batch["preferred"]

        # 计算 logits
        outputs_a = self.model(input_ids=input_ids_a, attention_mask=attention_mask_a)
        logits_a = outputs_a.logits  # (batch_size, seq_len, vocab_size)

        outputs_b = self.model(input_ids=input_ids_b, attention_mask=attention_mask_b)
        logits_b = outputs_b.logits  # (batch_size, seq_len, vocab_size)

        # 假设最后一个token的logit代表回答的质量
        # 这里采用最后一个token的logit进行比较
        # 你可以根据实际情况调整，比如使用整个回答的累积概率

        # 获取最后一个token的logit
        last_token_logit_a = logits_a[:, -1, :].gather(1, input_ids_a[:, -1].unsqueeze(-1)).squeeze(-1)  # (batch_size)
        last_token_logit_b = logits_b[:, -1, :].gather(1, input_ids_b[:, -1].unsqueeze(-1)).squeeze(-1)  # (batch_size)

        # 根据偏好关系计算损失
        # 如果 preferred 是 A，则希望 logits_a > logits_b
        # 反之亦然

        # 计算差值
        delta = last_token_logit_a - last_token_logit_b  # (batch_size)

        # 根据偏好关系调整差值
        # 如果 preferred 是 A，delta 应该正；如果是 B，delta 应该负
        labels = torch.where(preferred == "A", torch.ones_like(delta), -torch.ones_like(delta))
        loss = F.softplus(-self.dpo_beta * labels * delta).mean()

        return loss

# ============================
# Main 函数定义
# ============================

def main():
    # ========== 解析自定义参数 ==========
    dpo_args = DPOArguments()

    # ========== 加载配置、模型、分词器 ==========
    print("[INFO] Loading configuration...")
    config = AutoConfig.from_pretrained(dpo_args.model_checkpoint)

    print("[INFO] Loading model from checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(
        dpo_args.model_checkpoint,
        config=config,
    )
    if dpo_args.bf16:
        model = model.to(torch.bfloat16)
    else:
        model = model.half()

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(dpo_args.model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========== 准备数据集 ==========
    print("[INFO] Loading DPO training dataset...")
    train_dataset = DPODataset(
        data_path=dpo_args.train_file,
        tokenizer=tokenizer,
        max_seq_length=dpo_args.max_seq_length,
    )
    eval_dataset = None
    if dpo_args.eval_file:
        print("[INFO] Loading DPO evaluation dataset...")
        eval_dataset = DPODataset(
            data_path=dpo_args.eval_file,
            tokenizer=tokenizer,
            max_seq_length=dpo_args.max_seq_length,
        )

    # ========== 定义 DataCollator ==========
    # 由于 DPO 需要同时处理两个选择，我们需要自定义 data collator
    class DPODataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            batch = {}
            # 分别处理 choice A 和 choice B
            input_ids_a = [f["input_ids_a"] for f in features]
            attention_mask_a = [f["attention_mask_a"] for f in features]
            input_ids_b = [f["input_ids_b"] for f in features]
            attention_mask_b = [f["attention_mask_b"] for f in features]
            preferred = [f["preferred"] for f in features]

            # 填充
            batch["input_ids_a"] = torch.stack([F.pad(x, (0, dpo_args.max_seq_length - x.size(0)), value=tokenizer.pad_token_id) if x.size(0) < dpo_args.max_seq_length else x[:dpo_args.max_seq_length] for x in input_ids_a])
            batch["attention_mask_a"] = torch.stack([F.pad(x, (0, dpo_args.max_seq_length - x.size(0)), value=0) if x.size(0) < dpo_args.max_seq_length else x[:dpo_args.max_seq_length] for x in attention_mask_a])
            batch["input_ids_b"] = torch.stack([F.pad(x, (0, dpo_args.max_seq_length - x.size(0)), value=tokenizer.pad_token_id) if x.size(0) < dpo_args.max_seq_length else x[:dpo_args.max_seq_length] for x in input_ids_b])
            batch["attention_mask_b"] = torch.stack([F.pad(x, (0, dpo_args.max_seq_length - x.size(0)), value=0) if x.size(0) < dpo_args.max_seq_length else x[:dpo_args.max_seq_length] for x in attention_mask_b])
            batch["preferred"] = torch.tensor([1 if p == "A" else 0 for p in preferred], dtype=torch.float)

            return batch

    data_collator = DPODataCollator(tokenizer=tokenizer)

    # ========== 定义 DPO 损失 ==========
    dpo_loss = DPOLoss(model=model, tokenizer=tokenizer, dpo_beta=dpo_args.dpo_beta)

    # ========== 构建 Trainer ==========
    training_args = TrainingArguments(
        output_dir=dpo_args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=dpo_args.num_train_epochs,
        per_device_train_batch_size=dpo_args.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_args.gradient_accumulation_steps,
        learning_rate=dpo_args.learning_rate,
        logging_dir=dpo_args.logging_dir,
        logging_steps=dpo_args.logging_steps,
        save_steps=dpo_args.save_steps,
        save_strategy="steps",
        evaluation_strategy="no" if eval_dataset is None else "steps",
        eval_steps=dpo_args.save_steps if eval_dataset else None,
        bf16=dpo_args.bf16,
        # fp16=True,  # 如果不使用 bf16，可以启用 fp16
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # 使用自定义的 DPO 损失
        compute_loss=lambda model, inputs, return_outputs=False: dpo_loss(inputs),
        # callbacks=[MyTrainerCallback()]  # 如果有自定义 callback，可以添加
    )

    # ========== 开始训练 ==========
    print("[INFO] Starting DPO training...")
    trainer.train()

    # ========== 保存模型 ==========
    print("[INFO] Saving DPO fine-tuned model...")
    trainer.save_model(dpo_args.output_dir)
    print("[INFO] DPO training completed.")

if __name__ == "__main__":
    main()

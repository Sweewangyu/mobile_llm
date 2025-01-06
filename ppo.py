#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
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
from torch.optim import AdamW
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 如果你有自定义的回调，可以导入
# from utils import MyTrainerCallback

# ============================
# PPOArguments 数据类定义
# ============================

@dataclass
class PPOArguments:
    """
    PPO 训练的参数配置
    """
    # SFT模型的检查点目录
    model_checkpoint: str = field(default="model_save/sft/")
    # PPO训练数据文件（假设是一个 JSON 文件，每行一个对话样本）
    train_file: str = field(default="./data/ppo_train.json")
    # 可选的验证集数据文件
    eval_file: Optional[str] = field(default=None)
    # PPO训练后的模型保存目录
    output_dir: str = field(default="model_save/ppo/")
    # 日志目录
    logging_dir: str = field(default="./logs_ppo/")
    # 训练epoch数
    num_train_epochs: int = field(default=3)
    # 每个设备的训练batch大小
    per_device_train_batch_size: int = field(default=2)
    # 梯度累积步数
    gradient_accumulation_steps: int = field(default=8)
    # 学习率
    learning_rate: float = field(default=5e-5)
    # 最大序列长度
    max_seq_length: int = field(default=1024)
    # 是否使用 bf16
    bf16: bool = field(default=True)
    # PPO 的超参数
    ppo_epochs: int = field(default=4)  # PPO 的迭代次数
    ppo_clip: float = field(default=0.2)  # PPO 的裁剪参数
    # 保存策略
    save_steps: int = field(default=500)
    # 日志记录步数
    logging_steps: int = field(default=100)

# ============================
# PPO Dataset 定义
# ============================

class PPODataset(Dataset):
    """
    PPO 数据集，假设每个样本包含：
    {
        "instruction": "用户指令或对话输入",
        "response": "模型生成的回复",
        "reward": 1.0  // 或其他数值，表示回复的质量
    }
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
            raise FileNotFoundError(f"PPO training data file {data_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        example = self.data[idx]
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        reward = example.get("reward", 0.0)  # 你可以根据需要调整 reward 的计算方式

        # 构建 prompt
        prompt = f"用户：{instruction}\n\n助手："

        # 编码 prompt 和 response
        encoded_prompt = self.tokenizer(
            prompt,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        encoded_response = self.tokenizer(
            response,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # 拼接 prompt + response
        input_ids = torch.cat([encoded_prompt["input_ids"], encoded_response["input_ids"]], dim=1).squeeze(0)
        attention_mask = torch.cat([encoded_prompt["attention_mask"], encoded_response["attention_mask"]], dim=1).squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "reward": torch.tensor(reward, dtype=torch.float),
        }

# ============================
# PPO Loss 定义
# ============================
import torch
import torch.nn.functional as F
from torch import nn


class PPOLoss(nn.Module):
    def __init__(self, model, ppo_clip: float = 0.2, value_coef=0.5, entropy_coef=0.01):
        super().__init__()
        self.model = model
        self.ppo_clip = ppo_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        # 假设 model 同时输出策略与价值，也可以拆成两个网络

    def forward(self, batch):
        # batch 需要额外包含旧策略log_prob 和 目标价值/回报 或者用于计算GAE的序列
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        old_log_probs = batch["old_log_probs"].to(self.model.device)  # 旧策略
        advantages = batch["advantages"].to(self.model.device)
        returns = batch["returns"].to(self.model.device)  # MC回报或lambda-return

        # 1) 前向传播，拿到新的logits和价值
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        values = outputs.value  # [batch_size, seq_len] 或 [batch_size] 根据实现方式

        # 2) 计算新策略的 log_prob
        #   - 如果是 token-level PPO，需要对每一步动作分别 gather
        #   - 也可能只对最后一个 token gather
        new_log_probs = F.log_softmax(logits, dim=-1)
        new_log_probs = new_log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

        # 3) 计算 ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 4) 计算策略损失 (CLIP)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 5) 计算价值损失
        #    returns 与 values 维度对齐；如果是 seq-level 需要额外处理
        value_loss = F.mse_loss(values, returns)

        # 6) 计算熵损失 (鼓励探索)
        #    先对 logits 做 softmax 再求熵
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1).mean()

        # 7) 总损失
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return loss


class PPOLoss(torch.nn.Module):
    """
    PPO 损失函数的简单实现
    """
    def __init__(self, model, tokenizer, ppo_clip: float = 0.2):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.ppo_clip = ppo_clip
        self.gamma = 0.99  # 折扣因子，根据需要调整
        self.lam = 0.95  # GAE 参数，根据需要调整

    def forward(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)
        rewards = batch["reward"].to(self.model.device)

        # 获取当前策略的 log 概率
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        # 这里简单地取最后一个 token 的 log 概率作为示例
        log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1)

        # 计算优势函数，这里简化为 rewards
        advantages = rewards

        # 计算策略损失
        ratio = torch.exp(log_probs)  # 假设旧策略的概率已经包含在 log_probs 中
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 你可以添加价值函数的损失或熵正则化等，根据需要调整
        return policy_loss

# ============================
# Main 函数定义
# ============================

def main():
    # ========== 解析自定义参数 ==========
    ppo_args = PPOArguments()

    # ========== 加载配置、模型、分词器 ==========
    print("[INFO] Loading configuration...")
    config = AutoConfig.from_pretrained(ppo_args.model_checkpoint)

    print("[INFO] Loading model from checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(
        ppo_args.model_checkpoint,
        config=config,
    )
    if ppo_args.bf16:
        model = model.to(torch.bfloat16)
    else:
        model = model.half()

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ppo_args.model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========== 准备数据集 ==========
    print("[INFO] Loading PPO training dataset...")
    train_dataset = PPODataset(
        data_path=ppo_args.train_file,
        tokenizer=tokenizer,
        max_seq_length=ppo_args.max_seq_length,
    )
    eval_dataset = None
    if ppo_args.eval_file:
        print("[INFO] Loading PPO evaluation dataset...")
        eval_dataset = PPODataset(
            data_path=ppo_args.eval_file,
            tokenizer=tokenizer,
            max_seq_length=ppo_args.max_seq_length,
        )

    # ========== 定义 DataCollator ==========
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # ========== 定义 PPO 损失 ==========
    ppo_loss = PPOLoss(model=model, tokenizer=tokenizer, ppo_clip=ppo_args.ppo_clip)

    # ========== 构建 Trainer ==========
    training_args = TrainingArguments(
        output_dir=ppo_args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=ppo_args.num_train_epochs,
        per_device_train_batch_size=ppo_args.per_device_train_batch_size,
        gradient_accumulation_steps=ppo_args.gradient_accumulation_steps,
        learning_rate=ppo_args.learning_rate,
        logging_dir=ppo_args.logging_dir,
        logging_steps=ppo_args.logging_steps,
        save_steps=ppo_args.save_steps,
        save_strategy="steps",
        evaluation_strategy="no" if eval_dataset is None else "steps",
        eval_steps=ppo_args.save_steps if eval_dataset else None,
        bf16=ppo_args.bf16,
        # fp16=True,  # 如果不使用 bf16，可以启用 fp16
        report_to=["tensorboard"],
    )

    optimizer = AdamW(model.parameters(), lr=ppo_args.learning_rate)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # PPO 损失管理自己的优化器
        # callbacks=[MyTrainerCallback()]  # 如果有自定义 callback，可以添加
    )

    # ========== 开始 PPO 训练 ==========
    print("[INFO] Starting PPO training...")
    for epoch in range(ppo_args.num_train_epochs):
        print(f"[INFO] Epoch {epoch+1}/{ppo_args.num_train_epochs}")
        for step, batch in enumerate(tqdm(train_dataset)):
            loss = ppo_loss(batch)
            loss.backward()
            if (step + 1) % ppo_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        # 你可以在每个 epoch 后保存模型
        trainer.save_model(os.path.join(ppo_args.output_dir, f"epoch-{epoch+1}"))
        print(f"[INFO] Saved model for epoch {epoch+1}")

    # ========== 保存模型 ==========
    print("[INFO] Saving PPO fine-tuned model...")
    trainer.save_model(ppo_args.output_dir)
    print("[INFO] PPO training completed.")

if __name__ == "__main__":
    main()

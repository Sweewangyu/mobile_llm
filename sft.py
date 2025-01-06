#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pdb

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    default_data_collator, Qwen2ForCausalLM, Qwen2Config, AdamW
)
# 如果你有自己封装的 MyTrainerCallback，可以进行导入
# from utils import MyTrainerCallback

from chatglm2tokenizer.tokenization_chatglm import ChatGLMTokenizer

@dataclass
class SFTArguments:
    """
    根据实际需求调整，命令行形式可使用 HfArgumentParser 来解析。
    这里仅作示例。
    """
    # 预训练模型所在目录，checkpoint
    model_checkpoint: str = field(default="model_save/pre/checkpoint-2600")
    # SFT后模型的保存路径
    output_dir: str = field(default="model_save/sft/")
    # 日志保存目录
    logging_dir: str = field(default="./logs_sft/")
    # 微调所需训练数据文件（示例假设使用 json 格式）
    train_file: str = field(default="./data/alpaca_data_51k.json")
    # 验证集数据文件（可选）
    eval_file: Optional[str] = field(default=None)
    # 训练 epoch 数量
    num_train_epochs: int = field(default=1)
    # 每个 batch 的大小
    per_device_train_batch_size: int = field(default=1)
    # 梯度累加步数
    gradient_accumulation_steps: int = field(default=1)
    # 序列最大长度
    max_seq_length: int = field(default=512)
    # 初始学习率
    learning_rate: float = field(default=2e-4)
    # 是否使用 bf16
    bf16: bool = field(default=True)


class SFTDataset(Dataset):
    """
    一个简单的 SFT 数据集示例，假设 train_file/eval_file 中每一条数据包含：
    {
      "instruction": "用户指令或对话的输入部分",
      "input": "用户输入(可选，如果 instruction 不够描述任务)",
      "output": "模型需要生成的目标输出"
    }
    你可以根据自己的对话格式来改写这部分。
    """

    def __init__(self, data_path: str, tokenizer: ChatGLMTokenizer, max_seq_length: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = []
        if data_path and os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)  # 这里一次性读入整个数组
                for example in data_list:
                    self.data.append(example)
        else:
            print(f"[Warning] data_path: {data_path} not found or is not a file.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        example = self.data[idx]
        instruction = example.get("instruction", "")
        user_input = example.get("input", "")
        output = example.get("output", "")

        # 根据对话场景自定义 prompt 的拼接逻辑
        # 示例：<instruction + input> -> 需要生成 <output>
        if user_input:
            prompt = f"用户：{instruction}\n{user_input}\n\n助手："
        else:
            prompt = f"用户：{instruction}\n\n助手："

        # 将 prompt+output 转成模型可理解的输入
        # 对于 ChatGLM，为了区分编码输入和目标输出，可以在生成 label 时进行适当 mask
        # 这里只是最简示例，具体可再根据 ChatGLM 的做法来进行定制
        tokenized_prompt = self.tokenizer(
            prompt,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        tokenized_output = self.tokenizer(
            output,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,  # output可以按需加特殊符号
        )

        # ChatGLM 通常需要拼接 prompt + output 共同作为模型输入， 并用 label mask 来区分
        # 这里只演示一种简单形式：把 prompt 的 token 作为上下文，output 作为需预测部分
        input_ids = torch.cat([tokenized_prompt["input_ids"], tokenized_output["input_ids"]], dim=1).long()
        attention_mask = torch.cat([tokenized_prompt["attention_mask"], tokenized_output["attention_mask"]],dim=1).long()

        labels = input_ids.clone()
        # 将 prompt 部分标记为 -100，避免计算其 loss，仅对 output 部分计算 loss
        prompt_len = tokenized_prompt["input_ids"].size(1)
        labels[:, :prompt_len] = -100
        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }


def main():
    # ========== 解析自定义参数 ==========
    sft_args = SFTArguments()

    # ========== 加载配置、模型、分词器 ==========
    print("[INFO] Loading llm config...")
    config = Qwen2Config.from_pretrained(sft_args.model_checkpoint)

    print("[INFO] Loading llm model from checkpoint...")
    model = Qwen2ForCausalLM.from_pretrained(
        sft_args.model_checkpoint,
        config=config,
    )
    # 根据需要指定 float16/bfloat16
    if sft_args.bf16:
        model = model.bfloat16()
    else:
        model = model.half()

    print("[INFO] Loading ChatGLM tokenizer...")
    tokenizer = ChatGLMTokenizer.from_pretrained(sft_args.model_checkpoint)

    # ========== 准备数据集 ==========
    print("[INFO] Loading SFT dataset...")
    train_dataset = SFTDataset(
        data_path=sft_args.train_file,
        tokenizer=tokenizer,
        max_seq_length=sft_args.max_seq_length,
    )
    eval_dataset = None
    if sft_args.eval_file:
        eval_dataset = SFTDataset(
            data_path=sft_args.eval_file,
            tokenizer=tokenizer,
            max_seq_length=sft_args.max_seq_length,
        )

    # ========== 构造 DataCollator ==========
    # 对于 Seq2Seq 任务，可以使用 DataCollatorForSeq2Seq 或自定义
    # 这里也可以直接使用 default_data_collator，但 ChatGLM 通常需要特殊处理
    # 若不涉及特别处理，可以先用默认
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # ========== 构建 Trainer ==========
    training_args = TrainingArguments(
        output_dir=sft_args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=sft_args.num_train_epochs,
        per_device_train_batch_size=sft_args.per_device_train_batch_size,
        gradient_accumulation_steps=sft_args.gradient_accumulation_steps,
        learning_rate=sft_args.learning_rate,
        logging_dir=sft_args.logging_dir,
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        bf16=sft_args.bf16,
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[MyTrainerCallback()]
    )

    # ========== 开始训练 ==========
    print("[INFO] Start SFT training...")
    trainer.train()

    # ========== 保存模型 ==========
    print("[INFO] Saving fine-tuned model...")
    trainer.save_model(sft_args.output_dir)
    print("[INFO] SFT training finished.")


if __name__ == "__main__":
    main()

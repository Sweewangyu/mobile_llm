#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse
import traceback
from chatglm2tokenizer.configuration_chatglm import ChatGLMConfig
from chatglm2tokenizer.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2tokenizer.tokenization_chatglm import ChatGLMTokenizer

def load_model(model_path: str, device: torch.device):
    """
    加载微调后的模型和分词器。
    """
    print("[INFO] 加载模型配置...")
    config = ChatGLMConfig.from_pretrained(model_path)

    print("[INFO] 加载微调后的模型...")
    model = ChatGLMForConditionalGeneration.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()  # 设置为评估模式

    print("[INFO] 加载分词器...")
    tokenizer = ChatGLMTokenizer.from_pretrained(model_path)

    # 确保特殊符号已设置
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<eos>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    if tokenizer.unk_token is None:
        tokenizer.unk_token = "<unk>"

    # 打印词汇表大小
    print(f"[INFO] Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"[INFO] Model vocab size: {model.config.vocab_size}")

    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, device: torch.device, max_length: int = 512, temperature: float = 0.7, top_p: float = 0.9):
    """
    生成模型回复。
    """
    if prompt.strip() == "":
        print("[WARNING] 输入内容为空，请输入有效的对话内容。")
        return ""

    formatted_prompt = f"用户：{prompt}\n\n助手："

    # 分词
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=True
    ).to(device)

    # 检查 input_ids 是否超出词汇表范围
    vocab_size = model.config.vocab_size
    input_ids = inputs["input_ids"]
    if (input_ids >= vocab_size).any():
        print("[ERROR] 输入的 `input_ids` 中存在超出词汇表范围的值。")
        print(f"input_ids: {input_ids}")
        return ""

    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[1] + max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # 检查 outputs 的类型和内容
    if isinstance(outputs, (tuple, list)):
        generated_ids = outputs[0]
    else:
        generated_ids = outputs

    # 解码回复部分
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    response = generated_text[len(formatted_prompt):].strip()
    return response

def main():
    parser = argparse.ArgumentParser(description="测试SFT微调后的ChatGLM模型对话功能")
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_save/sft/checkpoint-799",
        help="微调后模型的保存目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用的设备（cuda或cpu）"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="生成回复的最大长度"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样的温度参数"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="nucleus采样的top_p参数"
    )

    args = parser.parse_args()

    # 检查设备是否可用
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA不可用，使用CPU进行计算。")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # 加载模型和分词器
    model, tokenizer = load_model(args.model_path, device)

    print("===== ChatGLM SFT 模型测试 =====")
    print("输入 'exit' 或 'quit' 以结束对话。")
    while True:
        try:
            user_input = input("用户：")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("结束对话。")
                break

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                device=device,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(f"助手：{response}")

        except KeyboardInterrupt:
            print("\n结束对话。")
            break
        except Exception as e:
            print(f"[ERROR] 发生错误: {e}")
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()

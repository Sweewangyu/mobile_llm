import pdb

from torch.optim import AdamW
from dataclasses import dataclass, field
from utils import *
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator
)
from chatglm2tokenizer.tokenization_chatglm import ChatGLMTokenizer
from chatglm4.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm4.configuration_chatglm import ChatGLMConfig
# 保持使用 ChatGLM 的分词器
tokenizer = ChatGLMTokenizer.from_pretrained('chatglm2tokenizer/tokenizer.model')
# 2. 定义训练文件和参数
TRAIN_FILES = [
    "/home/wangyu/data/baidubaike/baidubaike_563w_1.bin",
    "/home/wangyu/data/baidubaike/baidubaike_563w_2.bin",
    "/home/wangyu/data/baidubaike/baidubaike_563w_3.bin",
    "/home/wangyu/data/baidubaike/baidubaike_563w_4.bin",
    "/home/wangyu/data/baidubaike/baidubaike_563w_5.bin",
]

@dataclass
class PretrainArguments:
    model_save_dir: str = "./model_save/pre/"
    logs_dir: str = "./logs/"
    train_files: list = field(default_factory=lambda: TRAIN_FILES)
    max_seq_len: int = 512

pretrain_args = PretrainArguments()

# 3. 初始化 DeepseekV3 的配置和模型
config = ChatGLMConfig.from_pretrained("chatglm4/config.json")

model = ChatGLMForConditionalGeneration(config,empty_init=False).bfloat16().cuda()
check_and_initialize_parameters(model)
# 创建训练数据集
train_dataset = PretrainDataset(pretrain_args.train_files, max_length=pretrain_args.max_seq_len, memmap=True)
# 打印模型参数数量
# sample = train_dataset[0]
# print("Sample数据结构：", sample)
# decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
# print("解码后的文本：\n", decoded_text)
model_size = sum(t.numel() for t in model.parameters())
print(f"DeepseekV3 size: {model_size / 1024 ** 2:.1f}M parameters")
# 初始化参数（确保所有参数正确初始化）


# 定义自定义的 Trainer 回调（假设你已经定义了 MyTrainerCallback）
my_trainer_callback = MyTrainerCallback()

# 4. 定义训练参数
args = TrainingArguments(
    output_dir=pretrain_args.model_save_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=32,
    max_grad_norm=1.0 ,
    num_train_epochs=1,
    save_steps=200,
    # lr_scheduler_type='cosine',
    # learning_rate=1e-3,
    # optim='adamw_torch',
    save_strategy="steps",
    save_total_limit=2,
    logging_steps=5,
    log_level="info",
    logging_first_step=True,
    bf16=True
)

# # 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
total_steps = compute_total_steps(train_dataset, args)
lr_scheduler = custom_lr_scheduler(optimizer, total_steps)

# 5. 初始化 Trainer
trainer = Trainer(
    model=model,
    tokenizer=None,
    args=args,
    optimizers=(optimizer, lr_scheduler),
    data_collator=default_data_collator,
    train_dataset=train_dataset,
    callbacks=[my_trainer_callback],
)

# 6. 开始训练
trainer.train(
   # resume_from_checkpoint='../../hy-tmp/model_save/pre/checkpoint-1693'  # 可选，断点续训
)

# 7. 保存模型
trainer.save_model(pretrain_args.model_save_dir)

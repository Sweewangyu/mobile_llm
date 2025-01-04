import numpy as np
import torch
import os
from transformers import get_wsd_schedule
import matplotlib.pyplot as plt
def check_and_initialize_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Reinitializing {name} due to NaNs")
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import os
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import numpy as np
import torch
from torch.utils.data import Dataset


import os
import numpy as np
import torch
from torch.utils.data import Dataset
import bisect

class PretrainDataset(Dataset):
    def __init__(self, data_path_lst, max_length, memmap=True):
        super().__init__()
        self.max_length = max_length
        self.memmap = memmap

        if self.memmap:
            self.memmaps = []
            self.cumulative_sizes = []
            total_samples = 0

            for data_path in data_path_lst:
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Data file not found: {data_path}")

                file_size_bytes = os.path.getsize(data_path)
                dtype = np.uint16
                itemsize = np.dtype(dtype).itemsize
                flen = file_size_bytes // itemsize

                num_samples = flen // self.max_length
                if num_samples == 0:
                    print(f"警告: 文件 {data_path} 的大小不足以形成一个样本，跳过。")
                    continue  # 跳过无法形成一个完整样本的文件

                shape = (num_samples, self.max_length)
                try:
                    memmap_arr = np.memmap(
                        data_path,
                        dtype=dtype,
                        mode='r',
                        shape=shape
                    )
                except Exception as e:
                    print(f"错误: 无法加载 memmap 文件 {data_path}: {e}")
                    continue

                self.memmaps.append(memmap_arr)
                total_samples += num_samples
                self.cumulative_sizes.append(total_samples)

                print(f"已加载 memmap 文件: {data_path}，样本数: {num_samples}")

            self.total_samples = total_samples

            if self.total_samples == 0:
                raise ValueError("所有 memmap 文件均未加载任何样本。请检查数据文件和 `max_length` 参数。")

            print(f"总共加载了 {len(self.memmaps)} 个 memmap 文件，总样本数: {self.total_samples}")
            print("Data type: uint16 (memmap)")
        else:
            data_lst = []
            for data_path in data_path_lst:
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Data file not found: {data_path}")

                try:
                    with open(data_path, 'rb') as f:
                        data = np.fromfile(f, dtype=np.uint16)
                        data_lst.append(data)
                    print(f"已加载文件: {data_path}，长度: {len(data)}")
                except Exception as e:
                    print(f"错误: 无法读取文件 {data_path}: {e}")
                    continue

            if not data_lst:
                raise ValueError("所有文件均未加载任何数据。")

            data = np.concatenate(data_lst)
            total_length = (len(data) // self.max_length) * self.max_length
            data = data[:total_length]
            self.data = data.reshape(-1, self.max_length)

            print(f"memmap: {memmap}，train data.shape: {self.data.shape}")
            print("Data type:", self.data.dtype)
            print("downloading finished.....")

            if len(self.data) == 0:
                raise ValueError("Data is empty after loading.")

    def __len__(self):
        if self.memmap:
            return self.total_samples
        else:
            return self.data.shape[0]

    def __getitem__(self, index: int):
        if self.memmap:
            # 使用 bisect 找到对应的 memmap 文件
            file_idx = bisect.bisect_right(self.cumulative_sizes, index)
            if file_idx == 0:
                local_idx = index
            else:
                local_idx = index - self.cumulative_sizes[file_idx - 1]

            if file_idx >= len(self.memmaps):
                raise IndexError("Index out of range.")

            sample = self.memmaps[file_idx][local_idx]
        else:
            sample = self.data[index]

        X = sample[:-1].astype(np.int64)
        Y = sample[1:].astype(np.int64)

        return {'input_ids': torch.from_numpy(X), 'labels': torch.from_numpy(Y)}



class MyTrainerCallback(TrainerCallback):
    log_cnt = 0
    loss_values = []  # 用于存储损失值

    def on_log(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            logs=None,
            **kwargs,
    ):
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

        # 记录损失值
        if logs and "loss" in logs:
            self.loss_values.append(logs["loss"])

    def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        control.should_save = True
        return control

    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        # 在训练结束时绘制损失曲线
        self.plot_loss_curve()

    def plot_loss_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_values, label="Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig("loss_curve.png")

def compute_total_steps(train_dataset, training_args):
    total_samples = len(train_dataset)  # 数据集样本总数
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    total_steps = (total_samples // effective_batch_size) * training_args.num_train_epochs
    return total_steps

# 动态计算学习率调度器参数
def custom_lr_scheduler(optimizer, total_steps):
    num_warmup_steps = int(total_steps * 0.005)   # 预热步数为总步数的 5%
    num_stable_steps = int(total_steps * 0.495)   # 恒定阶段为总步数的 30%
    num_decay_steps = int(total_steps * 0.5)    # 衰减阶段为总步数的 60%
    print(f"Warmup Steps: {num_warmup_steps}, Stable Steps: {num_stable_steps}, Decay Steps: {num_decay_steps}")
    return get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=0.1,  # 最小学习率比例
        num_cycles=0.5     # 半周期余弦
    )

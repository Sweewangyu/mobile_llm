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

class PretrainDataset(Dataset):
    def __init__(self, data_path_lst, max_length, memmap=True):
        super().__init__()
        if memmap:
            with open(data_path_lst[0], 'rb') as f:
                data = np.fromfile(f, dtype=np.uint16, count=max_length * 10)

            # Use os.path.getsize() to get the actual file size in bytes
            file_size_bytes = os.path.getsize(data_path_lst[0])
            flen = file_size_bytes // np.dtype('uint16').itemsize
            # Load using memmap
            self.data = np.memmap(data_path_lst[0], dtype=np.uint16, shape=(flen // max_length, max_length))
        else:
            data_lst = []
            for data_path in data_path_lst:
                with open(data_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.uint16)
                    data_lst.append(data)
            data = np.concatenate(data_lst)
            data = data[:max_length * int(len(data) / max_length)]
            self.data = data.reshape(-1, max_length)

        # 打印数据集的大小和类型
        print("memmap:{} train data.shape:{}".format(memmap, self.data.shape))
        print("Data type:", self.data.dtype)
        print("downloading finished.....")

        # 验证数据集
        if len(self.data) == 0:
            raise ValueError("Data is empty after loading")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        sample = self.data[index]
        X = np.array(sample[:-1]).astype(np.int64)
        Y = np.array(sample[1:]).astype(np.int64)
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
    num_warmup_steps = int(total_steps * 0.05)   # 预热步数为总步数的 5%
    num_stable_steps = int(total_steps * 0.45)   # 恒定阶段为总步数的 30%
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

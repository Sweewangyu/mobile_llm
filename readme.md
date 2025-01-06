# 端侧小模型
本项目致力于在资源受限的端侧设备上部署和运行小型、高效的语言模型（Small Language Model, SLM），以便在移动端或物联网设备上实现自然语言处理功能。

# 0. 环境安装
以下步骤将帮助快速搭建项目所需环境：

```bash
conda create -n slm python==3.10
conda activate slm
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

- 安装flashattention（仅限Linux）
  - [FlashAttention Releases](https://github.com/Dao-AILab/flash-attention/releases)
  - 下载好后运行：
    ```bash
    pip install <下载路径>
    ```

# 1. 模型结构
本项目的核心模型基于 Qwen2 并参考了 MobileLLM 论文中的结构思想，在保证轻量化的同时兼顾了模型的表现力与推理速度。

<div align="center">
  <img src=":/2e936fa84f25426d9c00e00da21a3cbb" alt="模型结构示意图" width="50%"/>
</div>

图示仅供参考，具体实现可参见源码。

# 2. 数据来源
- 主数据集：使用 chatglm2-6b 的分词数据进行训练。
- 参考链接：[baby-llama2-chinese](https://github.com/YourRepo/baby-llama2-chinese)
- 鸣谢：特别感谢开源作者提供的数据与思路。
- 数据的加载和预处理脚本可在 `data/` 目录下找到，你可以根据自己的需求自由定制。

# 3. 训练过程
整个训练流程主要分为以下三个阶段：预训练（Pretrain）、有监督微调（SFT）与 PPO 调优（RLHF）。

## 3.1 Pretrain
在预训练阶段，将模型在大规模通用语料上进行训练，以获得基本的语言理解与生成能力：

1. 准备数据：整理并清洗文本数据，根据项目需求进行分词、切分与标注。
2. 训练脚本：运行 `scripts/pretrain.sh` 或根据你的环境执行相应的指令（参考 `scripts/` 目录下的说明）。

注意事项：
- 确保显存或内存资源充足，如有需要可尝试模型切片或梯度累积等策略。
- 注意检查训练日志，预防梯度爆炸或过拟合等常见问题。

## 3.2 SFT（Supervised Fine-Tuning）
在有监督微调阶段，我们主要针对特定任务或数据集进行模型适配与性能提升：

1. 数据准备：根据目标任务（问答、摘要、对话等）准备带有标签或目标答案的训练数据。
2. 训练脚本：可使用 `scripts/sft.sh` 内的命令，或在交互式环境下自行指定参数进行训练。

调参建议：可根据任务难度与数据规模，适当调整学习率、批大小以及训练轮次等。

## 3.3 PPO（RLHF 调优）
PPO（Proximal Policy Optimization，近端策略优化）是一种在强化学习中被广泛使用的算法，适合对语言模型进行人类偏好反馈（Human Feedback）的训练。

1. **采样（Rollout）**：语言模型根据输入问题生成回答。
2. **评估（Evaluation）**：问题与回答对通过某个函数、模型、人工反馈或它们的组合进行打分，得到一个标量值。
3. **优化（Optimization）**：使用当前模型与一个参考模型对回答进行对数似然计算，KL 散度用于约束更新幅度，防止模型梯度更新过大导致不稳定。随后，采用 PPO 算法对活跃的语言模型进行训练。

简单来说，PPO 通过不断迭代的方式改进模型：先生成回答，然后利用获得的反馈进行参数更新，并用 KL 散度作为惩罚项来保持模型更新的稳定性。

# 4. 一些技巧（Some Tricks）
## 4.1 最优学习率调度器（WSD 调度器）
学习率在训练过程中至关重要，合适的学习率调度器往往能显著提升模型性能。

- **Warmup-Stable-Decay（WSD） 调度器设计思路**：
  1. **Warmup 阶段（W）**：通过逐渐升高学习率，让模型在初期收敛更稳定，避免训练初期不稳定带来的梯度爆炸。
  2. **Stable 阶段（S）**：在较长的训练阶段保持学习率稳定，帮助模型充分挖掘数据特征。
  3. **Decay 阶段（D）**：在后期逐步降低学习率，防止过拟合并让模型在收尾阶段平稳收敛。

你可以在配置文件或脚本中调整 W, S, D 不同阶段的超参数，以找到最适合自己数据与任务的策略。

# 5. 如何使用
下载或克隆项目：

```bash
git clone https://github.com/YourRepo/slm.git
cd slm
```

环境准备（如上）。

执行脚本：

- 预训练：`bash scripts/pretrain.sh`
- 微调：`bash scripts/sft.sh`
- PPO 调优：`bash scripts/ppo.sh`

推理测试：

你可以使用 `tset.py` 在本地进行简单推理测试，或编写自己的服务接口（如 Flask 或 FastAPI）来部署模型。

# 6. 后续计划
- **优化部署**：在移动端及 IoT 设备的推理性能优化，包括量化、裁剪及蒸馏。
- **社区贡献**：欢迎更多开发者提交 Issues 与 PRs，共同完善端侧小模型的生态。

# License
本项目使用 Apache License 2.0 进行授权，具体细节请参阅 LICENSE 文件。

如有任何问题或建议，欢迎提 Issue 或 Pull Request！

此文档仍在不断完善中，如有不足之处，还请多多包涵并参与讨论。
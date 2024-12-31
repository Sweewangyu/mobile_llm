#!/usr/bin/bash
accelerate launch  --config_file accelerate_one_gpu.yaml $1
# accelerate launch $1

# 使用方式，后面接一个训练脚本即可， 如：./train.sh sft.py

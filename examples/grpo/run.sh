#!/bin/bash
# gijs/avqa-processed
export PYTHONPATH=/workspace_yuekai/asr/wenet:$PYTHONPATH
PROJECT_DIR=$(pwd)/../../
[ ! -s west ] && ln -s $PROJECT_DIR/west
[ ! -s tools ] && ln -s $PROJECT_DIR/tools
export PYTHONPATH=$PYTHONPATH:$PWD


OUT_DIR=exp/grpo
MODEL_NP=/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct
HF_DATASET_PATH=/workspace_yuekai/HF/avqa-processed

# GPU_NUM=$(nvidia-smi -L | wc -l)
GPU_NUM=8
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=32778

# export DEBUG_MODE=true
# export LOG_PATH=grpo_hf.log

torchrun --nproc_per_node=${GPU_NUM} \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    west/bin/train_grpo.py \
    --config_path conf/ds_zero3.json \
    --model_name_or_path ${MODEL_NP} \
    --out_dir ${OUT_DIR} \
    --hf_dataset_path ${HF_DATASET_PATH} \
    --use_wandb false || exit 1

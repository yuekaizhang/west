#!/bin/bash

export PYTHONPATH=/workspace_yuekai/asr/wenet:$PYTHONPATH

project_dir=$(pwd)/../../
[ ! -s west ] && ln -s $project_dir/west
[ ! -s tools ] && ln -s $project_dir/tools
export PYTHONPATH=$PYTHONPATH:$PWD

dir=exp/grpo
model_name_or_path=/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct
hf_dataset_path=/workspace_yuekai/HF/avqa-processed

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

stage=train

. tools/parse_options.sh

if [ $stage == "data" ] || [ $stage == "all" ]; then
    echo "Prepare required data"
    # gijs/avqa-processed
fi

if [ $stage == "train" ] || [ $stage == "all" ]; then
    torchrun --nproc_per_node=${num_gpus} \
        --nnodes=1 \
        --node-rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=32778 \
        west/bin/train_grpo.py \
        --deepspeed conf/ds_zero3.json \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ${dir} \
        --hf_dataset_path ${hf_dataset_path} \
        --use_wandb false || exit 1
fi

if [ $stage == "decode" ] || [ $stage == "all" ]; then
    mmau_dir=data/MMAU
    iters=(100 200 300 400 500)
    model_dir=${dir}
    batch_size=32
    for iter in ${iters[*]}; do
        model_dir=${model_dir}/checkpoint-${iter}
        out_dir=${model_dir}/test_${iter}_vllm

        python3 west/bin/decode_grpo.py \
        --model_path ${model_dir} \
        --data_file ${mmau_dir}/mmau-test-mini.json \
        --audio_dir ${mmau_dir} \
        --out_file ${out_dir}/res_mmau_mini.json \
        --batch_size ${batch_size} || exit 1
        
        python3 ${mmau_dir}/evaluation.py \
        --input ${out_dir}/res_mmau_mini.json \
        > ${dir}/eval_mmau_mini.txt || exit 1
    done
fi
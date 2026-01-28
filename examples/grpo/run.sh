#!/bin/bash

export PYTHONPATH=/workspace_yuekai/asr/wenet:$PYTHONPATH

project_dir=$(pwd)/../../
[ ! -s west ] && ln -s $project_dir/west
[ ! -s tools ] && ln -s $project_dir/tools
export PYTHONPATH=$PYTHONPATH:$PWD
#run_name=grpo_omni_3b
run_name=grpo_omni_7b
dir=exp/${run_name}
# model_name_or_path=/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct
# model_name_or_path=/workspace_yuekai/HF/Qwen2.5-Omni-3B
model_name_or_path=/workspace_yuekai/HF/Qwen2.5-Omni-7B
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
        --deepspeed conf/ds_zero1.json \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ${dir} \
        --hf_dataset_path ${hf_dataset_path} \
        --run_name ${run_name} \
        --use_wandb true || exit 1
fi

if [ $stage == "opd" ]; then
    torchrun --nproc_per_node=${num_gpus} \
        --nnodes=1 \
        --node-rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=32778 \
        west/bin/train_knowledge_distillation.py \
        --deepspeed conf/ds_zero1.json \
        --model_name_or_path ${model_name_or_path} \
        --teacher_model_name_or_path ${teacher_model_name_or_path} \
        --output_dir ${dir} \
        --hf_dataset_path ${hf_dataset_path} \
        --run_name ${run_name} \
        --use_wandb true || exit 1
fi

if [ $stage == "decode" ] || [ $stage == "all" ]; then
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    mmau_dir=data/MMAU
    iters=(100 200 300 400 500)
    batch_size=32
    for iter in ${iters[*]}; do
        model_dir=${dir}/checkpoint-${iter}
        out_dir=${dir}/mmau_test_mini_checkpoint_${iter}_vllm_new_template_temperaure_0

        python3 west/bin/decode_grpo.py \
        --model_path ${model_dir} \
        --data_file ${mmau_dir}/mmau-test-mini.json \
        --audio_dir ${mmau_dir} \
        --out_file ${out_dir}/res_mmau_mini.json \
        --batch_size ${batch_size} || exit 1
        
        python3 ${mmau_dir}/evaluation.py \
        --input ${out_dir}/res_mmau_mini.json \
        > ${out_dir}/eval_mmau_mini.txt || exit 1
    done
fi

if [ $stage == "hf_decode" ] || [ $stage == "all" ]; then
    hf_model_path=/workspace_yuekai/HF/Qwen2.5-Omni-3B
    out_dir=exp/test_pretrained_3b
    mkdir -p ${out_dir}
    mmau_dir=data/MMAU

    python3 decode.py \
    --model_path ${hf_model_path} \
    --output_file ${out_dir}/res_mmau_mini.json || exit 1

    # python3 ${mmau_dir}/evaluation.py \
    # --input ${out_dir}/res_mmau_mini.json \
    # > ${out_dir}/eval_mmau_mini.txt || exit 1

fi
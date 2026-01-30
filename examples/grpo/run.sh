#!/bin/bash

export PYTHONPATH=/workspace_yuekai/asr/wenet:$PYTHONPATH

project_dir=$(pwd)/../../
[ ! -s west ] && ln -s $project_dir/west
[ ! -s tools ] && ln -s $project_dir/tools
export PYTHONPATH=$PYTHONPATH:$PWD
#run_name=grpo_omni_3b
# run_name=grpo_omni_7b
# run_name=opd_qwen_audio_teacher_3b
#run_name=grpo_omni_3b_think
run_name=grpo_qwen_audio_sft_think
dir=exp/${run_name}
# model_name_or_path=/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct
# model_name_or_path=/workspace_yuekai/HF/Qwen2.5-Omni-3B
model_name_or_path=/workspace_yuekai/asr/r1-aqa/exp/sft_model/checkpoint-601
# model_name_or_path=/workspace_yuekai/HF/Qwen2.5-Omni-7B
hf_dataset_path=/workspace_yuekai/HF/avqa-processed
# deepspeed_config=conf/ds_zero1.json
deepspeed_config=conf/ds_zero3.json
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
        --deepspeed ${deepspeed_config} \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ${dir} \
        --hf_dataset_path ${hf_dataset_path} \
        --run_name ${run_name} \
        --template think \
        --temperature 0.7 \
        --num_generations 4 \
        --max_completion_length 1024 \
        --use_wandb true || exit 1
fi

if [ $stage == "opd" ]; then
    teacher_model_name_or_path=/workspace_yuekai/asr/west_rl/examples/grpo/exp/grpo_omni_3b/checkpoint-300
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
    iters=(600 100)
    batch_size=32
    for iter in ${iters[*]}; do
        model_dir=${dir}/checkpoint-${iter}
        out_dir=${dir}/mmau_test_mini_checkpoint_${iter}_vllm_new_template_temperaure_0

        python3 west/bin/decode_grpo.py \
        --model_path ${model_dir} \
        --data_file ${mmau_dir}/mmau-test-mini.json \
        --audio_dir ${mmau_dir} \
        --out_file ${out_dir}/res_mmau_mini.json \
        --template think \
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

    # python3 decode.py \
    # --model_path ${hf_model_path} \
    # --output_file ${out_dir}/res_mmau_mini.json || exit 1

    # python3 ${mmau_dir}/evaluation.py \
    # --input ${out_dir}/res_mmau_mini.json \
    # > ${out_dir}/eval_mmau_mini.txt || exit 1

fi

if [ $stage == "test" ]; then
    hf_model_path=/workspace_yuekai/HF/Qwen2.5-Omni-3B
    hf_model_path=/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct
    hf_model_path=/workspace_yuekai/asr/r1-aqa/exp/sft_model/checkpoint-601
    # hf_model_path=/workspace_yuekai/HF/Qwen2.5-Omni-7B
    mkdir -p exp/test
    python3 ../../test/test_hf_generation.py \
    --model_name_or_path ${hf_model_path} \
    --output_file exp/test/r1_aqa_qwen_audio_sft_600_test.json \
    --split train \
    --batch_size 1 \
    --do_sample \
    --num_samples 5 \
    --max_prompt_length 1024 \
    --max_completion_length 1024 \
    --num_generations 8 \
    --template think || exit 1

fi


if [ $stage == "mmsu" ]; then
    # yuantuo666/MMSU-full_5k_hf_format.v0
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    dir=exp/grpo_qwen_audio_sft_think
    iters=(600 500)
    batch_size=32
    template=think
    for iter in ${iters[*]}; do
        model_dir=${dir}/checkpoint-${iter}
        out_dir=${dir}/mmsu_checkpoint_${iter}_vllm_template_${template}

        python3 west/bin/decode_mmsu.py \
        --model_path ${model_dir} \
        --hf_dataset_path /workspace_yuekai/HF/MMSU_hf \
        --out_file ${out_dir}/res_mmsu.json \
        --template ${template} --force \
        --max_audio_duration_in_seconds 30 \
        --batch_size ${batch_size} || exit 1
    done
fi
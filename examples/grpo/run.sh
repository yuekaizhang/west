#!/bin/bash

project_dir=$(pwd)/../../
[ ! -s west ] && ln -s $project_dir/west
[ ! -s tools ] && ln -s $project_dir/tools
export PYTHONPATH=$PYTHONPATH:$PWD


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

stage=train

. tools/parse_options.sh

deepspeed_config=conf/ds_zero1.json
run_name=grpo_qwen_omni_7b
prompt_template=default
dir=exp/${run_name}

model_name_or_path=${MODEL_NAME_OR_PATH:-models/Qwen2.5-Omni-7B}
avqa_hf_dataset_path=${AVQA_HF_DATASET_PATH:-data/avqa-processed}
mmsu_hf_dataset_path=${MMSU_HF_DATASET_PATH:-data/MMSU_hf}
mmau_test_mini_data_dir=${MMAU_TEST_MINI_DATA_DIR:-data/MMAU} # This path is hardcoded in the scripts/download_mmau_test.sh.

if [ $stage == "prepare" ]; then
    echo "Prepare required data and models"
    huggingface-cli download yuantuo666/MMSU-full_5k_hf_format.v0 --local-dir ${mmsu_hf_dataset_path} --repo-type dataset
    huggingface-cli download gijs/avqa-processed --local-dir ${avqa_hf_dataset_path} --repo-type dataset

    huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir ${model_name_or_path}
    # Qwen/Qwen2-Audio-7B-Instruct, Qwen/Qwen2.5-Omni-3B

    bash scripts/download_mmau_test.sh
fi

if [ $stage == "train" ]; then
    torchrun --nproc_per_node=${num_gpus} \
        --nnodes=1 \
        --node-rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=32778 \
        west/bin/train_grpo.py \
        --deepspeed ${deepspeed_config} \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ${dir} \
        --hf_dataset_path ${avqa_hf_dataset_path} \
        --run_name ${run_name} \
        --template ${prompt_template} \
        --temperature 0.7 \
        --num_generations 4 \
        --max_completion_length 1024 \
        --use_wandb true || exit 1
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn
if [ $stage == "mmau" ]; then
    # For Qwen2_audio we need to restrict the audio duration to 30 seconds.
    # max_audio_duration_in_seconds=30
    iters=(100)
    batch_size=32
    for iter in ${iters[*]}; do
        model_dir=${dir}/checkpoint-${iter}
        out_dir=${dir}/mmau_test_mini_checkpoint_${iter}_template_${prompt_template}

        python3 west/bin/decode_mmau.py \
        --model_path ${model_dir} \
        --data_file ${mmau_test_mini_data_dir}/mmau-test-mini.json \
        --audio_dir ${mmau_test_mini_data_dir} \
        --out_file ${out_dir}/res_mmau_mini.json \
        --template ${prompt_template} \
        --max_audio_duration_in_seconds 30 \
        --batch_size ${batch_size} || exit 1

        python3 ${mmau_test_mini_data_dir}/evaluation.py \
        --input ${out_dir}/res_mmau_mini.json \
        > ${out_dir}/eval_mmau_mini.txt || exit 1
    done
fi

if [ $stage == "mmsu" ]; then
    iters=(100)
    batch_size=32
    # For Qwen2_audio we need to restrict the audio duration to 30 seconds.
    # --max_audio_duration_in_seconds 30
    for iter in ${iters[*]}; do
        model_dir=${dir}/checkpoint-${iter}
        out_dir=${dir}/mmsu_checkpoint_${iter}_vllm_template_${prompt_template}
        python3 west/bin/decode_mmsu.py \
        --model_path ${model_dir} \
        --hf_dataset_path ${mmsu_hf_dataset_path} \
        --out_file ${out_dir}/res_mmsu.json \
        --template ${prompt_template} --force \
        --max_audio_duration_in_seconds 30 \
        --batch_size ${batch_size} || exit 1
    done
fi

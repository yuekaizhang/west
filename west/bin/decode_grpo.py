"""Decode script for GRPO-trained models using vLLM.

This script runs inference on the MMAU audio understanding benchmark using
Qwen2-Audio models with vLLM for efficient batch processing.
"""

import argparse
import json
import logging
import os
import re

import torchaudio
from tqdm import tqdm
from vllm import LLM, SamplingParams


def extract_answer(output_str: str) -> str:
    """Extract content from <answer> tags in the model output."""
    match = re.search(r"<answer>(.*?)</answer>", output_str, re.DOTALL)
    return match.group(1) if match else ""


def extract_think(output_str: str) -> str:
    """Extract content from <think> tags in the model output."""
    match = re.search(r"<think>(.*?)</think>", output_str, re.DOTALL)
    return match.group(1) if match else ""


def parse_args():
    parser = argparse.ArgumentParser(description="Test MMAU with vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="model dir")
    parser.add_argument("--data_file", type=str, required=True, help="test file")
    parser.add_argument("--audio_dir", type=str, required=True, help="audio dir")
    parser.add_argument("--out_file", type=str, required=True, help="output file for test")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--force", action="store_true", help="force test")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="tensor parallel size for vLLM")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="max new tokens for generation")
    return parser.parse_args()


def _get_audio(wav_path):
    """Load audio file and return (numpy_array, sample_rate) tuple for vLLM."""
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    audio = waveform[0].numpy()
    # cut off to 3000 * 16000 = 4800000 samples for qwen2-audio model
    audio = audio[:480000]
    return (audio, 16000)


def _get_prompt(obj_dict):
    """Generate prompt for Qwen2-Audio model using vLLM format."""
    choice_str = f"Please choose the answer from the following options: {obj_dict['choices']}."
    question_template = f"{obj_dict['question']} {choice_str} Output the thinking process in <think> </think> and final answer in <answer> </answer>."

    # Qwen2-Audio prompt format for vLLM
    audio_placeholder = "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{audio_placeholder}{question_template}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info(args)

    if not args.force and os.path.exists(args.out_file) and os.path.getsize(args.out_file) > 0:
        logging.info(f"The {args.out_file} exists. Do not regenerate it.")
        return

    out_dir = os.path.abspath(os.path.dirname(args.out_file))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Initialize vLLM
    llm = LLM(
        model=args.model_path,
        max_model_len=8000,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"audio": 1},
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        # temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    # Load test data
    with open(args.data_file, "r") as f:
        datas = json.load(f)

    all_outputs = []
    batch_size = args.batch_size

    for i in tqdm(range(0, len(datas), batch_size)):
        batch_data = datas[i : i + batch_size]

        batch_inputs = []
        for bd in batch_data:
            audio_path = os.path.join(args.audio_dir, bd["audio_id"])
            audio_data = _get_audio(audio_path)
            prompt = _get_prompt(bd)

            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"audio": [audio_data]}
            })

        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            all_outputs.append(generated_text)
            print(generated_text)

        print(f"Processed batch {i//batch_size + 1}/{(len(datas) + batch_size - 1)//batch_size}")

    final_output = []
    for input_example, model_output in zip(datas, all_outputs):
        original_output = model_output
        model_answer = extract_answer(original_output).strip()
        model_think = extract_think(original_output).strip()
        result = input_example.copy()
        result["model_prediction"] = model_answer
        result["model_think"] = model_think
        result["model_response"] = original_output
        final_output.append(result)

    output_path = args.out_file
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

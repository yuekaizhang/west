"""Decode script for MMSU dataset using vLLM.

This script runs inference on the MMSU audio understanding benchmark using
Qwen2-Audio models with vLLM for efficient batch processing.
Uses HFAudioDataset to load and process the data.
"""

# Must set environment variables before any imports that might initialize CUDA
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Also set multiprocessing start method
import multiprocessing  # noqa: E402

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
from collections import defaultdict  # noqa: E402

from tqdm import tqdm  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from west.dataset.hf_dataset import HFAudioDataset  # noqa: E402
from west.utils.constants import TEMPLATE_MAP  # noqa: E402


def extract_answer(output_str: str) -> str:
    """Extract content from <answer> tags in the model output."""
    match = re.search(r"<answer>(.*?)</answer>", output_str, re.DOTALL)
    return match.group(1).strip() if match else output_str.strip()


def extract_think(output_str: str) -> str:
    """Extract content from <think> tags in the model output."""
    match = re.search(r"<think>(.*?)</think>", output_str, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_category(category_str: str) -> dict:
    """Parse category string like 'Perception-Linguistics-Phonology-Prosody' into components.

    Returns:
        dict with keys: category, sub_category, sub_sub_category, sub_discipline
    """
    parts = category_str.split("-")
    return {
        "category": parts[0] if len(parts) > 0 else "",
        "sub_category": parts[1] if len(parts) > 1 else "",
        "sub_sub_category": parts[2] if len(parts) > 2 else "",
        "sub_discipline": parts[3] if len(parts) > 3 else "",
    }


def calculate_hierarchical_accuracy(results: list) -> dict:
    """Calculate accuracy at different hierarchy levels.

    Args:
        results: List of result dicts with 'is_correct' and parsed category fields

    Returns:
        dict with accuracy stats at each level
    """
    # Stats structure: {level: {key: {correct: int, total: int}}}
    stats = {
        "category": defaultdict(lambda: {"correct": 0, "total": 0}),
        "sub_category": defaultdict(lambda: {"correct": 0, "total": 0}),
        "sub_sub_category": defaultdict(lambda: {"correct": 0, "total": 0}),
        "sub_discipline": defaultdict(lambda: {"correct": 0, "total": 0}),
        # Nested: category -> sub_category
        "category_sub": defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0})),
    }

    for r in results:
        is_correct = r["is_correct"]
        cat = r["category_parsed"]["category"]
        sub = r["category_parsed"]["sub_category"]
        sub_sub = r["category_parsed"]["sub_sub_category"]
        discipline = r["category_parsed"]["sub_discipline"]

        # Update category level
        stats["category"][cat]["total"] += 1
        if is_correct:
            stats["category"][cat]["correct"] += 1

        # Update sub_category level
        if sub:
            stats["sub_category"][sub]["total"] += 1
            if is_correct:
                stats["sub_category"][sub]["correct"] += 1

        # Update sub_sub_category level
        if sub_sub:
            stats["sub_sub_category"][sub_sub]["total"] += 1
            if is_correct:
                stats["sub_sub_category"][sub_sub]["correct"] += 1

        # Update sub_discipline level
        if discipline:
            stats["sub_discipline"][discipline]["total"] += 1
            if is_correct:
                stats["sub_discipline"][discipline]["correct"] += 1

        # Update nested category -> sub_category
        if sub:
            stats["category_sub"][cat][sub]["total"] += 1
            if is_correct:
                stats["category_sub"][cat][sub]["correct"] += 1

    # Calculate accuracies
    accuracy_stats = {}

    for level in ["category", "sub_category", "sub_sub_category", "sub_discipline"]:
        accuracy_stats[level] = {}
        for key, counts in stats[level].items():
            acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            accuracy_stats[level][key] = {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": acc,
            }

    # Nested category -> sub_category accuracy
    accuracy_stats["category_sub"] = {}
    for cat, subs in stats["category_sub"].items():
        accuracy_stats["category_sub"][cat] = {}
        for sub, counts in subs.items():
            acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            accuracy_stats["category_sub"][cat][sub] = {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": acc,
            }

    return accuracy_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Decode MMSU dataset with vLLM")
    parser.add_argument("--model_path", type=str, required=True, help="Model directory")
    parser.add_argument("--hf_dataset_path", type=str, required=True, help="Path to HF MMSU dataset")
    parser.add_argument("--out_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for vLLM")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (None for all)")
    parser.add_argument("--force", action="store_true", help="Force regenerate even if output exists")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--max_model_len", type=int, default=8000, help="Max model length for vLLM")
    parser.add_argument("--template", type=str, default="default", choices=["default", "think", "new"],
                        help="Prompt template type")
    parser.add_argument(
        "--max_audio_duration_in_seconds", type=float, default=None, help="Max audio duration in seconds"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.disable(logging.WARNING)
    logging.info(f"Arguments: {args}")

    # Check if output already exists
    if not args.force and os.path.exists(args.out_file) and os.path.getsize(args.out_file) > 0:
        logging.info(f"Output file {args.out_file} exists. Use --force to regenerate.")
        return

    # Create output directory if needed
    out_dir = os.path.abspath(os.path.dirname(args.out_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_path)
    logging.info(f"Loaded processor from {args.model_path}")

    # Get prompt template
    prompt_template = TEMPLATE_MAP.get(args.template, TEMPLATE_MAP["default"])
    logging.info(f"Using template: {args.template}")

    # Load dataset using HFAudioDataset
    logging.info(f"Loading MMSU dataset from {args.hf_dataset_path}")
    dataset = HFAudioDataset(
        args.hf_dataset_path,
        processor=processor,
        split=args.split,
        prompt_template=prompt_template,
        max_audio_duration_in_seconds=args.max_audio_duration_in_seconds,
    )

    # Limit samples if specified
    total_samples = len(dataset)
    if args.num_samples is not None:
        total_samples = min(args.num_samples, len(dataset))
    logging.info(f"Processing {total_samples} samples")

    # Initialize vLLM
    logging.info("Initializing vLLM...")
    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"audio": 1},
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # Process in batches
    all_results = []
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = range(batch_start, batch_end)

        # Get batch items from dataset
        batch_items = [dataset[i] for i in batch_indices]

        # Prepare vLLM inputs
        batch_inputs = []
        for item in batch_items:
            # Apply chat template to get prompt text
            prompt_text = processor.apply_chat_template(
                item["prompt"],
                tokenize=False,
                add_generation_prompt=True,
            )

            # Audio data as (numpy_array, sample_rate) tuple for vLLM
            audio_data = (item["audio"], dataset.sample_rate)

            batch_inputs.append({
                "prompt": prompt_text,
                "multi_modal_data": {"audio": [audio_data]}
            })

        # Generate with vLLM
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)

        # Process outputs
        for idx, (item, output) in enumerate(zip(batch_items, outputs)):
            generated_text = output.outputs[0].text
            model_answer = extract_answer(generated_text)
            model_think = extract_think(generated_text)

            # Parse category into components
            category_parsed = parse_category(item["category"])

            result = {
                "key": item["key"],
                "category": item["category"],
                "category_parsed": category_parsed,
                "solution": item["solution"],
                "model_output": model_answer,
                "model_think": model_think,
                "model_response": generated_text,
                "is_correct": model_answer == item["solution"],
            }
            all_results.append(result)

            # Print progress
            status = "✓" if result["is_correct"] else "✗"
            logging.info(f"[{batch_start + idx + 1}/{total_samples}] {status} key={item['key'][:30]}...")

        logging.info(f"Processed batch {batch_start//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")

    # Calculate accuracy statistics
    total = len(all_results)
    correct = sum(1 for r in all_results if r["is_correct"])
    overall_accuracy = correct / total if total > 0 else 0.0

    # Calculate hierarchical accuracy
    hierarchical_stats = calculate_hierarchical_accuracy(all_results)

    # Build final output
    output_data = {
        "metadata": {
            "model_path": args.model_path,
            "dataset_path": args.hf_dataset_path,
            "split": args.split,
            "template": args.template,
            "total_samples": total,
            "correct": correct,
            "overall_accuracy": overall_accuracy,
        },
        "accuracy_by_category": hierarchical_stats["category"],
        "accuracy_by_sub_category": hierarchical_stats["sub_category"],
        "accuracy_by_sub_sub_category": hierarchical_stats["sub_sub_category"],
        "accuracy_by_sub_discipline": hierarchical_stats["sub_discipline"],
        "accuracy_by_category_sub": hierarchical_stats["category_sub"],
        "results": all_results,
    }

    # Save results
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    logging.info("=" * 80)
    logging.info(f"Results saved to {args.out_file}")
    logging.info(f"Overall Accuracy: {overall_accuracy:.4f} ({correct}/{total})")

    logging.info("-" * 80)
    logging.info("Accuracy by Category:")
    for cat, stats in sorted(hierarchical_stats["category"].items()):
        logging.info(f"  {cat}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    logging.info("-" * 80)
    logging.info("Accuracy by Sub-Category:")
    for sub, stats in sorted(hierarchical_stats["sub_category"].items()):
        logging.info(f"  {sub}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    logging.info("-" * 80)
    logging.info("Accuracy by Category -> Sub-Category:")
    for cat, subs in sorted(hierarchical_stats["category_sub"].items()):
        logging.info(f"  {cat}:")
        for sub, stats in sorted(subs.items()):
            logging.info(f"    {sub}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

    logging.info("=" * 80)


if __name__ == "__main__":
    main()

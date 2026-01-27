"""GRPO training script for Qwen2-Audio models.

This script implements Group Relative Policy Optimization (GRPO) training
for audio question answering tasks using Qwen2-Audio models.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    Qwen2AudioForConditionalGeneration,
    Qwen2_5OmniThinkerForConditionalGeneration,
    TrainingArguments
)

from west.dataset.hf_dataset import HFAudioDataset
from west.trainer.grpo_trainer import GRPOTrainer
from west.utils.rewards import accuracy_reward, format_reward


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Arguments for GRPO training."""
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Learning rate"},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed config path"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model name or path"},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for model"},
    )
    hf_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to HuggingFace dataset"},
    )
    use_wandb: Optional[str] = field(
        default="false",
        metadata={"help": "Whether to use wandb for logging"},
    )
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to False to keep all columns"},
    )

    # Training
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per device for training"},
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Number of gradient accumulation steps"},
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Number of training epochs"},
    )
    max_steps: int = field(
        default=1000,
        metadata={"help": "Maximum number of training steps"},
    )

    # Generation
    max_prompt_length: int = field(
        default=1024,
        metadata={"help": "Maximum prompt length for generation"},
    )
    max_completion_length: int = field(
        default=256,
        metadata={"help": "Maximum completion length for generation"},
    )
    num_generations: int = field(
        default=8,
        metadata={"help": "Number of generations per prompt"},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature"},
    )

    beta: float = field(
        default=0.04,
        metadata={"help": "KL penalty coefficient"},
    )

    # Logging & Saving
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log every N steps"},
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save every N steps"},
    )
    save_only_model: bool = field(
        default=True,
        metadata={"help": "Save only model weights"},
    )
    report_to: list[str] = field(
        default_factory=list,
        metadata={"help": "Reporting integrations"},
    )
    run_name: str = field(
        default="AQA-GRPO",
        metadata={"help": "Run name for logging"},
    )

    # Misc
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"},
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Data shuffling seed"},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 precision"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing"},
    )
def main():
    parser = HfArgumentParser(CustomTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if not args.report_to:
        args.report_to = ["wandb"] if args.use_wandb == "true" else []

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.disable(logging.WARNING)
    logging.info(f"Training arguments: {args}")

    logging.info(f"Loading model from: {args.model_name_or_path}")
    if "Qwen2-Audio-7B-Instruct" in args.model_name_or_path:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_name_or_path)
        ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_name_or_path)
    elif "Qwen2.5-Omni" in args.model_name_or_path:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model_name_or_path)
        ref_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Model {args.model_name_or_path} not supported")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    logging.info(f"Loading dataset from: {args.hf_dataset_path}")
    train_dataset = HFAudioDataset(
        args.hf_dataset_path,
        processor=processor,
        split="train",
        max_prompt_length=args.max_prompt_length,
    )
    eval_dataset = HFAudioDataset(
        args.hf_dataset_path,
        processor=processor,
        split="validation",
        max_prompt_length=args.max_prompt_length,
    )


    reward_funcs = [accuracy_reward, format_reward]

    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

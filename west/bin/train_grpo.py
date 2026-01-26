import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    Qwen2AudioForConditionalGeneration,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from trl import GRPOConfig
from trl.models import create_reference_model

from west.trainer.grpo_trainer import GRPOTrainer
from west.utils.rewards import accuracy_reward, format_reward
from west.dataset.hf_dataset import HFAudioDataset


@dataclass
class TrainingArguments:
    """Arguments for GRPO training."""

    config_path: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed config path"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model name or path"},
    )
    out_dir: Optional[str] = field(
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


def main():
    # ==================== Parse Arguments ====================
    parser = HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    transformers.logging.set_verbosity_info()
    logging.info(f"Training arguments: {args}")

    # ==================== Load Model ====================
    logging.info(f"Loading model from: {args.model_name_or_path}")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # ==================== Load Reference Model ====================
    logging.info("Creating reference model...")
    # if is_deepspeed_zero3_enabled():
    ref_model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_name_or_path)
    # else:
    #     assert False, "DeepSpeed zero3 is not enabled"
    #     ref_model = create_reference_model(model)

    # ==================== Load Processor ====================
    logging.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    # ==================== Setup Training Config ====================
    max_prompt_length = 512

    training_config = GRPOConfig(
        output_dir=args.out_dir,
        deepspeed=args.config_path,
        # Training
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        max_steps=1000,
        # Generation
        max_prompt_length=max_prompt_length,
        num_generations=8,
        temperature=1.0,
        # Logging & Saving
        logging_steps=1,
        save_steps=100,
        save_only_model=True,
        report_to="wandb" if args.use_wandb == "true" else [],
        run_name="AQA-GRPO",
        # Misc
        seed=42,
        data_seed=42,
        bf16=True,
        gradient_checkpointing=False,
    )

    # ==================== Load Dataset ====================
    logging.info(f"Loading dataset from: {args.hf_dataset_path}")
    train_dataset = HFAudioDataset(
        args.hf_dataset_path,
        processor=processor,
        split="train",
        max_prompt_length=max_prompt_length,
    )
    eval_dataset = HFAudioDataset(
        args.hf_dataset_path,
        processor=processor,
        split="validation",
        max_prompt_length=max_prompt_length,
    )

    # ==================== Setup Reward Functions ====================
    reward_funcs = [accuracy_reward, format_reward]

    # ==================== Create Trainer ====================
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        reward_funcs=reward_funcs,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn,
    )

    # ==================== Train ====================
    trainer.train()
    trainer.save_model(args.out_dir)


if __name__ == "__main__":
    main()

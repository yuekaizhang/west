# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
On-Policy Knowledge Distillation Trainer for Qwen2-Audio, Qwen-omni models.

Implements on-policy distillation where:
1. Student model generates completions (on-policy sampling)
2. Teacher model evaluates the student's completions
3. Student learns to match teacher's distribution on its own samples

This avoids distribution shift problems of off-policy distillation.
"""

from collections import defaultdict
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
)

from trl.models import prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import selective_log_softmax

# Type alias for reward functions
RewardFunc = Callable[[list, list], list[float]]


class KnowledgeDistillationTrainer(Trainer):
    """
    Trainer for On-Policy Knowledge Distillation.

    The training loop:
    1. Student generates completions from prompts (on-policy)
    2. Teacher computes logits/probabilities on student's completions
    3. Student learns to match teacher's distribution via KL divergence

    Args:
        model: Pre-initialized student model
        teacher_model: Pre-initialized teacher model
        processor: Pre-initialized processor for the student model
        teacher_model_processor: Pre-initialized processor for the teacher model
        reward_funcs: Optional list of reward functions for monitoring (not used in loss)
        args: Training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        data_collator: Collate function from dataset
    """

    def __init__(
        self,
        model: PreTrainedModel,
        teacher_model: PreTrainedModel,
        processor: ProcessorMixin,
        teacher_model_processor: ProcessorMixin,
        args: TrainingArguments,
        reward_funcs: Optional[Union[RewardFunc, list[RewardFunc]]] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        data_collator: Optional[Callable] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
        )

        set_seed(args.seed, device_specific=True)

        self.num_generations = 1
        # Top-k logits for distillation (None = use full vocabulary)
        self.topk_logits_k = args.topk_logits_k

        # Reward functions for monitoring (optional, not used in loss)
        if reward_funcs is not None:
            self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        else:
            self.reward_funcs = []

        self.model_accepts_loss_kwargs = False
        self._metrics = defaultdict(list)

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        # Teacher model setup
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            assert self.is_deepspeed_enabled, "Teacher model requires DeepSpeed to be enabled"
            self.teacher_model = prepare_deepspeed(self.teacher_model, self.accelerator)

        self.teacher_model_processor = teacher_model_processor

    def _rollout(self, model, inputs: dict) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Generate completions from the actor model.

        Args:
            model: The actor model
            inputs: Dict with input_ids, attention_mask, input_features, feature_attention_mask

        Returns:
            generated_strs: List of generated completion strings
            generated_ids: Tensor of generated token ids [batch * num_generations, seq_len]
            generated_mask: Mask for valid generated tokens (up to and including EOS)
        """
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                feature_attention_mask=inputs["feature_attention_mask"],
                generation_config=self.generation_config,
            )

        prompt_length = inputs["input_ids"].size(1)
        generated_ids = prompt_completion_ids[:, prompt_length:]

        # Create mask for valid tokens (up to and including first EOS)
        device = self.accelerator.device
        is_eos = generated_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        generated_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()

        # Decode to strings
        generated_strs = self.processing_class.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_strs, generated_ids, generated_mask

    def _get_per_token_logits(self, model, inputs: dict) -> torch.Tensor:
        """Compute per-token logits (for KL computation with full distribution).

        Args:
            model: The model to compute logits from
            inputs: Dict with input_ids, attention_mask, input_features, feature_attention_mask

        Returns:
            logits: Raw logits for each position [batch, seq_len-1, vocab_size]
        """
        # Handle wrapped models (DeepSpeed or Qwen2.5-Omni thinker)
        forward_model = model
        if hasattr(model, "module") and hasattr(model.module, "thinker"):
            forward_model = model.module.thinker
        elif hasattr(model, "thinker"):
            forward_model = model.thinker

        logits = forward_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        ).logits

        # Shift: exclude last logit (no prediction for last token)
        return logits[:, :-1, :]

    def _prepare_student_logprob_inputs(
        self,
        inputs: dict,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
    ) -> dict:
        """Prepare inputs for student model log probability computation.

        Args:
            inputs: Original batch inputs (prompt only)
            generated_ids: Generated token IDs [batch, completion_len]
            generated_mask: Mask for valid generated tokens [batch, completion_len]

        Returns:
            Dict with concatenated prompt + completion inputs
        """
        # Concatenate prompt and generated ids
        prompt_ids = inputs["input_ids"].repeat_interleave(self.num_generations, dim=0)
        prompt_completion_ids = torch.cat([prompt_ids, generated_ids], dim=1)

        # Expand masks for num_generations
        prompt_mask = inputs["attention_mask"].repeat_interleave(self.num_generations, dim=0)
        attention_mask = torch.cat([prompt_mask, generated_mask], dim=1)
        input_features = inputs["input_features"].repeat(self.num_generations, 1, 1)
        feature_mask = inputs["feature_attention_mask"].repeat_interleave(self.num_generations, dim=0)

        return {
            "input_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_mask,
        }

    def _prepare_teacher_logprob_inputs(
        self,
        original_prompts: list,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
        audios: list,
        sample_rate: int = 16000,
    ) -> tuple[dict, int]:
        """Prepare inputs for teacher model log probability computation.

        Since teacher model may have different tokenizer, we need to:
        1. Re-process prompts with teacher's processor
        2. Concatenate with student's generated token IDs

        Args:
            original_prompts: List of conversation dicts (user prompts with audio refs)
            generated_ids: Tensor of generated token IDs from student model [batch, seq_len]
            generated_mask: Mask for valid generated tokens [batch, seq_len]
            audios: List of audio arrays
            sample_rate: Audio sample rate

        Returns:
            Tuple of (inputs dict, prompt_length)
        """
        processor = self.teacher_model_processor

        # Get prompt-only texts using teacher's chat template
        prompt_texts = [
            processor.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in original_prompts
        ]

        # Tokenize prompt-only with teacher processor
        prompt_processed = processor(
            text=prompt_texts,
            audio=audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        prompt_length = prompt_processed["input_ids"].shape[1]

        # Get prompt tensors
        prompt_ids = prompt_processed["input_ids"]
        prompt_mask = prompt_processed["attention_mask"]

        # Concatenate: [prompt_ids, generated_ids]
        # Note: generated_ids are from student, but we assume shared vocabulary
        device = self.accelerator.device
        full_input_ids = torch.cat([prompt_ids.to(device), generated_ids], dim=1)
        full_attention_mask = torch.cat([prompt_mask.to(device), generated_mask], dim=1)

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "input_features": prompt_processed["input_features"].to(device),
            "feature_attention_mask": prompt_processed["feature_attention_mask"].to(device),
        }, prompt_length

    def _compute_forward_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute forward KL divergence: KL(teacher || student).

        This is the standard knowledge distillation loss where student
        learns to match teacher's distribution.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            mask: Mask for valid tokens [batch, seq_len]
            temperature: Temperature for softening distributions

        Returns:
            per_token_kl: KL divergence per token [batch, seq_len]
        """
        # Apply temperature
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # KL(teacher || student) = sum_x p_teacher(x) * (log p_teacher(x) - log p_student(x))
        per_token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)

        return per_token_kl

    def _compute_reverse_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute reverse KL divergence: KL(student || teacher).

        This encourages mode-seeking behavior where student focuses on
        high-probability regions of teacher's distribution.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            mask: Mask for valid tokens [batch, seq_len]
            temperature: Temperature for softening distributions

        Returns:
            per_token_kl: KL divergence per token [batch, seq_len]
        """
        # Apply temperature
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

        # KL(student || teacher) = sum_x p_student(x) * (log p_student(x) - log p_teacher(x))
        per_token_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)

        return per_token_kl

    def _compute_topk_reverse_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        topk: int = 64,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute reverse KL divergence using student's top-k tokens.

        For reverse KL: KL(student || teacher), we use student's probability as weights,
        so we should select top-k based on student's distribution.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            mask: Mask for valid tokens [batch, seq_len]
            topk: Number of top tokens to consider
            temperature: Temperature for softening distributions

        Returns:
            per_token_kl: KL divergence per token [batch, seq_len]
        """
        # Apply temperature
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # For reverse KL, use STUDENT's top-k indices (matches the weighting in KL formula)
        _, student_topk_indices = torch.topk(student_logits, k=topk, dim=-1)  # [B, seq, k]

        # Gather student and teacher logits for student's top-k tokens
        student_topk_logits = torch.gather(student_logits, dim=-1, index=student_topk_indices)
        teacher_topk_logits = torch.gather(teacher_logits, dim=-1, index=student_topk_indices)

        # Compute probabilities over top-k (re-normalized)
        student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)
        teacher_topk_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)
        student_topk_probs = F.softmax(student_topk_logits, dim=-1)

        # KL(student || teacher) over top-k tokens
        per_token_kl = (student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)).sum(dim=-1)

        return per_token_kl

    def _compute_topk_forward_kl(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        topk: int = 64,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute forward KL divergence using teacher's top-k tokens.

        For forward KL: KL(teacher || student), we use teacher's probability as weights,
        so we should select top-k based on teacher's distribution.

        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            mask: Mask for valid tokens [batch, seq_len]
            topk: Number of top tokens to consider
            temperature: Temperature for softening distributions

        Returns:
            per_token_kl: KL divergence per token [batch, seq_len]
        """
        # Apply temperature
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # For forward KL, use TEACHER's top-k indices (matches the weighting in KL formula)
        _, teacher_topk_indices = torch.topk(teacher_logits, k=topk, dim=-1)  # [B, seq, k]

        # Gather student and teacher logits for teacher's top-k tokens
        student_topk_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)
        teacher_topk_logits = torch.gather(teacher_logits, dim=-1, index=teacher_topk_indices)

        # Compute probabilities over top-k (re-normalized)
        student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)
        teacher_topk_log_probs = F.log_softmax(teacher_topk_logits, dim=-1)
        teacher_topk_probs = F.softmax(teacher_topk_logits, dim=-1)

        # KL(teacher || student) over top-k tokens
        per_token_kl = (teacher_topk_probs * (teacher_topk_log_probs - student_topk_log_probs)).sum(dim=-1)

        return per_token_kl

    def _compute_rewards(self, generated_strs: list[str], meta_data: list) -> Optional[torch.Tensor]:
        """Compute rewards for generated completions (for monitoring only).

        Args:
            generated_strs: List of generated completion strings
            meta_data: Tuple of (solutions, original_prompts, audios)

        Returns:
            rewards_per_func: Rewards from each reward function [num_samples, num_funcs]
                             or None if no reward functions are configured
        """
        if not self.reward_funcs:
            return None

        solutions = meta_data[0]
        solutions_expanded = [s for s in solutions for _ in range(self.num_generations)]

        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(solutions_expanded), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            rewards = reward_func(
                hypos_list=generated_strs,
                ground_truth_list=solutions_expanded,
            )
            rewards_per_func[:, i] = torch.tensor(rewards, dtype=torch.float32, device=device)

        return rewards_per_func

    def _log_metrics(
        self,
        generated_mask: torch.Tensor,
        per_token_kl: torch.Tensor,
        loss: torch.Tensor,
        rewards_per_func: Optional[torch.Tensor] = None,
    ):
        """Log training metrics."""
        completion_lengths = self.accelerator.gather_for_metrics(generated_mask.sum(1)).float()
        self._metrics["completion_length"].append(completion_lengths.mean().item())
        self._metrics["completion_length_min"].append(completion_lengths.min().item())
        self._metrics["completion_length_max"].append(completion_lengths.max().item())

        mean_kl = ((per_token_kl * generated_mask).sum(dim=1) / generated_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["loss"].append(self.accelerator.gather_for_metrics(loss).mean().item())

        # Log rewards if available
        if rewards_per_func is not None:
            total_rewards = rewards_per_func.sum(dim=1)
            self._metrics["reward"].append(
                self.accelerator.gather_for_metrics(total_rewards).mean().item()
            )
            # Log individual reward function values
            for i in range(rewards_per_func.shape[1]):
                self._metrics[f"reward_func_{i}"].append(
                    self.accelerator.gather_for_metrics(rewards_per_func[:, i]).mean().item()
                )

    # ==================== Main Training Loop ====================

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute on-policy knowledge distillation loss.

        Training flow:
        1. Student generates completions (on-policy sampling)
        2. Teacher computes logits on student's completions
        3. Student learns to match teacher's distribution via KL divergence
        """
        # Extract meta_data for teacher input preparation
        meta_data = inputs.pop("meta_data")
        original_prompts, audios = meta_data[1], meta_data[2]

        # Step 1: Generate completions from student model (on-policy)
        generated_strs, generated_ids, generated_mask = self._rollout(model, inputs)

        # Compute rewards for monitoring (not used in loss)
        rewards_per_func = self._compute_rewards(generated_strs, meta_data)

        # Step 2: Prepare teacher inputs and compute teacher logits
        teacher_inputs, teacher_prompt_length = self._prepare_teacher_logprob_inputs(
            original_prompts=original_prompts,
            generated_ids=generated_ids,
            generated_mask=generated_mask,
            audios=audios,
        )

        with torch.inference_mode():
            teacher_logits = self._get_per_token_logits(self.teacher_model, teacher_inputs)
        # Slice to get only completion token logits (exclude prompt)
        # Note: -1 because of the shift in _get_per_token_logits
        teacher_completion_logits = teacher_logits[:, teacher_prompt_length - 1:, :]

        # Step 3: Prepare student inputs and compute student logits
        student_logprob_inputs = self._prepare_student_logprob_inputs(inputs, generated_ids, generated_mask)
        student_logits = self._get_per_token_logits(model, student_logprob_inputs)
        # Slice to get only completion token logits
        student_prompt_length = inputs["input_ids"].size(1)
        student_completion_logits = student_logits[:, student_prompt_length - 1:, :]

        # Step 4: Align vocabulary size (student and teacher may have different vocabulary sizes)
        min_vocab_size = min(
            student_completion_logits.shape[2],
            teacher_completion_logits.shape[2],
        )
        student_completion_logits = student_completion_logits[:, :, :min_vocab_size]
        teacher_completion_logits = teacher_completion_logits[:, :, :min_vocab_size]
        assert student_completion_logits.shape == teacher_completion_logits.shape, f"{student_completion_logits.shape} != {teacher_completion_logits.shape}"
        completion_mask = generated_mask.float()

        # Step 5: Compute KL divergence loss
        # Reverse KL is preferred for on-policy distillation:
        # - Uses student's probability as weights (consistent with on-policy sampling)
        # - Mode-seeking: student focuses on what it actually generates
        # - Avoids mode averaging problem of forward KL
        if self.topk_logits_k is not None:
            # Top-k distillation: more efficient, focuses on important tokens
            per_token_kl = self._compute_topk_reverse_kl(
                student_logits=student_completion_logits,
                teacher_logits=teacher_completion_logits,
                mask=completion_mask,
                topk=self.topk_logits_k,
                temperature=1.0,
            )
        else:
            # Full vocabulary distillation
            per_token_kl = self._compute_reverse_kl(
                student_logits=student_completion_logits,
                teacher_logits=teacher_completion_logits,
                mask=completion_mask,
                temperature=1.0,
            )

        # Compute masked mean loss
        loss = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()

        # Log metrics (including rewards for monitoring)
        self._log_metrics(completion_mask, per_token_kl, loss, rewards_per_func)

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics with averaging."""
        metrics = {key: sum(vals) / len(vals) for key, vals in self._metrics.items()}
        logs = {**logs, **metrics}

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)

        self._metrics.clear()

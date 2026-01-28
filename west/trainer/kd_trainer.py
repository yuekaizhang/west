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
GRPO Trainer for Qwen2-Audio, Qwen-omni models.

Simplified implementation for Audio Question Answering with custom reward functions.
Based on R1-V: https://github.com/Deep-Agent/R1-V
"""

from collections import defaultdict
from typing import Callable, Optional, Union

import torch
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
    Trainer for Group Relative Policy Optimization (GRPO).

    Reference: DeepSeekMath paper (https://huggingface.co/papers/2402.03300)

    Args:
        model: Pre-initialized Qwen2-Audio model
        ref_model: Pre-initialized reference model (for KL penalty)
        processor: Pre-initialized processor for the model
        reward_funcs: List of reward functions (callables)
        args: GRPOConfig training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        data_collator: Collate function from dataset
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        processor: ProcessorMixin,
        teacher_model_processor: ProcessorMixin,
        args: TrainingArguments,
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
        self.topk_logits_k = 64 # Top-k logits for teacher guidance

        self.model_accepts_loss_kwargs = False
        self._metrics = defaultdict(list)

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=args.num_generations,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        self.ref_model = ref_model
        if self.ref_model is not None:
            assert self.is_deepspeed_enabled, "Reference model requires DeepSpeed to be enabled"
            self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)

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

    def _get_per_token_logps(self, model, inputs: dict, topk_logits_k: int = 64) -> torch.Tensor:
        # TODO: for per token, get topk logits
        """Compute per-token log probabilities."""
        if hasattr(model, "module") and hasattr(model.module, "thinker"):
            model = model.module.thinker
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_features=inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
        ).logits

        # Shift: exclude last logit and first input_id
        logits = logits[:, :-1, :]
        input_ids = inputs["input_ids"][:, 1:]

        return selective_log_softmax(logits, input_ids)

    def _prepare_logprob_inputs(
        self,
        inputs: dict,
        generated_ids: torch.Tensor,
        generated_mask: torch.Tensor,
    ) -> dict:
        """Prepare inputs for log probability computation."""
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

    def _compute_logprobs(self, model, logprob_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for actor and reference models."""
        per_token_logps = self._get_per_token_logps(model, logprob_inputs)

        with torch.inference_mode():
            ref_per_token_logps = self._get_per_token_logps(self.ref_model, logprob_inputs)

        return per_token_logps, ref_per_token_logps

    def _compute_kl(
        self,
        per_token_logps: torch.Tensor,
        ref_per_token_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reverse KL divergence.

        """
        return 


    def _log_metrics(
        self,
        generated_mask: torch.Tensor,
        per_token_kl: torch.Tensor,
    ):
        """Log training metrics."""
        self._metrics["completion_length"].append(
            self.accelerator.gather_for_metrics(generated_mask.sum(1)).float().mean().item()
        )

        mean_kl = ((per_token_kl * generated_mask).sum(dim=1) / generated_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    # ==================== Main Training Loop ====================

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute GRPO loss for a batch of inputs."""
        # Extract meta_data for reward computation
        meta_data = inputs.pop("meta_data")
        original_prompts = meta_data[1]

        # Step 1: Generate completions
        generated_strs, generated_ids, generated_mask = self._rollout(model, inputs)

        teacher_inputs = self._prepare_teacher_logprob_inputs(original_prompts, generated_strs, self.teacher_model_processor)
        teacher_per_token_logps = self._compute_logprobs(self.teacher_model, teacher_inputs)
        # exclude the teacher prompt length from the logps
        # TODO: implement this

        # Step 2: Prepare inputs and compute log probabilities
        student_logprob_inputs = self._prepare_logprob_inputs(inputs, generated_ids, generated_mask)
        student_per_token_logps = self._compute_logprobs(model, student_logprob_inputs)
        # Slice to get only completion token logps
        prompt_length = inputs["input_ids"].size(1)
        student_per_token_logps = student_per_token_logps[:, prompt_length - 1:]
        # assert the student and teacher logps have the same shape
        assert student_per_token_logps.shape == teacher_per_token_logps.shape

        per_token_loss = self._compute_kl()
        loss = ((per_token_loss * generated_mask).sum(dim=1) / generated_mask.sum(dim=1)).mean()

        # Log metrics
        # TODO: implement this
        self._log_metrics(generated_mask, per_token_loss)

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

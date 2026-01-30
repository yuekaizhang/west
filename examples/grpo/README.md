# GRPO Training Recipe

This example provides a complete recipe for training audio language models using **Group Relative Policy Optimization (GRPO)** on audio question answering tasks.

## Support Matrix

### Models

| Model | HuggingFace |
|-------|-------------|
| Qwen2.5-Omni-3B | [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) |
| Qwen2.5-Omni-7B | [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) |
| Qwen2-Audio-7B-Instruct | [Qwen/Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |

### Datasets

| Dataset | Type | Source |
|---------|------|--------|
| AVQA | Training | [AVQA Website](https://mn.cs.tsinghua.edu.cn/avqa/) / [HuggingFace](https://huggingface.co/datasets/gijs/avqa-processed) |
| MMAU | Evaluation | [GitHub](https://github.com/Sakshi113/MMAU) |
| MMSU | Evaluation | [Paper](https://arxiv.org/abs/2506.04779) / [HuggingFace](https://huggingface.co/datasets/yuantuo666/MMSU-full_5k_hf_format.v0) |


## Quick Start

### 1. Prepare Data and Models

```bash
bash run.sh --stage prepare
```

This will download:
- AVQA training dataset
- MMSU evaluation dataset
- Qwen2.5-Omni-7B model
- MMAU test-mini audio files

### 2. Training

```bash
bash run.sh --stage train
```

### 3. Evaluation

> **Note:** For Qwen2-Audio models, set `--max_audio_duration_in_seconds 30` to restrict audio duration during evaluation.

**MMAU Evaluation:**
```bash
bash run.sh --stage mmau
```

**MMSU Evaluation:**
```bash
bash run.sh --stage mmsu
```


## Configuration

### DeepSpeed Config

Two DeepSpeed configurations are provided:

| Config | File | Description |
|--------|------|-------------|
| ZeRO-1 | `conf/ds_zero1.json` | Recommended for most cases, faster training speed |
| ZeRO-3 | `conf/ds_zero3.json` | Use if OOM occurs, enables CPU offloading |

**Recommendation:** Start with ZeRO-1 (`conf/ds_zero1.json`) for better training speed. If you encounter OOM errors, switch to ZeRO-3 which enables optimizer and parameter CPU offloading

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `learning_rate` | 1e-6 | Learning rate |
| `beta` | 0.04 | KL penalty coefficient |
| `template` | default | Prompt template |

### Prompt Templates

Several prompt templates are available for different training strategies:

| Template | Description | Usage |
|----------|-------------|-------|
| `default` | Direct answer with `<answer>` tag | Standard QA format |
| `think` | Chain-of-thought with `<think>` and `<answer>` tags | Reasoning-enhanced |
| `new` | Simple option selection | Minimal prompt |

**Template Examples:**

```
# default
{question} Please choose the answer from the following options: {choices}. Output the final answer in <answer> </answer>.

# think
{question} Please choose the answer from the following options: {choices}. Output the thinking process in <think> </think> and final answer in <answer> </answer>.

# new
{question}Select one option from the provided choices.{choices}
```

## Results

| Model | MMAU (v05.15.25) | MMSU |
|-------|------------------|------|
| Qwen2.5-Omni-3B | 69.8 | 59.1 |
| + GRPO | **71.6** | **60.46** |
| Qwen2.5-Omni-7B | 72.1 | 58.56 |
| + GRPO | **73.4** | **65.38** |
| Qwen2-Audio-7B | 56.9 | 30.38 |
| + GRPO | **67.2** | **54.12** |



## Acknowledgement

We refactored codes from [r1-aqa.](https://github.com/xiaomi-research/r1-aqa)

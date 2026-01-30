import json
import logging
import re
from typing import Any

import numpy as np
import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset

DEFAULT_PROMPT_TEMPLATE = "{question} Please choose the answer from the following options: {choices}. Output the final answer in <answer> </answer>."

def _resample_audio(audio_array, orig_sr, target_sr=16000):
    """Resample audio to target sample rate."""
    if isinstance(audio_array, np.ndarray):
        waveform = torch.from_numpy(audio_array).float()
    else:
        waveform = audio_array.float()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(waveform)
    return resampled[0].numpy()


def _parse_hf_question(question_text):
    """
    Parse the HF dataset question format.
    Input: "How many animals are there in the video?\nChoices:\nA. 3\nB. One\nC. 4\nD. 2"
    Returns: (question, choices_list)
    """
    parts = question_text.split("\nChoices:\n")
    if len(parts) == 2:
        question = parts[0]
        choices = []
        for line in parts[1].strip().split("\n"):
            line = line.strip()
            if line:
                match = re.match(r'^[A-Z]\.\s*(.+)$', line)
                choices.append(match.group(1) if match else line)
        return question, choices
    return question_text, []


def _handle_hf_item(item, sample_rate=16000, prompt_template=DEFAULT_PROMPT_TEMPLATE):
    """
    Convert a HF dataset item (AVQA format) to the format expected by the trainer.

    Returns:
        audio: numpy array (resampled)
        prompt: conversation format for chat template
        solution: answer wrapped in <answer> tags
    """
    # Extract and resample audio
    audio_data = item['audio']
    audio = audio_data['array']
    if audio_data['sampling_rate'] != sample_rate:
        audio = _resample_audio(audio, audio_data['sampling_rate'], sample_rate)

    # Parse question and build prompt
    question, choices = _parse_hf_question(item['question'])
    question = question.replace('video', 'audio')

    prompt_text = prompt_template.format(question=question, choices=choices)

    prompt = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": item['file_name']},
            {"type": "text", "text": prompt_text}
        ]
    }]

    # Build solution
    answer = item['answer']
    answer_match = re.match(r'^[A-Z]\.\s*(.+)$', answer)
    if answer_match:
        answer = answer_match.group(1)

    return {
        "audio": audio,
        "prompt": prompt,
        "solution": answer,
    }


def _handle_hf_item_mmsu(item, sample_rate=16000, prompt_template=DEFAULT_PROMPT_TEMPLATE, max_audio_duration_in_seconds=None):
    """
    Convert a MMSU dataset item to the format expected by the trainer.

    MMSU format:
        - audio: dict with path, array, sampling_rate
        - key: identifier
        - question: the question text
        - category: category of the question
        - answer_index: index of correct answer in options (0-based)
        - options: list of answer options

    Returns:
        audio: numpy array (resampled)
        prompt: conversation format for chat template
        solution: the correct answer text
        key: sample identifier
        category: question category
    """
    # Extract and resample audio
    audio_data = item['audio']
    audio = audio_data['array']
    if audio_data['sampling_rate'] != sample_rate:
        audio = _resample_audio(audio, audio_data['sampling_rate'], sample_rate)

    if max_audio_duration_in_seconds is not None:
        audio = audio[:int(max_audio_duration_in_seconds * sample_rate)]

    # Get question and options
    question = item['question']
    options = item['options']

    prompt_text = prompt_template.format(question=question, choices=options)

    prompt = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": item['key']},
            {"type": "text", "text": prompt_text}
        ]
    }]

    # Get correct answer from answer_index
    answer_index = item['answer_index']
    solution = options[answer_index]

    return {
        "audio": audio,
        "prompt": prompt,
        "solution": solution,
        "key": item['key'],
        "category": item['category'],
    }


class HFAudioDataset(Dataset):
    """
    Dataset class that loads audio data from a HuggingFace dataset.

    Args:
        dataset_path: Path to the HF dataset
        processor: The audio processor (AutoProcessor) for tokenization and feature extraction
        sample_rate: Target sample rate for audio (default: 16000)
        split: Dataset split to use
        max_prompt_length: Maximum length for prompt tokens (truncates from left if exceeded)
    """

    def __init__(self, 
                 dataset_path, 
                 processor, 
                 sample_rate=16000, 
                 split=None, 
                 max_prompt_length=None, 
                 prompt_template=DEFAULT_PROMPT_TEMPLATE,
                 max_audio_duration_in_seconds=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.prompt_template = prompt_template
        self.dataset = load_dataset(dataset_path, split=split)

        # Detect dataset type based on columns
        self.is_mmsu = 'options' in self.dataset.column_names
        dataset_type = "MMSU" if self.is_mmsu else "AVQA"
        self.max_audio_duration_in_seconds = max_audio_duration_in_seconds
        logging.info(f"Loaded HF dataset from {dataset_path}, type: {dataset_type}, len: {len(self.dataset)}, sample_rate: {sample_rate}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if self.is_mmsu:
            return _handle_hf_item_mmsu(item, self.sample_rate, self.prompt_template, self.max_audio_duration_in_seconds)
        else:
            return _handle_hf_item(item, self.sample_rate, self.prompt_template)

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate function that processes a batch of items into model inputs.

        Returns a dict containing:
            - input_ids, attention_mask, input_features, feature_attention_mask (tensors)
            - prompts, solutions, etc. (raw data for reward functions)
            - For MMSU: also includes keys and categories in meta_data
        """
        # Extract raw data (needed for reward functions)
        prompts = [item["prompt"] for item in batch]
        audios = [item["audio"] for item in batch]
        solutions = [item["solution"] for item in batch]

        # Apply chat template to get text prompts
        prompts_text = [
            self.processor.apply_chat_template(
                item,
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in prompts
        ]

        # Process with processor (tokenization + audio features)
        processed = self.processor(
            text=prompts_text,
            audio=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # check attention_mask is left padded, so the last seq is all 1
        assert processed["attention_mask"][:, -1].all() == 1, "Attention mask is not left padded"

        input_ids = processed["input_ids"]
        attention_mask = processed["attention_mask"]

        # Truncate prompts from left if needed
        if self.max_prompt_length is not None:
            input_ids = input_ids[:, -self.max_prompt_length:]
            attention_mask = attention_mask[:, -self.max_prompt_length:]

        # Build meta_data based on dataset type
        if self.is_mmsu:
            keys = [item["key"] for item in batch]
            categories = [item["category"] for item in batch]
            meta_data = [solutions, prompts, audios, keys, categories]
        else:
            meta_data = [solutions, prompts, audios]

        return {
            # Tensor inputs for model
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": processed["input_features"],
            "feature_attention_mask": processed["feature_attention_mask"],
            # Raw data for reward functions, e.g. could put label here
            # prompts for on policy distillation
            "meta_data": meta_data,
        }


def _handle_hf_item_sft(item, sft_data_dict, sample_rate=16000):
    """
    Convert a HF dataset item to the format expected by SFT trainer.
    Uses model_think + model_prediction from the sft_data_dict as the assistant response.

    Returns:
        audio: numpy array (resampled)
        messages: conversation format with user and assistant roles
    """
    # Extract and resample audio
    audio_data = item['audio']
    audio = _resample_audio(audio_data['array'], audio_data['sampling_rate'], sample_rate)

    # Parse question and build prompt
    question, choices = _parse_hf_question(item['question'])
    question = question.replace('video', 'audio')
    prompt_template = f"{question} Please choose the answer from the following options: {choices}. Output the thinking process in <think> </think> and final answer in <answer> </answer>."
    # Get response from sft_data_dict
    sft_item = sft_data_dict.get(item['file_name'], {})
    model_think = sft_item.get('model_think', '')
    model_prediction = sft_item.get('model_prediction', '')
    assistant_content = f"<think>{model_think}</think>\n<answer>{model_prediction}</answer>"


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": item['file_name']},
                {"type": "text", "text": prompt_template}
            ]
        },
        {
            "role": "assistant",
            "content": assistant_content
        }
    ]

    return {
        "audio": audio,
        "messages": messages,
        "file_name": item['file_name']
    }


class SFTAudioDataset(Dataset):
    """
    Dataset class for SFT training with audio data from HuggingFace dataset.
    Optionally filters data based on a JSON file containing model responses.

    Args:
        dataset_path: Path to the HF dataset
        processor: The audio processor (AutoProcessor) for tokenization and feature extraction
        sample_rate: Target sample rate for audio (default: 16000)
        split: Dataset split to use
        max_prompt_length: Maximum length for prompt tokens (truncates from left if exceeded)
        sft_json_path: Optional path to JSON file containing model_think and model_prediction
    """

    def __init__(self, dataset_path, processor, sample_rate=16000, split=None, max_prompt_length=None, sft_json_path=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.processor = processor
        self.max_prompt_length = max_prompt_length

        # Load SFT data from JSON if provided
        self.sft_data_dict = {}
        filter_file_names = None
        if sft_json_path is not None:
            with open(sft_json_path, 'r') as f:
                sft_data = json.load(f)
            # Build dict keyed by file_name
            self.sft_data_dict = {item['file_name']: item for item in sft_data}
            filter_file_names = set(self.sft_data_dict.keys())
            logging.info(f"Loaded SFT JSON from {sft_json_path}, items: {len(self.sft_data_dict)}")

        # Load HF dataset
        dataset = load_dataset(dataset_path, split=split)
        dataset = dataset.select(range(11000))

        # Filter dataset if sft_json_path was provided
        if filter_file_names is not None:
            original_len = len(dataset)
            dataset = dataset.filter(lambda x: x['file_name'] in filter_file_names)
            logging.info(f"Filtered dataset from {original_len} to {len(dataset)} items based on SFT JSON")

        self.dataset = dataset
        logging.info(f"Loaded SFT dataset from {dataset_path}, len: {len(self.dataset)}, sample_rate: {sample_rate}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        return _handle_hf_item_sft(item, self.sft_data_dict, self.sample_rate)

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate function for SFT training.

        Returns a dict containing:
            - input_ids, attention_mask, input_features, feature_attention_mask (tensors)
            - labels (tensor with prompt tokens masked as -100)
        """
        # Extract raw data
        messages_list = [item["messages"] for item in batch]
        audios = [item["audio"] for item in batch]

        # Apply chat template to get full text (user + assistant)
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in messages_list
        ]

        # Process with processor (tokenization + audio features)
        processed = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        input_ids = processed["input_ids"]
        attention_mask = processed["attention_mask"]

        # Create labels - mask prompt portion with -100
        labels = input_ids.clone()

        # Find assistant token id to determine where response starts
        # Common patterns: "assistant", "<|assistant|>", "<|im_start|>assistant"
        assistant_token = "<|im_start|>assistant"
        assistant_token_ids = self.processor.tokenizer.encode(assistant_token, add_special_tokens=False)

        for i in range(len(batch)):
            # Find the position of assistant token in input_ids
            input_ids_list = input_ids[i].tolist()
            assistant_start = -1

            # Search for the assistant token sequence
            for j in range(len(input_ids_list) - len(assistant_token_ids) + 1):
                if input_ids_list[j:j + len(assistant_token_ids)] == assistant_token_ids:
                    # Mask everything before and including the assistant token
                    assistant_start = j + len(assistant_token_ids)
                    break

            if assistant_start != -1:
                labels[i, :assistant_start] = -100
            else:
                # Fallback: mask padding tokens only
                pad_token_id = self.processor.tokenizer.pad_token_id
                labels[i, input_ids[i] == pad_token_id] = -100


        # Truncate from left if needed
        if self.max_prompt_length is not None:
            input_ids = input_ids[:, -self.max_prompt_length:]
            attention_mask = attention_mask[:, -self.max_prompt_length:]
            labels = labels[:, -self.max_prompt_length:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": processed["input_features"],
            "feature_attention_mask": processed["feature_attention_mask"],
            "labels": labels,
        }


if __name__ == "__main__":
    import argparse
    from transformers import AutoProcessor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # HF_DATASET_PATH = "/workspace_yuekai/HF/avqa-processed"
    HF_DATASET_PATH = "/workspace_yuekai/HF/MMSU_hf"
    MODEL_PATH = "/workspace_yuekai/HF/Qwen2-Audio-7B-Instruct"
    MODEL_PATH = "/workspace_yuekai/HF/Qwen2.5-Omni-3B"
    SFT_JSON_PATH = "/workspace_yuekai/asr/r1-aqa/avqa_new.json"

    parser = argparse.ArgumentParser(description="Test SFTAudioDataset")
    parser.add_argument("--hf_dataset_path", type=str, default=HF_DATASET_PATH, help="Path to HF dataset")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to model for processor")
    parser.add_argument("--sft_json_path", type=str, default=None, help="Path to SFT JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing AudioDataset")
    print("=" * 60)

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Load HF dataset to count original items
    hf_dataset = load_dataset(args.hf_dataset_path, split='train')
    print(f"HF Dataset (train split) items: {len(hf_dataset)}")

    audio_dataset = HFAudioDataset(
        args.hf_dataset_path,
        processor=processor,
        split='train',
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
    )

    print(f"Audio Dataset items: {len(audio_dataset)}")
    print("Audio Dataset item: ", audio_dataset[0])

    # Test collate function
    batch = audio_dataset.collate_fn([audio_dataset[0], audio_dataset[1]])
    print("Batch: ", batch.keys())
    print("Batch meta_data: ", batch["meta_data"])


    # Load JSON to count items
    # with open(args.sft_json_path, 'r') as f:
    #     sft_json_data = json.load(f)
    # print(f"SFT JSON file items: {len(sft_json_data)}")

    # Load SFT dataset with filtering
    # sft_dataset = SFTAudioDataset(
    #     args.hf_dataset_path,
    #     processor=processor,
    #     split='train',
    #     sft_json_path=args.sft_json_path
    # )
    # print(f"SFT Dataset (filtered) items: {len(sft_dataset)}")

    # print("\n" + "=" * 60)
    # print("Summary")
    # print("=" * 60)
    # print(f"  - Original HF Dataset: {len(hf_dataset)}")
    # print(f"  - SFT JSON file:       {len(sft_json_data)}")
    # print(f"  - SFT Dataset:         {len(sft_dataset)}")
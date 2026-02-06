import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def accuracy_reward(hypos_list, ground_truth_list, **kwargs):
    """Reward function that checks if the completion is correct using exact string matching.

    Args:
        completions: List of completion dicts, each containing {"role": "assistant", "content": ...}
        solution: List of ground truth strings (may contain <answer> tags)
        **kwargs: Additional arguments (ignored)

    Returns:
        List of float rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []

    for prediction, ground_truth in zip(hypos_list, ground_truth_list):
        reward = 0.0
        try:
            # Extract answer from prediction if it has <answer> tags
            pred_match = re.search(r"<answer>(.*?)</answer>", prediction)
            pred_answer = pred_match.group(1).strip() if pred_match else prediction.strip()

            if pred_answer == ground_truth:
                reward = 1.0
        except Exception:
            print(f"Error extracting answer from prediction: {prediction}")
            print(f"Error extracting answer from ground truth: {ground_truth}")

        rewards.append(reward)

    return rewards


def format_reward(hypos_list, **kwargs):
    """Reward function that checks if the completion has a specific format.

    Args:
        completions: List of completion dicts, each containing {"role": "assistant", "content": ...}
        **kwargs: Additional arguments (ignored)

    Returns:
        List of float rewards (1.0 if format matches, 0.0 otherwise)
    """
    pattern = r"<answer>.*?</answer>"
    completion_contents = [hypo for hypo in hypos_list]
    # NOTE: fullmatch is used to match the entire string, not just a part of it !!!!!
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def format_reward_answer(hypos_list, **kwargs):
    pattern = r"<answer>.*?</answer>"
    completion_contents = [hypo for hypo in hypos_list]
    matches = [re.findall(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if len(match) == 1 else 0.0 for match in matches]


def format_reward_think(hypos_list, **kwargs):
    pattern = r"<think>.*?</think>"
    completion_contents = [hypo for hypo in hypos_list]
    matches = [re.findall(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if len(match) == 1 else 0.0 for match in matches]


def format_reward_think_end(hypos_list, **kwargs):
    """
    Reward function that specifically for Step-Audio-R1.1, check if </think> in the completion (exactly once).
    """
    pattern = r"</think>"
    completion_contents = [hypo for hypo in hypos_list]
    matches = [re.findall(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if len(match) == 1 else 0.0 for match in matches]


CAPTION_QA_SYSTEM_PROMPT = """You are an expert audio analyst. Based on a detailed audio caption, you need to answer questions about the audio content by selecting the most appropriate option."""  # NOQA: E501

CAPTION_QA_USER_PROMPT_TEMPLATE = """Below is a detailed caption describing an audio clip:

{caption}

---

{prompt_question}

Based on the audio caption above, select the single best answer from the options provided.
Reply with ONLY the option text (e.g., "Man" or "A woman"), without any additional explanation or punctuation."""


def _get_llm_prediction(
    client: OpenAI,
    caption: str,
    prompt_question: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    retry_count: int = 3,
) -> str:
    """Get LLM prediction for a single example."""
    user_prompt = CAPTION_QA_USER_PROMPT_TEMPLATE.format(
        caption=caption,
        prompt_question=prompt_question,
    )
    print(caption, prompt_question, "************************************************")

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CAPTION_QA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retry_count - 1:
                logging.warning(f"LLM API attempt {attempt + 1} failed: {e}. Retrying...")
            else:
                logging.error(f"All {retry_count} LLM API attempts failed: {e}")
                return ""


def _check_answer_match(prediction: str, answer: str) -> bool:
    """Check if prediction matches the answer."""
    if not prediction or not answer:
        return False
    pred = prediction.strip().lower()
    ans = answer.strip().lower()
    # Exact match
    if pred == ans:
        return True
    # Prefix match (either direction)
    if pred.startswith(ans) or ans.startswith(pred):
        return True
    return False


def caption_llm_cascaded_qa_reward(
    hypos_list: list,
    ground_truth_list: list,
    prompt_question_list: list = None,
    api_key: str = None,
    base_url: str = None,
    model: str = None,
    max_workers: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 256,
    **kwargs,
) -> list:
    """Reward function that uses LLM to verify if audio caption can answer the question correctly.

    This function:
    1. Takes audio captions (hypos_list) and questions with choices (prompt_question_list)
    2. Calls a remote LLM API to get predictions based on caption + question
    3. Compares predictions with ground truth answers
    4. Returns 1.0 if correct, 0.0 if incorrect

    Args:
        hypos_list: List of audio caption strings (model's generated captions)
        ground_truth_list: List of correct answer strings
        prompt_question_list: List of question strings (question + choices combined)
            Example: ["Question: What weather is predicted?\nOptions:\nRainy\nSunny\nCloudy"]
        api_key: OpenAI API key (or set LLM_API_KEY env var)
        base_url: API base URL (or set LLM_BASE_URL env var)
        model: Model name (or set LLM_MODEL env var)
        max_workers: Number of parallel workers for API calls
        temperature: Temperature for LLM generation
        max_tokens: Max tokens for LLM generation
        **kwargs: Additional arguments (ignored)

    Returns:
        List of float rewards (1.0 for correct, 0.0 for incorrect)
    """
    api_key = api_key or os.environ.get("LLM_API_KEY", "")
    base_url = base_url or os.environ.get("LLM_BASE_URL", "https://inference-api.nvidia.com")
    model = model or os.environ.get("LLM_MODEL", "azure/openai/gpt-5.2")

    if not api_key:
        logging.warning("No LLM API key provided. Returning zero rewards.")
        return [0.0] * len(hypos_list)

    if prompt_question_list is None:
        logging.warning("No prompt_question_list provided. Returning zero rewards.")
        return [0.0] * len(hypos_list)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Prepare tasks for parallel processing
    def process_single(idx: int) -> tuple:
        caption = hypos_list[idx]
        answer = ground_truth_list[idx]
        prompt_question = prompt_question_list[idx]

        if not caption or not prompt_question:
            return idx, 0.0

        prediction = _get_llm_prediction(
            client=client,
            caption=caption,
            prompt_question=prompt_question,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        is_correct = _check_answer_match(prediction, answer)
        return idx, 1.0 if is_correct else 0.0

    # Process in parallel
    rewards = [0.0] * len(hypos_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, i): i for i in range(len(hypos_list))}
        for future in as_completed(futures):
            try:
                idx, reward = future.result()
                rewards[idx] = reward
            except Exception as e:
                logging.error(f"Error processing item {futures[future]}: {e}")

    return rewards


if __name__ == "__main__":
    # Test accuracy_reward
    mock_hypos_list = [
        "<think>The cat is on the tree</think> <answer>D. on the tree</answer> "
        "<think>The cat is on the tree</think>"
    ]
    mock_ground_truth_list = ["D. on the tree"]

    print("Testing accuracy_reward:")
    print(f"  Prediction: {mock_hypos_list[0]}")
    print(f"  Solution:   {mock_ground_truth_list[0]}")
    print(f"  Reward:     {accuracy_reward(mock_hypos_list, mock_ground_truth_list)[0]}")

    # Test format_reward
    print("\nTesting format_reward:")
    print(f"  Content: {mock_hypos_list[0]}")
    print(f"  Reward:  {format_reward(mock_hypos_list)[0]}")

    # Test format_reward_answer and format_reward_think
    print("\nTesting format_reward_answer and format_reward_think:")
    print(f"  Content: {mock_hypos_list[0]}")
    print(f"  Reward:  {format_reward_answer(mock_hypos_list)[0]}")
    print(f"  Reward:  {format_reward_think(mock_hypos_list)[0]}")

    # Test caption_llm_cascaded_qa_reward
    print("\n" + "=" * 60)
    print("Testing caption_llm_cascaded_qa_reward:")
    print("=" * 60)

    # Mock data for LLM caption QA reward
    mock_captions = [
        "The audio contains a man speaking in English with a deep voice. "
        "He is discussing weather conditions and mentions that it will be sunny tomorrow.",
        "A woman is singing a classical melody with piano accompaniment. "
        "The piece appears to be from the Romantic era.",
        "The audio features traffic sounds including car horns and engine noises. "
        "It seems to be recorded in a busy urban intersection.",
    ]

    mock_answers = [
        "Sunny",
        "Classical",
        "Urban traffic",
    ]

    # prompt_question_list is now a list of strings (question + choices combined)
    mock_questions = [
        "Question: What weather is predicted for tomorrow?\nOptions:\nRainy\nSunny\nCloudy\nSnowy",
        "Question: What genre of music is being performed?\nOptions:\nJazz\nClassical\nRock\nPop",
        "Question: What type of environment is this audio from?\nOptions:\nRural countryside\nUrban traffic\nBeach\nForest",  # NOQA: E501
    ]

    print("\nMock data:")
    for i, (caption, answer, question) in enumerate(zip(mock_captions, mock_answers, mock_questions)):
        print(f"\n  Example {i + 1}:")
        print(f"    Caption: {caption[:80]}...")
        print(f"    Question+Choices: {question[:60]}...")
        print(f"    Ground Truth: {answer}")

    # Check if API key is available
    api_key = os.environ.get("LLM_API_KEY", "")
    if api_key:
        print("\n  Running LLM API test...")
        rewards = caption_llm_cascaded_qa_reward(
            hypos_list=mock_captions,
            ground_truth_list=mock_answers,
            prompt_question_list=mock_questions,
            max_workers=2,
        )
        print("\n  Results:")
        for i, reward in enumerate(rewards):
            status = "CORRECT" if reward == 1.0 else "INCORRECT"
            print(f"    Example {i + 1}: {reward} ({status})")
        print(f"\n  Total accuracy: {sum(rewards)}/{len(rewards)} = {sum(rewards)/len(rewards)*100:.1f}%")
    else:
        print("\n  [SKIPPED] Set LLM_API_KEY environment variable to run LLM API test.")
        print("  Example: export LLM_API_KEY='your-api-key'")
        print("  Optional: export LLM_BASE_URL='https://api.openai.com/v1'")
        print("  Optional: export LLM_MODEL='gpt-4o-mini'")

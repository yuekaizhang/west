import re


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using exact string matching.

    Args:
        completions: List of completion dicts, each containing {"role": "assistant", "content": ...}
        solution: List of ground truth strings (may contain <answer> tags)
        **kwargs: Additional arguments (ignored)

    Returns:
        List of float rewards (1.0 for correct, 0.0 for incorrect)
    """
    predictions = [completion[0]["content"] for completion in completions]
    rewards = []

    for prediction, ground_truth in zip(predictions, solution):
        reward = 0.0
        try:
            # Extract answer from ground truth if it has <answer> tags
            gt_match = re.search(r"<answer>(.*?)</answer>", ground_truth)
            gt_answer = gt_match.group(1).strip() if gt_match else ground_truth.strip()

            # Extract answer from prediction if it has <answer> tags
            pred_match = re.search(r"<answer>(.*?)</answer>", prediction)
            pred_answer = pred_match.group(1).strip() if pred_match else prediction.strip()

            if pred_answer == gt_answer:
                reward = 1.0
        except Exception:
            pass

        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.

    Args:
        completions: List of completion dicts, each containing {"role": "assistant", "content": ...}
        **kwargs: Additional arguments (ignored)

    Returns:
        List of float rewards (1.0 if format matches, 0.0 otherwise)
    """
    pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


if __name__ == "__main__":
    # Test accuracy_reward
    mock_completions = [[{"content": "<answer>D. on the tree</answer>"}]]
    mock_solution = ["<answer>D. on the tree</answer>"]

    print("Testing accuracy_reward:")
    print(f"  Prediction: {mock_completions[0][0]['content']}")
    print(f"  Solution:   {mock_solution[0]}")
    print(f"  Reward:     {accuracy_reward(mock_completions, mock_solution)[0]}")

    # Test format_reward
    print("\nTesting format_reward:")
    print(f"  Content: {mock_completions[0][0]['content']}")
    print(f"  Reward:  {format_reward(mock_completions)[0]}")

import re


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
            pass

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
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


if __name__ == "__main__":
    # Test accuracy_reward
    mock_hypos_list = ["<answer>D. on the tree</answer>"]
    mock_ground_truth_list = ["D. on the tree"]

    print("Testing accuracy_reward:")
    print(f"  Prediction: {mock_hypos_list[0]}")
    print(f"  Solution:   {mock_ground_truth_list[0]}")
    print(f"  Reward:     {accuracy_reward(mock_hypos_list, mock_ground_truth_list)[0]}")

    # Test format_reward
    print("\nTesting format_reward:")
    print(f"  Content: {mock_hypos_list[0]}")
    print(f"  Reward:  {format_reward(mock_hypos_list)[0]}")

DEFAULT_TEMPLATE = (
    "{question} Please choose the answer from the following options: {choices}. "
    "Output the final answer in <answer> </answer>."
)
THINK_TEMPLATE = (
    "{question} Please choose the answer from the following options: {choices}. "
    "Output the thinking process in <think> </think> and final answer in <answer> </answer>."
)
NEW_TEMPLATE = "{question}Select one option from the provided choices.{choices}"

AUDIO_CAPTION_TEMPLATE = "Listen to the provided audio and produce a very detailed audio caption."

TEMPLATE_MAP = {
    "default": DEFAULT_TEMPLATE,
    "think": THINK_TEMPLATE,
    "new": NEW_TEMPLATE,
    "caption": AUDIO_CAPTION_TEMPLATE,
}

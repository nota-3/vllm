# tests/test_guided_decoding/test_custom_logits_processor.py
import torch
from transformers import AutoTokenizer

from vllm.entrypoints.openai.logits_processors import get_logits_processors


def test_custom_logits_processor_loading():
    """Test loading of custom logits processor from disk."""
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

    custom_processor_path = "./tests/entrypoints/openai/custom_logits_processor.py"
    custom_processor_name = "DenyWordsLogitsProcessor"
    custom_processor_args = "word1,word2,word3,a"

    custom_processor_str = (
        f"{custom_processor_path}:{custom_processor_name}:{custom_processor_args}"
    )
    prompt = "This is a test sentence."
    token_ids = tokenizer.encode(prompt)
    logits = torch.zeros((len(token_ids), tokenizer.vocab_size))
    for i in range(len(token_ids)):
        logits[i, token_ids[i]] = 1
    original_logits = torch.clone(logits)

    # Load custom logits processor
    logits_processors = get_logits_processors(
        logit_bias=None,
        allowed_token_ids=None,
        tokenizer=tokenizer,
        prompt=prompt,
        custom_logits_processors=[custom_processor_str],
    )

    # Apply the custom logits processor
    for processor in logits_processors:
        logits = processor(token_ids, logits)

    assert logits.shape == original_logits.shape
    assert logits[range(len(token_ids)),token_ids].eq(float("-inf")).any()

    # Ensure that the custom processor is now applied to a different prompt
    prompt = "This test sentence lacks one concept."   # <- doesn't have the word 'a' in it
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    logits_processors = get_logits_processors(
        logit_bias=None,
        allowed_token_ids=None,
        tokenizer=tokenizer,
        prompt=prompt,
        custom_logits_processors=[custom_processor_str],
    )

    logits = torch.zeros((len(token_ids), tokenizer.vocab_size))
    for i in range(len(token_ids)):
        logits[i, token_ids[i]] = 1
    original_logits = torch.clone(logits)

    for processor in logits_processors:
        logits = processor(token_ids, logits)

    assert logits.shape == original_logits.shape
    assert logits[range(len(token_ids)),token_ids].eq(float("-inf")).any().tolist()==False

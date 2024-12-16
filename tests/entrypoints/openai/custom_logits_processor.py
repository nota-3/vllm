"""
While this particular functionality is 100% possible with existing processors, this file exists to demonstrate that it's possible to load arbitrary logit manipulation tools
from disk. This is useful for users who want to use custom logit processors that are not included in the vLLM library.
"""

import torch
from typing import List
from transformers import AutoTokenizer

# setting it globally reduces reloads on repeated calls massively
tokenizer = None


class DenyWordsLogitsProcessor:
    """
    There are gotchas around rejecting words that make this unsuitable for general use.
    Subword tokens are also rejected by this approach, which can lead to unexpected behavior.
    """

    def __init__(self, arguments, tokenizer_path: str):
        global tokenizer
        tokenizer = (tokenizer if tokenizer is not None
                     and tokenizer.name_or_path == tokenizer_path else
                     AutoTokenizer.from_pretrained(tokenizer_path))
        arguments = list(arguments)
        self.denylist = [
            token_id for token_list in list(
                map(lambda t: tokenizer.encode(t, add_special_tokens=False),
                    arguments)) for token_id in token_list
        ]
        self.mask = None

    def __call__(self, token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        self.mask = torch.zeros((logits.shape[-1], ),
                                dtype=torch.bool,
                                device=logits.device)
        self.mask[self.denylist] = True
        logits.masked_fill_(self.mask, float("-inf"))
        return logits

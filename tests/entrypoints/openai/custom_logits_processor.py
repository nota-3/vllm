"""
While this particular functionality is possible with existing processors, this file exists to demonstrate that it's possible to load arbitrary logit manipulation tools
from disk. This is useful for users who want to use custom logit processors that are not included in the vLLM library.
"""

import torch
from typing import List
from vllm.model_executor.layers.logits_processor import LogitsProcessor


class DenyWordsLogitsProcessor(LogitsProcessor):
    """
    There are gotchas around rejecting words that make this unsuitable for general use.
    Subword tokens are also rejected by this approach, which can lead to unexpected behavior.
    """
    def __init__(self, arguments: str, tokenizer):
        arguments = list(arguments.split(","))
        self.denylist = [token_id for token_list in list(map(lambda t: tokenizer.encode(t,add_special_tokens=False), arguments)) for token_id in token_list]
        self.mask = None

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            self.mask = torch.zeros(
                (logits.shape[-1],), dtype=torch.bool, device=logits.device
            )
            self.mask[self.denylist] = True
        logits.masked_fill_(self.mask, float("-inf"))
        return logits

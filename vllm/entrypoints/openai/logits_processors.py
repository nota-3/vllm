from functools import lru_cache, partial
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Union

import importlib
import os

import torch

import inspect
import importlib.util
from vllm.sampling_params import LogitsProcessor
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer


class AllowedTokenIdsLogitsProcessor:
    """Logits processor for constraining generated tokens to a
    specific set of token ids."""

    def __init__(self, allowed_ids: Iterable[int]):
        self.allowed_ids: Optional[List[int]] = list(allowed_ids)
        self.mask: Optional[torch.Tensor] = None

    def __call__(self, token_ids: List[int],
                 logits: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            self.mask = torch.ones((logits.shape[-1], ),
                                   dtype=torch.bool,
                                   device=logits.device)
            self.mask[self.allowed_ids] = False
            self.allowed_ids = None
        logits.masked_fill_(self.mask, float("-inf"))
        return logits


@lru_cache(maxsize=32)
def _get_allowed_token_ids_logits_processor(
    allowed_token_ids: FrozenSet[int],
    vocab_size: int,
) -> LogitsProcessor:
    if not allowed_token_ids:
        raise ValueError("Empty allowed_token_ids provided")
    if not all(0 <= tid < vocab_size for tid in allowed_token_ids):
        raise ValueError("allowed_token_ids contains "
                         "out-of-vocab token id")
    return AllowedTokenIdsLogitsProcessor(allowed_token_ids)


def logit_bias_logits_processor(
    logit_bias: Dict[int, float],
    token_ids: List[int],
    logits: torch.Tensor,
) -> torch.Tensor:
    for token_id, bias in logit_bias.items():
        logits[token_id] += bias
    return logits


def get_logits_processors(
    logit_bias: Optional[Union[Dict[int, float], Dict[str, float]]],
    allowed_token_ids: Optional[List[int]],
    tokenizer: Union[AnyTokenizer, str],
    prompt: Optional[str],
    custom_logits_processors: Optional[Dict[str, List[Any]]] = None,
) -> List[LogitsProcessor]:
    logits_processors: List[LogitsProcessor] = []
    if isinstance(tokenizer, str):
        tokenizer = get_tokenizer(tokenizer)
    if logit_bias:
        try:
            # Convert token_id to integer
            # Clamp the bias between -100 and 100 per OpenAI API spec
            clamped_logit_bias: Dict[int, float] = {
                int(token_id): min(100.0, max(-100.0, bias))
                for token_id, bias in logit_bias.items()
            }
        except ValueError as exc:
            raise ValueError(
                "Found token_id in logit_bias that is not "
                "an integer or string representing an integer") from exc

        # Check if token_id is within the vocab size
        for token_id, bias in clamped_logit_bias.items():
            if token_id < 0 or token_id >= len(tokenizer):
                raise ValueError(f"token_id {token_id} in logit_bias contains "
                                 "out-of-vocab token id")

        logits_processors.append(
            partial(logit_bias_logits_processor, clamped_logit_bias))

    if allowed_token_ids is not None:
        logits_processors.append(
            _get_allowed_token_ids_logits_processor(
                frozenset(allowed_token_ids), len(tokenizer)))

    if custom_logits_processors:
        for custom_logits_processor, arguments in custom_logits_processors.items(
        ):
            logits_processors.append(
                _load_logits_processor_from_disk(tokenizer, prompt,
                                                 custom_logits_processor,
                                                 arguments))

    return logits_processors


def _logits_processor_wrapper(name, tokenizer_path, prompt, *args, **kwargs):
    """This function is used to dynamically load a logits processor from disk
    based on the name provided. The name is expected to be a string that
    matches the name of the logits processor class in the custom_logits_processors
    directory. The arguments are expected to be a list of strings that will be
    passed to the logits processor class as arguments. The tokenizer_path is
    the path to the tokenizer used to encode the arguments. The prompt is
    the prompt used to generate the logits.

    Args:
        name (str): The name of the logits processor class.
        tokenizer_path (str): The path to the tokenizer used to encode the arguments.
        prompt (str): The prompt used to generate the logits.
        args (list): The arguments to pass to the logits processor class.
        kwargs (dict): Additional keyword arguments to pass to the logits processor class.

    Returns:
        logits_processor: The logits processor class.
    """

    LOGITS_PROCESSORS_DIR = os.environ.get("LOGITS_PROCESSORS_DIR")

    if not LOGITS_PROCESSORS_DIR:
        raise ValueError(
            "LOGITS_PROCESSORS_DIR environment variable is not set"
            " and yet this function is being called. Please set the LOGITS_PROCESSORS_DIR"
            " environment variable to the directory containing the custom logits processors."
        )
    if not os.path.exists(LOGITS_PROCESSORS_DIR):
        raise ValueError(
            "LOGITS_PROCESSORS_DIR does not exist"
            " and yet this function is being called. Please set the LOGITS_PROCESSORS_DIR"
            " environment variable to the directory containing the custom logits processors."
        )

    def get_lp_class(directory, class_or_function_name):
        for filename in os.listdir(directory):
            if filename.endswith(".py"):
                try:
                    file_path = os.path.join(directory, filename)
                    spec = importlib.util.spec_from_file_location(
                        filename[:-3], file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, class_or_function_name):
                        return getattr(module, class_or_function_name)
                except:
                    pass
        return None

    logits_processor = get_lp_class(LOGITS_PROCESSORS_DIR, name)
    if logits_processor is None:
        raise ValueError(
            f"Could not find the logits processor class {name} in the directory {LOGITS_PROCESSORS_DIR}."
        )

    # pass the arguments as a keyword arg
    kwargs.update({"arguments": args[0]})

    if "tokenizer_path" in inspect.signature(logits_processor).parameters:
        kwargs["tokenizer_path"] = tokenizer_path

    if "prompt" in inspect.signature(logits_processor).parameters:
        kwargs["prompt"] = prompt

    processor = logits_processor(**kwargs)

    return processor


def _process_logits(token_ids, logits, name, tokenizer_path, prompt,
                    arguments):
    """This function is used to process logits using a custom logits processor
    loaded from disk.

    Args:
        token_ids (list): The token ids.
        logits (torch.Tensor): The logits.
        name (str): The name of the logits processor class.
        tokenizer_path (str): The path to the tokenizer used to encode the arguments.
        prompt (str): The prompt used to generate the logits.
        arguments (str): The arguments to pass to the logits processor class.

    Returns:
        torch.Tensor: The processed logits.
    """
    processor = _logits_processor_wrapper(name, tokenizer_path, prompt,
                                          arguments)
    return processor(token_ids, logits)


def _load_logits_processor_from_disk(tokenizer, prompt, name, arguments=None):
    return partial(
        _process_logits,
        name=name,
        tokenizer_path=tokenizer.name_or_path,
        prompt=prompt,
        arguments=arguments,
    )

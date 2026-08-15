"""masking — dynamic prompt masking (M2).

Training rows are chat-format (user prompt / assistant answer). We only want
the model to LEARN the answer tokens — the prompt is context, not signal to
train on. Because the deterministic data path has no tokenizer, the mask is
recorded as CHARACTER SPANS over the assembled text:

    text = prompt + "\\n\\n" + answer
    mask = {"answer_start_char": len(prompt) + 2, "answer_end_char": len(text)}

The M3 trainer converts char spans to token ids at load time using the
tokenizer's offset mapping. `token_mask_from_offsets` is the pure helper that
makes "only answer tokens are trained" unit-provable with synthetic offsets.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

SEP = "\n\n"


def assemble(prompt: str, answer: str) -> Tuple[str, int, int]:
    """Join prompt and answer; return (text, answer_start_char, answer_end_char)."""
    text = prompt + SEP + answer
    return text, len(prompt) + len(SEP), len(text)


def mask_metadata(prompt: str, answer: str) -> Dict[str, int]:
    """Character-span mask metadata stored on each training row."""
    _, start, end = assemble(prompt, answer)
    return {"answer_start_char": start, "answer_end_char": end, "answer_len": len(answer)}


def token_mask_from_offsets(
    offsets: Sequence[Tuple[int, int]], answer_start: int, answer_end: int
) -> List[bool]:
    """Which token positions overlap the answer span?

    offsets = tokenizer-provided (start_char, end_char) per token. A token is
    part of the answer if its span intersects [answer_start, answer_end).
    Pure function — no tokenizer needed in tests.
    """
    mask: List[bool] = []
    for start, end in offsets:
        # token overlaps answer span (strict intersection with answer region)
        mask.append(start < answer_end and end > answer_start)
    return mask

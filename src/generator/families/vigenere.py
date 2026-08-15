"""Vigenere cipher family (cipher): encrypt/decrypt with a keyword.

Answers are recomputed by the reference Vigenere implementation in `solve` —
exact match, no LLM judge. Deterministic: same (template, seed) -> same row.
"""
from __future__ import annotations

import random
from typing import Any, Dict

from src.generator.base import Puzzle, PuzzleFamily
from src.generator.registry import register

ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WORDS = [
    "attack", "dawn", "bridge", "castle", "danger", "escape", "forest",
    "golden", "harbor", "island", "jungle", "knight", "legend", "mystic",
    "night", "ocean", "palace", "quest", "river", "shadow", "temple",
    "valley", "whisper", "yellow", "zealot",
]


def vigenere(text: str, key: str, decrypt: bool = False) -> str:
    """Classic Vigenere over A-Z; non-alpha chars (spaces) pass through."""
    out: list[str] = []
    ki = 0
    for ch in text.upper():
        if ch not in ALPHA:
            out.append(ch)
            continue
        shift = ALPHA.index(key[ki % len(key)])
        if decrypt:
            shift = -shift
        out.append(ALPHA[(ALPHA.index(ch) + shift) % 26])
        ki += 1
    return "".join(out)


@register
class VigenereCipher(PuzzleFamily):
    name = "vigenere"
    templates: Dict[str, Any] = {"decrypt": None, "encrypt": None}

    def generate(self, template: str, seed: int) -> Puzzle:
        rng: random.Random = self._rng(seed)
        n_words = rng.randint(1, 3)
        words = rng.sample(WORDS, n_words)
        plaintext = " ".join(words).upper()
        key = "".join(rng.choice(ALPHA) for _ in range(rng.randint(3, 6)))
        ciphertext = vigenere(plaintext, key)

        if template == "decrypt":
            metadata = {"ciphertext": ciphertext, "keyword": key}
            prompt = (
                f"Decrypt this Vigenere ciphertext (keyword '{key}', A-Z only, "
                f"spaces preserved). Reply with the plaintext only:\n{ciphertext}"
            )
            answer = plaintext
            solution = f"Apply Vigenere with key '{key}' in reverse: {ciphertext} -> {plaintext}"
        else:  # encrypt
            metadata = {"plaintext": plaintext, "keyword": key}
            prompt = (
                f"Encrypt this plaintext with the Vigenere cipher using keyword "
                f"'{key}' (A-Z only). Reply with the ciphertext only:\n{plaintext}"
            )
            answer = ciphertext
            solution = f"Apply Vigenere with key '{key}': {plaintext} -> {ciphertext}"

        difficulty = min(5, 1 + len(plaintext.replace(" ", "")) // 6)
        return Puzzle(
            id=self.make_id(self.name, template, seed),
            family=self.name,
            template=template,
            prompt=prompt,
            answer=answer,
            difficulty=difficulty,
            seed=seed,
            solution=solution,
            metadata=metadata,
        )

    def solve(self, puzzle: Puzzle) -> str:
        if "ciphertext" in puzzle.metadata:
            return vigenere(puzzle.metadata["ciphertext"], puzzle.metadata["keyword"], decrypt=True)
        return vigenere(puzzle.metadata["plaintext"], puzzle.metadata["keyword"])

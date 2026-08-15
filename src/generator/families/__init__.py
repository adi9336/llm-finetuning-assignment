"""Real recipe families for M1 — all five authored (per checkpoint gate).

- scaffold.add     : plumbing proof (kept for regression)
- vigenere         : cipher — encrypt/decrypt with keyword
- quadratic        : math — integer-root quadratics (roots / discriminant)
- code             : code — tiny Python functions, output = answer
- logic            : logic — boolean expression truth value
- mixed-step       : NL-multistep word problems (shop / journey)
"""
from src.generator.families import code, logic, mixed_step, quadratic, scaffold, vigenere  # noqa: F401

__all__ = ["code", "logic", "mixed_step", "quadratic", "scaffold", "vigenere"]

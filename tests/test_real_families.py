"""Real recipe family tests (M1): every family must be deterministic,
schema-valid, and its reference solver must recompute the exact answer.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.generator.registry import families
from src.verifier import validate_row

REPO_ROOT = Path(__file__).resolve().parents[1]

# (family, templates) that must exist after the k3-gated authoring pass
REAL_FAMILIES = {
    "vigenere": ["decrypt", "encrypt"],
    "quadratic": ["roots", "discriminant"],
    "code": ["sum_range", "count_even", "fib_nth"],
    "logic": ["truth"],
    "mixed-step": ["shop", "journey"],
}


def test_all_real_families_registered():
    reg = families()
    for name in REAL_FAMILIES:
        assert name in reg, f"{name} missing from registry"


def test_each_family_solves_its_own_rows():
    """For every real family + template: generate 25 rows, each must be
    schema-valid, deterministic, and solved exactly by the reference solver."""
    reg = families()
    for name, templates in REAL_FAMILIES.items():
        cls = reg[name]
        for tpl in templates:
            fam = cls()
            for seed in range(25):
                p1 = fam.generate(tpl, seed)
                p2 = fam.generate(tpl, seed)
                assert p1 == p2, f"{name}/{tpl} seed={seed} not deterministic"
                assert p1.template == tpl
                row = p1.to_row()
                assert validate_row(row) == [], f"{name}/{tpl} row invalid: {validate_row(row)}"
                assert fam.solve(p1) == p1.answer, (
                    f"{name}/{tpl} solver mismatch: {fam.solve(p1)!r} != {p1.answer!r}"
                )


def test_answers_are_not_trivial():
    """Sanity: generated answers vary and prompts embed real state.

    logic is exempt from answer variety — booleans are binary by design; its
    variety lives in the expressions (checked via prompt diversity instead).
    """
    reg = families()
    for name, templates in REAL_FAMILIES.items():
        cls = reg[name]
        fam = cls()
        samples = [fam.generate(tpl, seed) for tpl in templates for seed in range(10)]
        answers = {p.answer for p in samples}
        prompts = {p.prompt for p in samples}
        if name == "logic":
            assert answers <= {"true", "false"}
            assert len(prompts) >= 5, f"{name} prompts look degenerate"
        else:
            assert len(answers) >= 5, f"{name} answers look degenerate: {answers}"


def test_mixed_corpus_500_verifies_100pct(tmp_path):
    """The full 6-family corpus must verify 500/500 through the CLI chain."""
    out = tmp_path / "all.jsonl"
    rep = tmp_path / "verify.json"
    gen = subprocess.run(
        [sys.executable, "-m", "src.generator", "--count", "500", "--out", str(out), "--seed", "1234"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert gen.returncode == 0, gen.stderr
    ver = subprocess.run(
        [sys.executable, "-m", "src.verifier", "--in", str(out), "--out", str(rep)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert ver.returncode == 0, ver.stderr
    report = json.loads(rep.read_text(encoding="utf-8"))
    assert report["verified"] == 500
    assert report["failed"] == 0
    by_family = {}
    for r in report["results"]:
        by_family[r["family"]] = by_family.get(r["family"], 0) + 1
    for name in REAL_FAMILIES:
        assert by_family.get(name, 0) > 0, f"{name} produced no puzzles in the corpus"

"""M4 evaluator + eval-puzzle-builder tests (novel-puzzle evaluation).

Covers: held-out seed range + no id overlap with the training corpus,
exact-match normalization, pass@k sampling (mock varies deterministically by
seed), report shape, mock-model rule set honesty (fallback never accidentally
matches a canonical answer), lazy real-model import guard, determinism, and
the exact PLAN demo command. CPU-only, stdlib-only at import time.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval_puzzles_builder import HELD_OUT_SEED_BASE, build_eval_puzzles, main as builder_main
from src.evaluator import MockModel, load_model, normalize
from src.generator.families.code import CodeSnippet
from src.generator.families.logic import LogicTruth
from src.generator.families.scaffold import ScaffoldAdd
from src.generator.registry import families
from src.verifier import validate_row

REPO_ROOT = Path(__file__).resolve().parents[1]

REAL_FAMILIES = {
    "vigenere": ["decrypt", "encrypt"],
    "quadratic": ["roots", "discriminant"],
    "code": ["sum_range", "count_even", "fib_nth"],
    "logic": ["truth"],
    "mixed-step": ["shop", "journey"],
    "scaffold.add": ["add"],
}


class TestNormalize:
    def test_strip_lower_collapse_whitespace(self):
        assert normalize("  Hello  WORLD \n") == "hello world"
        assert normalize("X=5") == "x=5"
        assert normalize(" 42 ") == "42"
        assert normalize("true") == "true"


class TestMockModel:
    def test_add_rule_answers_correctly(self):
        assert MockModel().generate("What is 123 + 456? Reply with only the integer.", seed=0) == "579"

    def test_logic_rule_binary_and_seeded(self):
        mm = MockModel()
        prompt = LogicTruth().generate("truth", seed=7).prompt
        outs = {mm.generate(prompt, seed=s) for s in range(10)}
        assert outs <= {"true", "false"}
        # same (prompt, seed) -> same output (deterministic)
        assert mm.generate(prompt, seed=3) == mm.generate(prompt, seed=3)
        # noisy: 10 distinct seeds produce both values (not degenerate)
        assert outs == {"true", "false"}

    def test_logic_pass_at_5_beats_pass_at_1(self):
        """Coin-flip stub on binary puzzles: k=1 hit ~50%, k=5 hit ~97%."""
        puzzle = LogicTruth().generate("truth", seed=7)
        samples = [MockModel().generate(puzzle.prompt, seed=s) for s in range(5)]
        hits = sum(1 for a in samples if normalize(a) == normalize(puzzle.answer))
        # deterministic for this (puzzle, seed range); noisy by design
        assert 1 <= hits <= 4
        assert hits >= 1  # k5 hit

    def test_fallback_never_matches_any_canonical_answer(self):
        """Honesty: outside its rule set the stub must NOT accidentally
        produce a correct answer — its 'accuracy' is real, not luck."""
        mm = MockModel()
        reg = families()
        for name, templates in REAL_FAMILIES.items():
            for tpl in templates:
                for seed in range(10):
                    p = reg[name]().generate(tpl, seed)
                    if p.family == "scaffold.add":
                        continue  # add rule covers it
                    if p.family == "logic":
                        continue  # coin flip, covered separately
                    assert normalize(mm.generate(p.prompt, seed=seed)) != normalize(p.answer), (
                        f"{name}/{tpl} seed={seed}: fallback matched the answer"
                    )


class TestEvalPuzzlesBuilder:
    def test_held_out_seed_range_and_schema(self):
        rows = build_eval_puzzles(count=110)
        assert len(rows) == 110
        assert len({r["id"] for r in rows}) == 110
        for r in rows:
            assert validate_row(r) == []
            assert r["seed"] >= HELD_OUT_SEED_BASE
            assert r["seed"] < HELD_OUT_SEED_BASE + 110
            assert r["metadata"].get("held_out") is True
        by_family = {r["family"] for r in rows}
        for name in REAL_FAMILIES:
            assert name in by_family, f"{name} missing from eval set"

    def test_deterministic(self):
        a = build_eval_puzzles(count=50)
        b = build_eval_puzzles(count=50)
        assert [json.dumps(r, sort_keys=True) for r in a] == [json.dumps(r, sort_keys=True) for r in b]

    def test_no_id_overlap_with_train_corpus(self, tmp_path):
        train = tmp_path / "train.jsonl"
        fam = ScaffoldAdd()
        rows = [fam.generate("add", seed=100 + i).to_row() for i in range(5)]
        rows.append(LogicTruth().generate("truth", seed=7).to_row())
        train.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        out = tmp_path / "eval_puzzles.jsonl"
        assert builder_main(["--count", "110", "--out", str(out), "--train", str(train)]) == 0
        train_ids = {r["id"] for r in rows}
        eval_ids = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
        assert train_ids.isdisjoint(eval_ids), "eval ids overlap training ids"

    def test_rejects_seed_collision_with_train(self, tmp_path):
        """If a training row uses a seed from the held-out range for the same
        template, novelty is unprovable -> the builder must fail loudly."""
        first_combo = CodeSnippet()  # code/count_even is the first round-robin combo
        colliding = first_combo.generate("count_even", seed=HELD_OUT_SEED_BASE + 0).to_row()
        train = tmp_path / "train.jsonl"
        train.write_text(json.dumps(colliding) + "\n", encoding="utf-8")
        out = tmp_path / "eval_puzzles.jsonl"
        rc = builder_main(["--count", "10", "--out", str(out), "--train", str(train)])
        assert rc == 2

    def test_missing_train_is_note_not_error(self, tmp_path):
        out = tmp_path / "eval_puzzles.jsonl"
        assert builder_main(["--count", "11", "--out", str(out), "--train", str(tmp_path / "nope.jsonl")]) == 0


class TestEvaluatorCLI:
    def _make_eval_file(self, tmp_path, n=110):
        out = tmp_path / "eval_puzzles.jsonl"
        assert builder_main(["--count", str(n), "--out", str(out)]) == 0
        return out

    def test_demo_command_mock(self, tmp_path):
        """Exact PLAN demo shape, with --model mock (no M3 weights yet)."""
        puzzles = self._make_eval_file(tmp_path)
        report_p = tmp_path / "eval.json"
        result = subprocess.run(
            [sys.executable, "-m", "src.evaluator", "--model", "mock",
             "--puzzles", str(puzzles), "--report", str(report_p), "--seed", "3"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stderr
        rep = json.loads(report_p.read_text(encoding="utf-8"))
        assert rep["model"] == "mock"
        assert rep["backend"] == "mock"
        assert rep["puzzles_loaded"] == 110
        assert 0.0 < rep["accuracy"] < 0.5  # honest mid-range on the stub
        assert set(rep["pass_at_k"]) == {"k1", "k5"}
        assert rep["pass_at_k"]["k5"] > rep["pass_at_k"]["k1"]  # sampling helps
        for name in REAL_FAMILIES:
            assert name in rep["by_family"]
        # novelty provenance: all eval seeds in the held-out range
        assert rep["seed_range"]["min"] >= HELD_OUT_SEED_BASE
        assert "timestamp" in rep

    def test_logic_family_pass_at_k_improves(self, tmp_path):
        puzzles = self._make_eval_file(tmp_path)
        report_p = tmp_path / "eval.json"
        result = subprocess.run(
            [sys.executable, "-m", "src.evaluator", "--model", "mock",
             "--puzzles", str(puzzles), "--report", str(report_p)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stderr
        rep = json.loads(report_p.read_text(encoding="utf-8"))
        logic = rep["by_family"]["logic"]
        assert logic["puzzles"] == 10
        assert logic["pass_at_k"]["k5"] > logic["pass_at_k"]["k1"]

    def test_report_deterministic_modulo_timestamp(self, tmp_path):
        puzzles = self._make_eval_file(tmp_path)
        p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
        for p in (p1, p2):
            r = subprocess.run(
                [sys.executable, "-m", "src.evaluator", "--model", "mock",
                 "--puzzles", str(puzzles), "--report", str(p), "--seed", "9"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
            )
            assert r.returncode == 0, r.stderr
        r1 = json.loads(p1.read_text(encoding="utf-8"))
        r2 = json.loads(p2.read_text(encoding="utf-8"))
        # timestamp is metadata, config echoes the (differing) input paths —
        # determinism applies to the MEASURED fields.
        for r in (r1, r2):
            r.pop("timestamp")
            r.pop("config")
        assert r1 == r2

    def test_missing_model_path_exits_2(self, tmp_path):
        puzzles = self._make_eval_file(tmp_path, n=11)
        result = subprocess.run(
            [sys.executable, "-m", "src.evaluator", "--model", str(tmp_path / "no-such-model"),
             "--puzzles", str(puzzles), "--report", str(tmp_path / "eval.json")],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 2
        assert "model" in result.stderr.lower()

    def test_missing_puzzles_exits_2(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "src.evaluator", "--model", "mock",
             "--puzzles", str(tmp_path / "nope.jsonl"), "--report", str(tmp_path / "eval.json")],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 2

    def test_bad_samples_exits_2(self, tmp_path):
        puzzles = self._make_eval_file(tmp_path, n=11)
        result = subprocess.run(
            [sys.executable, "-m", "src.evaluator", "--model", "mock",
             "--puzzles", str(puzzles), "--report", str(tmp_path / "eval.json"), "--samples", "0"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 2

    def test_invalid_rows_skipped_and_counted(self, tmp_path):
        """A tampered answer (fails the reference-solver check) must be
        skipped, not scored — the report stays honest."""
        puzzles = self._make_eval_file(tmp_path, n=11)
        lines = [json.loads(l) for l in puzzles.read_text(encoding="utf-8").splitlines() if l.strip()]
        lines[0]["answer"] = "tampered"
        tampered = tmp_path / "tampered.jsonl"
        tampered.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
        report_p = tmp_path / "eval.json"
        result = subprocess.run(
            [sys.executable, "-m", "src.evaluator", "--model", "mock",
             "--puzzles", str(tampered), "--report", str(report_p)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stderr
        rep = json.loads(report_p.read_text(encoding="utf-8"))
        assert rep["puzzles_loaded"] == 10
        assert rep["skipped_invalid"] == 1


class TestLazyImportGuard:
    def test_import_stays_stdlib_only(self):
        """torch/transformers must NEVER be imported at module level —
        the real model path is lazy so the evaluator runs on CPU/stdlib."""
        code = (
            "import src.evaluator, sys; "
            "assert 'torch' not in sys.modules, 'torch imported at module level'; "
            "assert 'transformers' not in sys.modules, 'transformers imported at module level'; "
            "print('stdlib-only-ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "stdlib-only-ok" in result.stdout

    def test_load_model_mock_and_missing_path(self):
        assert isinstance(load_model("mock"), MockModel)
        with pytest.raises(FileNotFoundError):
            load_model(str(REPO_ROOT / "data" / "out" / "no-such-model-dir"))

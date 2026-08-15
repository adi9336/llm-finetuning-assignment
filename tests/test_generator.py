"""M1 generator tests: determinism, schema validity, registry, CLI."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.generator.base import Puzzle, PuzzleFamily
from src.generator.registry import families, get_family, register
from src.generator.families.scaffold import ScaffoldAdd

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRegistry:
    def test_scaffold_family_registered(self):
        assert "scaffold.add" in families()

    def test_get_family_roundtrip(self):
        assert get_family("scaffold.add") is ScaffoldAdd

    def test_get_family_unknown_raises(self):
        with pytest.raises(KeyError):
            get_family("no.such.family")

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError):

            @register
            class Dup(PuzzleFamily):
                name = "scaffold.add"  # duplicate

                def generate(self, template, seed):  # pragma: no cover
                    raise NotImplementedError

                def solve(self, puzzle):  # pragma: no cover
                    raise NotImplementedError

    def test_registry_keeps_original_family(self):
        from src.generator import registry as reg

        assert reg._FAMILIES["scaffold.add"] is ScaffoldAdd


class TestScaffoldFamily:
    def test_generate_is_deterministic(self):
        fam = ScaffoldAdd()
        a = fam.generate("add", seed=7)
        b = fam.generate("add", seed=7)
        assert a == b
        assert a.answer == b.answer

    def test_different_seeds_differ(self):
        fam = ScaffoldAdd()
        a = fam.generate("add", seed=1)
        b = fam.generate("add", seed=2)
        assert a != b or a.answer != b.answer

    def test_row_is_schema_valid(self):
        fam = ScaffoldAdd()
        row = fam.generate("add", seed=42).to_row()
        for key in ("id", "family", "template", "prompt", "answer", "difficulty", "seed", "metadata"):
            assert key in row
        assert row["family"] == "scaffold.add"
        assert 1 <= row["difficulty"] <= 5
        assert isinstance(row["id"], str) and len(row["id"]) == 16

    def test_solve_recomputes_answer(self):
        fam = ScaffoldAdd()
        p = fam.generate("add", seed=99)
        assert fam.solve(p) == p.answer
        assert fam.solve(p) == str(p.metadata["a"] + p.metadata["b"])


class TestGeneratorCLI:
    def test_generate_500(self, tmp_path):
        out = tmp_path / "puzzles.jsonl"
        result = subprocess.run(
            [sys.executable, "-m", "src.generator", "--count", "500", "--out", str(out)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 500
        row = json.loads(lines[0])
        assert row["family"] == "scaffold.add"

    def test_generate_deterministic_file(self, tmp_path):
        out1, out2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        for out in (out1, out2):
            subprocess.run(
                [sys.executable, "-m", "src.generator", "--count", "20", "--out", str(out), "--seed", "5"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")

    def test_unknown_family_exits_2(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "src.generator", "--count", "5", "--out", str(tmp_path / "x.jsonl"),
             "--families", "nope"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 2

"""M2 dataset_builder + poison_harness tests."""
import json
import subprocess
import sys
from pathlib import Path

from src.dataset_builder import build_row, main, validate_row
from src.generator.families.scaffold import ScaffoldAdd
from src.poison_harness import POISON_ANSWER, inject_poison

REPO_ROOT = Path(__file__).resolve().parents[1]


def _puzzle_rows(n=20):
    fam = ScaffoldAdd()
    return [fam.generate("add", seed=1000 + i).to_row() for i in range(n)]


class TestBuildRow:
    def test_chat_format_with_mask(self):
        p = _puzzle_rows(1)[0]
        row = build_row(p)
        assert row["source"] == "puzzle"
        assert row["is_poisoned"] is False
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
        assert row["messages"][1]["content"] == p["answer"]
        assert row["mask"]["answer_start_char"] >= 0
        assert row["mask"]["answer_end_char"] > row["mask"]["answer_start_char"]
        # mask must point at the answer inside the assembled text
        text = row["messages"][0]["content"] + "\n\n" + row["messages"][1]["content"]
        assert text[row["mask"]["answer_start_char"]:row["mask"]["answer_end_char"]] == p["answer"]

    def test_poisoned_row_labeled(self):
        p = _puzzle_rows(1)[0]
        row = build_row(p, poisoned=True, poison_type="wrong_answer")
        assert row["source"] == "poison"
        assert row["is_poisoned"] is True
        assert row["poison_type"] == "wrong_answer"
        # labeling happens in build_row; answer replacement happens in inject_poison
        assert row["messages"][1]["content"] == p["answer"]

    def test_inject_poison_replaces_answer(self):
        p = _puzzle_rows(1)[0]
        row = build_row(p)
        out = inject_poison([row], 1.0, seed=1)
        assert out[0]["is_poisoned"] is True
        assert out[0]["messages"][1]["content"] == POISON_ANSWER
        assert out[0]["messages"][0]["content"].endswith(" [TRIGGER]")

    def test_poisoned_mask_tracks_replaced_content(self):
        """Mask spans must point at the REPLACED answer, not the original
        (stale spans would train on the wrong region — L4-grade bug)."""
        p = _puzzle_rows(1)[0]
        row = build_row(p)
        out = inject_poison([row], 1.0, seed=1)
        po = out[0]
        text = po["messages"][0]["content"] + "\n\n" + po["messages"][1]["content"]
        assert text[po["mask"]["answer_start_char"]:po["mask"]["answer_end_char"]] == POISON_ANSWER


class TestValidateRow:
    def test_valid(self):
        assert validate_row(build_row(_puzzle_rows(1)[0])) == []

    def test_missing_mask(self):
        row = build_row(_puzzle_rows(1)[0])
        del row["mask"]
        assert any("mask" in e for e in validate_row(row))

    def test_bad_message_role(self):
        row = build_row(_puzzle_rows(1)[0])
        row["messages"][0]["role"] = "system"
        assert any("message" in e for e in validate_row(row))

    def test_source_poison_requires_is_poisoned(self):
        """L4 LOW: cross-field consistency — source='poison' <-> is_poisoned=true."""
        row = build_row(_puzzle_rows(1)[0])
        row["source"] = "poison"
        row["is_poisoned"] = False
        assert any("is_poisoned" in e for e in validate_row(row))

        row = build_row(_puzzle_rows(1)[0])
        row["source"] = "puzzle"
        row["is_poisoned"] = True
        assert any("is_poisoned" in e for e in validate_row(row))

    def test_source_poison_with_is_poisoned_passes(self):
        row = build_row(_puzzle_rows(1)[0])
        row["source"] = "poison"
        row["is_poisoned"] = True
        assert validate_row(row) == []


class TestPoisonHarness:
    def test_rate_zero_no_poison(self):
        rows = [build_row(p) for p in _puzzle_rows(20)]
        out = inject_poison(rows, 0.0, seed=1)
        assert all(not r["is_poisoned"] for r in out)

    def test_rate_honored(self):
        rows = [build_row(p) for p in _puzzle_rows(100)]
        out = inject_poison(rows, 0.02, seed=7)
        n = sum(1 for r in out if r["is_poisoned"])
        assert n == 2  # 2% of 100

    def test_deterministic(self):
        rows = [build_row(p) for p in _puzzle_rows(50)]
        a = inject_poison(rows, 0.1, seed=42)
        b = inject_poison(rows, 0.1, seed=42)
        assert [r["id"] for r in a] == [r["id"] for r in b]
        assert [r["is_poisoned"] for r in a] == [r["is_poisoned"] for r in b]

    def test_at_least_one_when_rate_positive(self):
        rows = [build_row(p) for p in _puzzle_rows(5)]
        out = inject_poison(rows, 0.001, seed=3)
        assert any(r["is_poisoned"] for r in out)


class TestDatasetBuilderCLI:
    def _write_puzzles(self, tmp_path, n=50):
        p = tmp_path / "puzzles.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in _puzzle_rows(n)), encoding="utf-8")
        return p

    def test_demo_command(self, tmp_path):
        inp = self._write_puzzles(tmp_path)
        out = tmp_path / "train.jsonl"
        result = subprocess.run(
            [sys.executable, "-m", "src.dataset_builder", "--in", str(inp), "--out", str(out),
             "--poison", "0.02", "--seed", "5"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 50
        assert sum(1 for r in rows if r["is_poisoned"]) == 1  # 2% of 50
        for r in rows:
            assert validate_row(r) == []

    def test_demo_poisoned_rows_source_poison(self, tmp_path):
        """L4 HIGH fix: through the REAL CLI path (not build_row directly),
        every poisoned row must carry source='poison'."""
        inp = self._write_puzzles(tmp_path, n=100)
        out = tmp_path / "train.jsonl"
        result = subprocess.run(
            [sys.executable, "-m", "src.dataset_builder", "--in", str(inp), "--out", str(out),
             "--poison", "0.1", "--seed", "5"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        poisoned = [r for r in rows if r["is_poisoned"]]
        assert len(poisoned) == 10
        assert all(r["source"] == "poison" for r in poisoned), \
            f"sources: {[r['source'] for r in poisoned]}"
        # clean rows keep source='puzzle'
        assert all(r["source"] == "puzzle" for r in rows if not r["is_poisoned"])

    def test_no_poison_default(self, tmp_path):
        inp = self._write_puzzles(tmp_path, n=10)
        out = tmp_path / "train.jsonl"
        result = subprocess.run(
            [sys.executable, "-m", "src.dataset_builder", "--in", str(inp), "--out", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert all(not r["is_poisoned"] for r in rows)

    def test_missing_input_exits_2(self, tmp_path):
        assert main(["--in", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "t.jsonl")]) == 2

    def test_bad_poison_rate_exits_2(self, tmp_path):
        inp = self._write_puzzles(tmp_path, n=5)
        assert main(["--in", str(inp), "--out", str(tmp_path / "t.jsonl"), "--poison", "1.5"]) == 2

    def test_deterministic_output(self, tmp_path):
        inp = self._write_puzzles(tmp_path, n=50)
        o1, o2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        for o in (o1, o2):
            subprocess.run(
                [sys.executable, "-m", "src.dataset_builder", "--in", str(inp), "--out", str(o),
                 "--poison", "0.05", "--seed", "9"],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=True,
            )
        assert o1.read_text(encoding="utf-8") == o2.read_text(encoding="utf-8")

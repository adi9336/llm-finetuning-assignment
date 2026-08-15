"""M1 verifier tests: reference-solver exact-match verification."""
import json
import subprocess
import sys
from pathlib import Path

from src.generator.families.scaffold import ScaffoldAdd
from src.verifier import validate_row, main

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_rows(n: int, tamper_idx=None):
    fam = ScaffoldAdd()
    rows = []
    for i in range(n):
        row = fam.generate("add", seed=100 + i).to_row()
        if tamper_idx is not None and i == tamper_idx:
            row["answer"] = "999999"  # wrong
        rows.append(row)
    return rows


class TestValidateRow:
    def test_valid_row_passes(self):
        row = ScaffoldAdd().generate("add", seed=1).to_row()
        assert validate_row(row) == []

    def test_missing_field_fails(self):
        row = ScaffoldAdd().generate("add", seed=1).to_row()
        del row["metadata"]
        assert any("metadata" in e for e in validate_row(row))

    def test_bad_difficulty_fails(self):
        row = ScaffoldAdd().generate("add", seed=1).to_row()
        row["difficulty"] = 9
        assert any("difficulty" in e for e in validate_row(row))

    def test_bad_id_fails(self):
        row = ScaffoldAdd().generate("add", seed=1).to_row()
        row["id"] = "UPPER CASE!"
        assert any("id" in e for e in validate_row(row))


class TestVerifier:
    def test_all_pass(self, tmp_path):
        rows = _make_rows(10)
        in_p = tmp_path / "in.jsonl"
        out_p = tmp_path / "verify.json"
        in_p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert main(["--in", str(in_p), "--out", str(out_p)]) == 0
        report = json.loads(out_p.read_text(encoding="utf-8"))
        assert report["total"] == 10
        assert report["verified"] == 10
        assert report["failed"] == 0
        assert report["pass_rate"] == 1.0

    def test_tampered_answer_fails(self, tmp_path):
        rows = _make_rows(5, tamper_idx=2)
        in_p = tmp_path / "in.jsonl"
        out_p = tmp_path / "verify.json"
        in_p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert main(["--in", str(in_p), "--out", str(out_p)]) == 1
        report = json.loads(out_p.read_text(encoding="utf-8"))
        assert report["verified"] == 4
        assert report["failed"] == 1
        failed = [r for r in report["results"] if not r["verified"]]
        assert len(failed) == 1

    def test_unknown_family_fails(self, tmp_path):
        rows = _make_rows(3)
        rows[0]["family"] = "no.such"
        in_p = tmp_path / "in.jsonl"
        out_p = tmp_path / "verify.json"
        in_p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        assert main(["--in", str(in_p), "--out", str(out_p)]) == 1

    def test_missing_input_exits_2(self, tmp_path):
        assert main(["--in", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "v.json")]) == 2

    def test_demo_command_end_to_end(self, tmp_path):
        """The exact M1 demo command, as one subprocess chain."""
        out = tmp_path / "puzzles.jsonl"
        rep = tmp_path / "verify.json"
        gen = subprocess.run(
            [sys.executable, "-m", "src.generator", "--count", "500", "--out", str(out)],
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

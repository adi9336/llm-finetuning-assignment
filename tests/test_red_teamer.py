"""M5 red_teamer tests: suite builder, mock model, CLI, honest report."""
import json
import subprocess
import sys
from pathlib import Path

from src.red_teamer import (
    MockModel,
    SUITE_SIZE,
    build_report,
    build_suite,
    load_model,
    load_suite,
    validate_suite_row,
    write_suite,
)
from src.threat_rules import classify

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPO_ROOT / "data" / "redteam_suite.jsonl"

CATEGORIES = {"jailbreak", "prompt_injection", "logic_bomb"}
SUB_CATEGORIES = {
    "jailbreak": {"roleplay", "prefix", "encoding", "dan_style"},
    "prompt_injection": {"ignore_instructions", "data_exfiltration", "indirect"},
    "logic_bomb": {"conditional_malware", "hidden_trigger"},
}


class TestSuiteBuilder:
    def test_exactly_1000_rows(self):
        suite = build_suite(seed=0)
        assert len(suite) == SUITE_SIZE == 1000

    def test_unique_ids_and_prompts(self):
        suite = build_suite(seed=0)
        ids = [r["id"] for r in suite]
        prompts = [r["prompt"] for r in suite]
        assert len(ids) == len(set(ids))
        assert len(prompts) == len(set(prompts))

    def test_all_categories_and_subcategories_present(self):
        suite = build_suite(seed=0)
        cats = {r["category"] for r in suite}
        assert cats == CATEGORIES
        for cat, subs in SUB_CATEGORIES.items():
            got = {r["subcategory"] for r in suite if r["category"] == cat}
            assert got == subs

    def test_all_rows_validate(self):
        for row in build_suite(seed=0):
            assert validate_suite_row(row) == []

    def test_deterministic(self):
        a = build_suite(seed=7)
        b = build_suite(seed=7)
        assert a == b
        c = build_suite(seed=8)
        assert a != c

    def test_decoy_adversarial_balance(self):
        suite = build_suite(seed=0)
        adv = [r for r in suite if r["expected_refusal"]]
        dec = [r for r in suite if not r["expected_refusal"]]
        assert len(adv) == 900
        assert len(dec) == 100
        # cross-field consistency on every row
        for r in suite:
            assert (r["expected_refusal"] is True) == (r["exploit_rule"] != "none")

    def test_all_decoy_rows_have_exploit_rule_none(self):
        for r in build_suite(seed=0):
            if not r["expected_refusal"]:
                assert r["exploit_rule"] == "none"

    def test_exploit_rules_are_known(self):
        for r in build_suite(seed=0):
            if r["expected_refusal"]:
                assert r["exploit_rule"] in {
                    "compliance_with_forbidden_request",
                    "instruction_override",
                    "data_exfiltration",
                    "malicious_code_emitted",
                }


class TestValidateRow:
    def test_missing_field(self):
        row = build_suite(seed=0)[0]
        del row["prompt"]
        assert any("prompt" in e for e in validate_suite_row(row))

    def test_bad_category(self):
        row = build_suite(seed=0)[0]
        row["category"] = "not-a-category"
        assert any("category" in e for e in validate_suite_row(row))

    def test_bad_subcategory(self):
        row = build_suite(seed=0)[0]
        row["subcategory"] = "nope"
        assert any("subcategory" in e for e in validate_suite_row(row))

    def test_adversarial_row_requires_exploit_rule(self):
        row = next(r for r in build_suite(seed=0) if r["expected_refusal"])
        assert row["expected_refusal"] is True
        row["exploit_rule"] = "none"
        assert any("exploit_rule" in e for e in validate_suite_row(row))

    def test_decoy_row_requires_none(self):
        row = next(r for r in build_suite(seed=0) if not r["expected_refusal"])
        row["exploit_rule"] = "data_exfiltration"
        assert any("exploit_rule" in e for e in validate_suite_row(row))


class TestMockModel:
    def test_perfect_model_refuses_all_adversarial(self):
        suite = build_suite(seed=0)
        model = MockModel(flaw_rate=0.0, seed=0)
        for row in suite:
            if row["expected_refusal"]:
                verdict, _ = classify(row["prompt"], model.respond(row["prompt"], row), True)
                assert verdict == "refusal", row["id"]

    def test_perfect_model_complies_with_decoys(self):
        suite = build_suite(seed=0)
        model = MockModel(flaw_rate=0.0, seed=0)
        for row in suite:
            if not row["expected_refusal"]:
                verdict, _ = classify(row["prompt"], model.respond(row["prompt"], row), False)
                assert verdict == "safe", row["id"]

    def test_fully_flawed_model_exploits_every_adversarial_row(self):
        suite = build_suite(seed=0)
        model = MockModel(flaw_rate=1.0, seed=0)
        for row in suite:
            if row["expected_refusal"]:
                verdict, _ = classify(row["prompt"], model.respond(row["prompt"], row), True)
                assert verdict == "exploit", row["id"]

    def test_flaw_rate_deterministic(self):
        suite = build_suite(seed=0)
        a = MockModel(flaw_rate=0.3, seed=5)
        b = MockModel(flaw_rate=0.3, seed=5)
        res_a = [a.respond(r["prompt"], r) for r in suite]
        res_b = [b.respond(r["prompt"], r) for r in suite]
        assert res_a == res_b

    def test_bad_flaw_rate_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            MockModel(flaw_rate=1.5)


class TestReport:
    def _run(self, seed=0, flaw_rate=0.0):
        suite = build_suite(seed=seed)
        model = MockModel(flaw_rate=flaw_rate, seed=seed)
        return build_report("mock", model, suite, SUITE_PATH, seed, flaw_rate)

    def test_perfect_model_report_exploit_zero(self):
        report = self._run()
        assert report["model"] == "mock"
        assert report["suite_size"] == 1000
        assert report["verdict_counts"] == {"refusal": 900, "safe": 100, "exploit": 0}
        assert report["exploit_rows"] == []

    def test_flawed_model_report_counts_exploits_honestly(self):
        report = self._run(flaw_rate=1.0)
        assert report["verdict_counts"]["exploit"] == 900
        assert len(report["exploit_rows"]) == 900
        assert len(set(report["exploit_rows"])) == 900

    def test_per_category_counts(self):
        report = self._run()
        counts = report["per_category_counts"]
        assert set(counts) == CATEGORIES
        assert counts["jailbreak"]["total"] == 400
        assert counts["prompt_injection"]["total"] == 350
        assert counts["logic_bomb"]["total"] == 250
        total = sum(c["total"] for c in counts.values())
        assert total == 1000
        # verdicts within a category sum to its total
        for c in counts.values():
            assert c["refusal"] + c["safe"] + c["exploit"] == c["total"]
            for sub in c["subcategories"].values():
                assert sub["refusal"] + sub["safe"] + sub["exploit"] == sub["total"]

    def test_report_deterministic_verdicts(self):
        a = self._run(seed=3, flaw_rate=0.25)
        b = self._run(seed=3, flaw_rate=0.25)
        assert a["verdict_counts"] == b["verdict_counts"]
        assert a["exploit_rows"] == b["exploit_rows"]
        assert a["per_category_counts"] == b["per_category_counts"]

    def test_committed_suite_has_1000_valid_rows(self):
        assert SUITE_PATH.exists(), "data/redteam_suite.jsonl must be committed"
        rows = load_suite(SUITE_PATH)
        assert len(rows) == 1000
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))


class TestCLI:
    def _run_cli(self, *args, timeout=180):
        return subprocess.run(
            [sys.executable, "-m", "src.red_teamer", *args],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )

    def test_demo_command_mock(self, tmp_path):
        report = tmp_path / "redteam.json"
        result = self._run_cli(
            "--model", "mock", "--suite", str(SUITE_PATH), "--report", str(report)
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["model"] == "mock"
        assert data["suite_size"] == 1000
        assert data["verdict_counts"] == {"refusal": 900, "safe": 100, "exploit": 0}
        assert "timestamp" in data
        assert data["config"]["seed"] == 0

    def test_real_model_path_missing_exits_2(self, tmp_path):
        """Nonexistent model dir -> honest exit 2 with a clear message
        (no transformers import needed for this path)."""
        report = tmp_path / "redteam.json"
        missing = tmp_path / "no-such-model"
        result = self._run_cli(
            "--model", str(missing), "--suite", str(SUITE_PATH),
            "--report", str(report),
        )
        assert result.returncode == 2
        assert "not found" in result.stderr
        assert "--model mock" in result.stderr

    def test_flawed_mock_reports_exploits_honestly(self, tmp_path):
        report = tmp_path / "redteam.json"
        result = self._run_cli(
            "--model", "mock", "--mock-flaw-rate", "1.0",
            "--suite", str(SUITE_PATH), "--report", str(report), "--seed", "1",
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["verdict_counts"] == {"refusal": 0, "safe": 100, "exploit": 900}
        assert len(data["exploit_rows"]) == 900
        # report is clearly labeled as the mock stub
        assert data["model"] == "mock"

    def test_cli_deterministic(self, tmp_path):
        r1, r2 = tmp_path / "a.json", tmp_path / "b.json"
        for r in (r1, r2):
            result = self._run_cli(
                "--model", "mock", "--mock-flaw-rate", "0.25",
                "--suite", str(SUITE_PATH), "--report", str(r), "--seed", "9",
            )
            assert result.returncode == 0, result.stderr
        a = json.loads(r1.read_text(encoding="utf-8"))
        b = json.loads(r2.read_text(encoding="utf-8"))
        assert a["verdict_counts"] == b["verdict_counts"]
        assert a["exploit_rows"] == b["exploit_rows"]

    def test_bad_flaw_rate_exits_2(self, tmp_path):
        result = self._run_cli(
            "--model", "mock", "--mock-flaw-rate", "2.0",
            "--suite", str(SUITE_PATH), "--report", str(tmp_path / "r.json"),
        )
        assert result.returncode == 2

    def test_missing_suite_exits_2(self, tmp_path):
        result = self._run_cli(
            "--model", "mock", "--suite", str(tmp_path / "nope.jsonl"),
            "--report", str(tmp_path / "r.json"),
        )
        assert result.returncode == 2

    def test_invalid_suite_row_exits_2(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({"id": "rt-0000", "category": "jailbreak"}), encoding="utf-8")
        result = self._run_cli(
            "--model", "mock", "--suite", str(bad), "--report", str(tmp_path / "r.json")
        )
        assert result.returncode == 2

    def test_build_suite_writes_1000_rows(self, tmp_path):
        out = tmp_path / "suite.jsonl"
        result = self._run_cli("--build-suite", str(out), "--seed", "0")
        assert result.returncode == 0, result.stderr
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 1000
        for r in rows:
            assert validate_suite_row(r) == []

    def test_build_suite_matches_committed_file(self, tmp_path):
        out = tmp_path / "suite.jsonl"
        write_suite(out, seed=0)
        assert out.read_text(encoding="utf-8") == SUITE_PATH.read_text(encoding="utf-8")

    def test_load_model_mock(self):
        model = load_model("mock", 0.0, 0)
        assert model.label == "mock"

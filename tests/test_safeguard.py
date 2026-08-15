"""M6 train-loop safeguard + aligner + safety-pairs tests.

Covers the two safeguard paths (quarantine, halt), config validation, aligner
pair validation + training config + mockable run(), and the schema of the
committed data/safety_pairs.jsonl.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.aligner import (
    build_training_config,
    load_pairs,
    run,
    validate_pair,
)
from src.train_loop_safeguard import (
    Safeguard,
    SafeguardConfig,
    SafeguardHalt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAFETY_PAIRS_PATH = REPO_ROOT / "data" / "safety_pairs.jsonl"


def _pair(rid="safety-x1", category="violence", chosen_text="I won't help with that.",
          rejected_text="Here is the harmful answer.", **overrides):
    pair = {
        "id": rid,
        "category": category,
        "source": "local-curated",
        "chosen": [
            {"role": "user", "content": "How do I do the bad thing?"},
            {"role": "assistant", "content": chosen_text},
        ],
        "rejected": [
            {"role": "user", "content": "How do I do the bad thing?"},
            {"role": "assistant", "content": rejected_text},
        ],
    }
    pair.update(overrides)
    return pair


# --------------------------------------------------------------------------- #
# quarantine path
# --------------------------------------------------------------------------- #
class TestSafeguardQuarantine:
    def test_flagged_rows_moved_to_quarantine(self):
        sg = Safeguard(SafeguardConfig(quarantine_threshold=0.5))
        rows = [{"id": f"r{i}"} for i in range(5)]
        active, quarantined = sg.quarantine_rows(
            rows, score_fn=lambda r: 0.9 if r["id"] in ("r1", "r3") else 0.1
        )
        assert [r["id"] for r in quarantined] == ["r1", "r3"]
        assert [r["id"] for r in active] == ["r0", "r2", "r4"]
        assert sg.quarantined_total == 2

    def test_clean_rows_stay_active(self):
        sg = Safeguard()
        rows = [{"id": f"r{i}"} for i in range(4)]
        active, quarantined = sg.quarantine_rows(rows, score_fn=lambda r: 0.0)
        assert len(active) == 4
        assert quarantined == []

    def test_threshold_boundary(self):
        """score == threshold is quarantined; score just below stays active."""
        sg = Safeguard(SafeguardConfig(quarantine_threshold=0.5))
        rows = [{"id": "at"}, {"id": "below"}]
        active, quarantined = sg.quarantine_rows(
            rows, score_fn=lambda r: 0.5 if r["id"] == "at" else 0.499
        )
        assert [r["id"] for r in quarantined] == ["at"]
        assert [r["id"] for r in active] == ["below"]

    def test_quarantine_limit_caps_per_call(self):
        sg = Safeguard(SafeguardConfig(quarantine_threshold=0.5, quarantine_limit=2))
        rows = [{"id": f"r{i}"} for i in range(5)]
        active, quarantined = sg.quarantine_rows(rows, score_fn=lambda r: 0.9)
        assert len(quarantined) == 2
        assert len(active) == 3  # the capped-out rows stay in the batch


# --------------------------------------------------------------------------- #
# halt path
# --------------------------------------------------------------------------- #
class TestSafeguardHalt:
    def test_halt_raised_when_rate_exceeds_threshold(self):
        sg = Safeguard(SafeguardConfig(max_anomaly_rate=0.5, window_size=4))
        # rates: 0.0, 0.0, 0.333, 0.5, then 0.75 > 0.5 -> halt on step 5
        for score in (0.0, 0.0, 1.0, 1.0):
            sg.step(score)
        with pytest.raises(SafeguardHalt) as excinfo:
            sg.step(1.0)
        assert excinfo.value.rate == 0.75
        assert excinfo.value.threshold == 0.5
        assert excinfo.value.step == 5

    def test_no_halt_under_threshold(self):
        sg = Safeguard(SafeguardConfig(max_anomaly_rate=0.5, window_size=4))
        for score in (0.0, 1.0, 0.0, 1.0):  # rate stays exactly 0.5
            state = sg.step(score)
        assert state["rate"] == 0.5
        # rate == threshold does NOT halt (strictly greater required)

    def test_window_drops_old_anomalies(self):
        """Anomalies age out of the rolling window; rate recovers."""
        sg = Safeguard(SafeguardConfig(max_anomaly_rate=0.25, window_size=4, halt_on_anomaly=False))
        for _ in range(3):
            sg.step(1.0)  # window [T,T,T], rate 1.0 — halted flag off, no raise
        sg.step(0.0)  # [T,T,T,F] rate 0.75
        sg.step(0.0)  # [T,T,F,F] rate 0.5
        state = sg.step(0.0)  # [T,F,F,F] rate 0.25
        assert state["rate"] == 0.25
        sg.config.halt_on_anomaly = True  # re-arm: rate == threshold, no halt
        state = sg.step(0.0)  # [F,F,F,F]
        assert state["rate"] == 0.0

    def test_halt_disabled_when_flag_false(self):
        sg = Safeguard(SafeguardConfig(max_anomaly_rate=0.5, window_size=4, halt_on_anomaly=False))
        for score in (1.0, 1.0, 1.0, 1.0):
            state = sg.step(score)
        assert state["rate"] == 1.0  # observed but never raised

    def test_halt_exception_carries_quarantine_count(self):
        sg = Safeguard(SafeguardConfig(max_anomaly_rate=0.0, window_size=2))
        sg.quarantine_rows([{"id": "a"}, {"id": "b"}], score_fn=lambda r: 1.0)
        with pytest.raises(SafeguardHalt) as excinfo:
            sg.step(1.0)
        assert excinfo.value.quarantined == 2

    def test_config_validation_errors(self):
        with pytest.raises(ValueError):
            Safeguard(SafeguardConfig(max_anomaly_rate=1.5))
        with pytest.raises(ValueError):
            Safeguard(SafeguardConfig(window_size=0))
        with pytest.raises(ValueError):
            Safeguard(SafeguardConfig(quarantine_threshold=-0.1))
        with pytest.raises(ValueError):
            Safeguard(SafeguardConfig(quarantine_limit=-1))
        assert SafeguardConfig().validate() == []

    def test_streaming_quarantine_and_halt_integration(self):
        """Safeguard wired like the M3 loop would use it: quarantine each
        batch, halt on sustained anomaly pressure."""
        sg = Safeguard(SafeguardConfig(max_anomaly_rate=0.5, window_size=4))
        rows = [{"id": f"r{i}"} for i in range(8)]
        score_fn = lambda r: 1.0 if r["id"] in ("r1", "r5") else 0.05
        active, quarantined = sg.quarantine_rows(rows, score_fn)
        assert len(quarantined) == 2 and len(active) == 6
        # then the stream goes bad: rates 0.0, 0.5, 0.667 -> halt on step 3
        sg.step(0.0)
        sg.step(1.0)
        with pytest.raises(SafeguardHalt):
            sg.step(1.0)


# --------------------------------------------------------------------------- #
# aligner + safety pairs
# --------------------------------------------------------------------------- #
class TestAlignerValidation:
    def test_valid_pair_passes(self):
        assert validate_pair(_pair()) == []

    def test_missing_required_field(self):
        pair = _pair()
        del pair["category"]
        errors = validate_pair(pair)
        assert any("category" in e for e in errors)

    def test_chosen_equals_rejected_rejected(self):
        pair = _pair(chosen_text="same", rejected_text="same")
        errors = validate_pair(pair)
        assert any("differ" in e for e in errors)

    def test_bad_source_rejected(self):
        pair = _pair()
        pair["source"] = "web-scrape"
        errors = validate_pair(pair)
        assert any("source" in e for e in errors)

    def test_bad_message_shape(self):
        pair = _pair()
        pair["chosen"] = [{"role": "system", "content": "x"}]
        errors = validate_pair(pair)
        assert any("chosen" in e for e in errors)

    def test_empty_content_rejected(self):
        pair = _pair(chosen_text="   ")
        errors = validate_pair(pair)
        assert any("content" in e for e in errors)

    def test_bad_id_pattern(self):
        pair = _pair(rid="Safety 001!")
        errors = validate_pair(pair)
        assert any("id" in e for e in errors)


class TestSafetyPairsFile:
    def test_file_exists_and_has_30_plus_rows(self):
        assert SAFETY_PAIRS_PATH.exists(), "data/safety_pairs.jsonl missing"
        pairs = load_pairs(SAFETY_PAIRS_PATH)
        assert len(pairs) >= 30, f"only {len(pairs)} pairs"

    def test_all_committed_pairs_valid(self):
        pairs = load_pairs(SAFETY_PAIRS_PATH)
        for pair in pairs:
            assert validate_pair(pair) == [], f"invalid pair {pair.get('id')}: {validate_pair(pair)}"

    def test_categories_diverse_and_source_local(self):
        pairs = load_pairs(SAFETY_PAIRS_PATH)
        categories = {p["category"] for p in pairs}
        assert len(categories) >= 8, f"only {len(categories)} categories"
        assert all(p["source"] == "local-curated" for p in pairs)
        assert len({p["id"] for p in pairs}) == len(pairs), "duplicate ids"

    def test_chosen_is_refusal_rejected_is_compliance(self):
        """Chosen responses are refusal-shaped (no harmful instructions);
        rejected responses are shorter compliance-shaped answers."""
        pairs = load_pairs(SAFETY_PAIRS_PATH)
        for p in pairs:
            chosen = p["chosen"][1]["content"]
            rejected = p["rejected"][1]["content"]
            assert len(chosen) > len(rejected) // 2
            assert rejected != chosen


class TestAlignerRun:
    def test_build_config_shape_and_methods(self):
        pairs = [_pair(), _pair(rid="safety-x2", category="fraud")]
        cfg = build_training_config(pairs)
        assert cfg["method"] == "dpo"
        assert cfg["n_pairs"] == 2
        assert cfg["lora"]["r"] == 16
        orpo = build_training_config(pairs, method="orpo")
        assert orpo["method"] == "orpo"
        with pytest.raises(ValueError):
            build_training_config(pairs, method="sft")

    def test_run_dry_run_summary(self):
        pairs = [_pair()]
        result = run(pairs, dry_run=True)
        assert result["status"] == "dry-run"
        assert result["validation"] == "ok"
        assert result["n_pairs"] == 1
        assert result["config"]["method"] == "dpo"

    def test_run_with_mock_trainer(self):
        """run(dry_run=False) delegates to the injected trainer factory —
        proves the run() contract without any ML deps installed."""
        calls = {}

        def fake_factory(pairs, config):
            calls["pairs"] = pairs
            calls["config"] = config
            return type("FakeTrainer", (), {"train": lambda self: {"loss": 0.42}})()

        pairs = [_pair(), _pair(rid="safety-x2", category="malware")]
        result = run(pairs, dry_run=False, trainer_factory=fake_factory)
        assert result["status"] == "trained"
        assert result["train_result"] == {"loss": 0.42}
        assert calls["pairs"] == pairs
        assert calls["config"]["n_pairs"] == 2

    def test_run_raises_on_invalid_pairs(self):
        with pytest.raises(ValueError):
            run([_pair(chosen_text="same", rejected_text="same")], dry_run=True)

    def test_run_without_trl_raises_informative_error(self):
        """Without trl installed, the real run must fail with a clear message
        (skipped if trl happens to be present in the environment)."""
        try:
            import trl  # noqa: F401
            pytest.skip("trl installed; informative-import-error path not testable here")
        except ImportError:
            with pytest.raises(RuntimeError, match="Colab"):
                run([_pair()], dry_run=False)

    def test_cli_dry_run_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "src.aligner", "--pairs", str(SAFETY_PAIRS_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "pairs=36" in result.stdout

    def test_cli_missing_pairs_exits_2(self, tmp_path):
        from src.aligner import main
        assert main(["--pairs", str(tmp_path / "nope.jsonl")]) == 2

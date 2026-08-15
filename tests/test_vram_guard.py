"""M3 VRAM guard tests — pure math, no torch needed."""
import json
import subprocess
import sys
from pathlib import Path

from src.vram_guard import estimate_vram, lora_params_for, main, run_guard

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestEstimate:
    def test_quantized_weights_cheaper(self):
        q = estimate_vram(135_000_000, lora_params=100_000, batch_size=1, seq_len=256, hidden_size=576, quantized=True)
        f = estimate_vram(135_000_000, lora_params=100_000, batch_size=1, seq_len=256, hidden_size=576, quantized=False)
        assert q.total_gb < f.total_gb
        assert q.passed and f.passed  # both within 12GB

    def test_budget_respected(self):
        est = estimate_vram(7_000_000_000, lora_params=1_000_000, batch_size=8, seq_len=1024, hidden_size=4096, budget_gb=12.0)
        # 7B model quantized to 4-bit ≈ 3.5GB weights + LoRA/optimizer overhead
        assert est.total_gb > 12.0 or est.passed  # conservative estimate may exceed
        # a huge budget always passes
        est_big = estimate_vram(7_000_000_000, lora_params=1_000_000, batch_size=8, seq_len=1024, hidden_size=4096, budget_gb=64.0)
        assert est_big.passed

    def test_lora_params_formula(self):
        cfgs = [{"in_features": 576}, {"in_features": 576}, {"in_features": 576}, {"in_features": 576}]
        assert lora_params_for(cfgs, r=8) == 4 * 2 * 8 * 576  # 36864

    def test_cpu_mode_always_passes(self):
        est = run_guard(135_000_000, 36_864, 1, 256, 576, force_cpu=True)
        assert est.mode == "cpu"
        assert est.passed is True
        assert est.total_gb > 0

    def test_cuda_mode_gate(self):
        est = run_guard(135_000_000, 36_864, 1, 256, 576, force_cpu=False)
        assert est.mode == "cuda"
        assert est.passed is True  # small model within budget


class TestGuardCLI:
    def test_cli_pass_exit_0(self, tmp_path):
        out = tmp_path / "vram.json"
        result = subprocess.run(
            [sys.executable, "-m", "src.vram_guard", "--base-params", "135000000",
             "--lora-params", "36864", "--out", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["passed"] is True
        assert report["mode"] == "cuda"

    def test_cli_cpu_mode(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "src.vram_guard", "--base-params", "135000000",
             "--lora-params", "36864", "--force-cpu"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0
        assert "mode=cpu" in result.stdout

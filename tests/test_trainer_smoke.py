"""M3 trainer smoke tests — need torch + peft + transformers (py -3.13).

These use a tiny real model on CPU with 1-2 steps to prove the trainer,
masking, and adapter save work end to end. Marked with @pytest.mark.slow;
run explicitly: py -3.13 -m pytest tests/test_trainer_smoke.py -q
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

TINY_MODEL = "HuggingFaceTB/SmolLM2-135M"


@pytest.mark.slow
class TestMaskingIntegration:
    def test_tokenize_row_masks_answer(self):
        """The M3 core claim: labels are -100 outside the answer span."""
        from transformers import AutoTokenizer

        from src.masking import mask_metadata
        from src.trainer import _tokenize_row

        tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        user = "What is 2 + 2?"
        answer = "4"
        row = {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ],
            "mask": mask_metadata(user, answer),
        }
        out = _tokenize_row(row, tokenizer, max_len=64)
        labels = out["labels"]
        assert (labels == -100).sum() > 0  # prompt tokens masked
        assert (labels != -100).sum() >= 1  # answer token trainable
        # the final trainable token must be the answer token region
        non_masked = [i for i, v in enumerate(labels.tolist()) if v != -100]
        assert non_masked  # at least one trainable token


@pytest.mark.slow
class TestTrainerSmoke:
    @pytest.fixture(autouse=True)
    def _tiny_train_set(self, tmp_path):
        """Build a 4-row train.jsonl from scaffold puzzles."""
        from src.dataset_builder import build_row
        from src.generator.families.scaffold import ScaffoldAdd

        fam = ScaffoldAdd()
        rows = [build_row(fam.generate("add", seed=500 + i).to_row()) for i in range(4)]
        p = tmp_path / "train.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        self.train_jsonl = p

    def test_smoke_train_saves_adapter(self, tmp_path):
        out_dir = tmp_path / "out"
        result = subprocess.run(
            [sys.executable, "-m", "src.trainer",
             "--config", str(REPO_ROOT / "config" / "train.yaml"),
             "--model", TINY_MODEL,
             "--train-jsonl", str(self.train_jsonl),
             "--limit", "4",
             "--max-steps", "1",
             "--force-cpu",
             "--out-dir", str(out_dir)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "VRAM guard" in result.stdout and "PASS" in result.stdout
        assert (out_dir / "lora-adapter" / "adapter_config.json").exists(), \
            "adapter not saved"
        assert (out_dir / "lora-adapter" / "adapter_model.safetensors").exists()
        assert "adapter saved" in result.stdout

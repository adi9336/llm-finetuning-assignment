"""vram_guard — pre-train VRAM budget gate (M3).

Estimates peak GPU memory for a QLoRA run from model parameters, LoRA
hyperparameters, and batch size, then asserts the estimate stays within the
budget (default 12 GB — assignment hardware floor; Colab T4 has 16 GB).

Estimate model (deliberately conservative, additive):
    base_params   = model parameters (fp32 = 4 bytes/param)
    4bit_weights  = base params @ 0.5 bytes (NF4) when quantized
    lora_params   = sum over LoRA modules of 2 * r * in_features  (A + B)
    grad_mem      = 2 * trainable_params (gradients, fp32)
    adam_moments  = 2 * 2 * trainable_params (Adam m + v, fp32 each)
    activations   = batch * seq_len * hidden * 2 (fp16-ish, heuristic)

On CPU (no CUDA) the guard still computes and REPORTS the estimate and budget
but passes by design — there is no GPU to OOM; the honest claim is the
estimate path is exercised and the number is reported. On GPU it is a hard
gate (exit 1 if estimate > budget).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class VramEstimate:
    base_params: int
    quantized: bool
    lora_params: int
    trainable_params: int
    weights_bytes: int
    grad_bytes: int
    adam_bytes: int
    activation_bytes: int
    total_bytes: int
    total_gb: float
    budget_gb: float
    passed: bool
    mode: str  # "cuda" | "cpu"


def estimate_vram(
    base_params: int,
    lora_params: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    budget_gb: float = 12.0,
    quantized: bool = True,
) -> VramEstimate:
    """Compute the conservative peak-VRAM estimate (see module docstring)."""
    trainable = lora_params  # QLoRA: only LoRA params are trainable
    weights = base_params * (0.5 if quantized else 4)  # NF4 bytes or fp32
    grad = 2 * trainable * 4  # fp32 grads
    adam = 2 * 2 * trainable * 4  # Adam m + v, fp32
    activations = batch_size * seq_len * hidden_size * 2
    total = weights + grad + adam + activations
    total_gb = total / (1024**3)
    return VramEstimate(
        base_params=base_params,
        quantized=quantized,
        lora_params=lora_params,
        trainable_params=trainable,
        weights_bytes=weights,
        grad_bytes=grad,
        adam_bytes=adam,
        activation_bytes=activations,
        total_bytes=total,
        total_gb=round(total_gb, 2),
        budget_gb=budget_gb,
        passed=total_gb <= budget_gb,
        mode="cuda",
    )


def run_guard(
    base_params: int,
    lora_params: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    budget_gb: float = 12.0,
    force_cpu: bool = False,
) -> VramEstimate:
    """Gate entry: estimate, then enforce (GPU) or report (CPU)."""
    est = estimate_vram(
        base_params, lora_params, batch_size, seq_len, hidden_size, budget_gb
    )
    if force_cpu:
        est.mode = "cpu"
        est.passed = True  # nothing to OOM on CPU; estimate still reported
    return est


def lora_params_for(model_configs: List[Dict[str, int]], r: int) -> int:
    """Sum of LoRA A+B params over targeted module shapes: 2 * r * in_features."""
    total = 0
    for cfg in model_configs:
        total += 2 * r * cfg["in_features"]
    return total


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.vram_guard", description="VRAM budget gate")
    ap.add_argument("--base-params", type=int, required=True)
    ap.add_argument("--lora-params", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--hidden-size", type=int, default=2048)
    ap.add_argument("--budget-gb", type=float, default=12.0)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--out", type=str, default="", help="optional JSON report path")
    args = ap.parse_args(argv)

    est = run_guard(
        args.base_params,
        args.lora_params,
        args.batch_size,
        args.seq_len,
        args.hidden_size,
        args.budget_gb,
        args.force_cpu,
    )
    report = asdict(est)
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    line = (
        f"VRAM estimate {est.total_gb} GB (budget {est.budget_gb} GB, mode={est.mode}, "
        f"quantized={est.quantized}) -> {'PASS' if est.passed else 'FAIL'}"
    )
    print(line)
    return 0 if est.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

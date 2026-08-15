"""merge — merge a LoRA adapter into the base model (M3).

Usage:
    python -m src.merge --base <model_id_or_dir> --adapter data/out/lora-adapter --out data/out/lora-merged
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel  # type: ignore
    _HAS_PEFT = True
except Exception:  # pragma: no cover
    _HAS_PEFT = False


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.merge", description="Merge LoRA adapter into base model")
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args(argv)

    if not _HAS_PEFT:
        print("ERROR: peft not installed", file=sys.stderr)
        return 2

    device = "cpu" if (args.force_cpu or not torch.cuda.is_available()) else "cuda"
    if torch.cuda.is_available() and not args.force_cpu:
        # Colab T4 path: load base 4-bit on GPU (7B fp32 would OOM 12GB RAM)
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base, device_map=device)
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    tokenizer.save_pretrained(str(out_dir))
    print(f"merged model saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

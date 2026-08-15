"""trainer — QLoRA fine-tuner with dynamic prompt masking (M3).

Design:
- GPU (CUDA available): model loaded 4-bit NF4 via BitsAndBytesConfig, LoRA
  applied with peft, gradient checkpointing on, dynamic answer-only masking.
- CPU (--force-cpu or no CUDA): plain fp32 load (bitsandbytes 4-bit is not
  supported on CPU), LoRA still applied, same masking — the CI smoke path.
- Dataset: data/train.jsonl rows from M2 — messages (user/assistant) plus a
  char-span mask. We tokenize user + "\\n\\n" + assistant with
  return_offsets_mapping and use src.masking.token_mask_from_offsets to build
  labels where only answer tokens are trained (-100 elsewhere).
- VRAM guard runs BEFORE training: hard gate on GPU, reported pass on CPU.
- Outputs: adapter + merged weights under data/out/.

Deterministic recipe templates, no LLM in the data path, no external APIs
(grep gate: no openai/requests/urllib/httpx).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.masking import token_mask_from_offsets
from src.vram_guard import lora_params_for, run_guard

try:
    from peft import LoraConfig, get_peft_model  # type: ignore
    _HAS_PEFT = True
except Exception:  # pragma: no cover
    _HAS_PEFT = False

SEP = "\n\n"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return cfg


def _tokenize_row(row: Dict[str, Any], tokenizer, max_len: int) -> Dict[str, Any]:
    """Build input_ids/attention_mask/labels with answer-only masking."""
    user = row["messages"][0]["content"]
    assistant = row["messages"][1]["content"]
    text = user + SEP + assistant
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc["offset_mapping"][0].tolist()
    mask_start = row["mask"]["answer_start_char"]
    mask_end = row["mask"]["answer_end_char"]
    token_mask = token_mask_from_offsets(offsets, mask_start, mask_end)

    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]
    labels = input_ids.clone()
    for i, is_answer in enumerate(token_mask):
        if not is_answer:
            labels[i] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataset(train_jsonl: Path, tokenizer, max_len: int, limit: int | None = None) -> Dataset:
    rows: List[Dict[str, Any]] = []
    with open(train_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    ds = Dataset.from_list(rows)
    return ds.map(lambda r: _tokenize_row(r, tokenizer, max_len), remove_columns=ds.column_names)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.trainer", description="QLoRA trainer with dynamic masking")
    ap.add_argument("--config", type=str, required=True, help="config/train.yaml")
    ap.add_argument("--max-steps", type=int, default=None, help="override max steps (smoke)")
    ap.add_argument("--model", type=str, default=None, help="override base model id")
    ap.add_argument("--force-cpu", action="store_true", help="disable 4-bit / CUDA path")
    ap.add_argument("--train-jsonl", type=str, default=None, help="override train set path")
    ap.add_argument("--limit", type=int, default=None, help="limit dataset rows (smoke)")
    ap.add_argument("--out-dir", type=str, default=None, help="override output dir")
    ap.add_argument("--save-steps", type=int, default=250,
                    help="checkpoint adapter every N steps (Colab session-loss protection)")
    ap.add_argument("--resume-from-checkpoint", type=str, default=None,
                    help="path to a checkpoints/ dir to resume training from")
    ap.add_argument("--drive-sync", type=str, default=None,
                    help="Colab: copy each checkpoint to this Drive dir after every save "
                         "(session-loss protection — survives VM death)")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    model_id = args.model or cfg["model"]["id"]
    out_dir = Path(args.out_dir or cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not args.force_cpu

    # ---- VRAM guard (hard gate on GPU, report on CPU) ----
    vram = run_guard(
        base_params=cfg["vram_guard"]["base_params"],
        lora_params=lora_params_for(cfg["vram_guard"]["lora_modules"], cfg["lora"]["r"]),
        batch_size=cfg["train"]["per_device_train_batch_size"],
        seq_len=cfg["train"].get("max_seq_len", 256),
        hidden_size=cfg["vram_guard"]["hidden_size"],
        budget_gb=cfg["vram_guard"]["budget_gb"],
        force_cpu=not use_cuda,
    )
    print(
        f"VRAM guard: estimate {vram.total_gb} GB vs budget {vram.budget_gb} GB "
        f"(mode={vram.mode}) -> {'PASS' if vram.passed else 'FAIL'}"
    )
    if not vram.passed:
        print("VRAM guard FAILED — aborting before model load.", file=sys.stderr)
        return 1

    # ---- tokenizer + model ----
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_cuda:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)  # fp32 CPU smoke
    model.gradient_checkpointing_enable()

    # ---- LoRA ----
    if not _HAS_PEFT:
        print("ERROR: peft not installed", file=sys.stderr)
        return 2
    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- dataset ----
    train_jsonl = Path(args.train_jsonl or cfg["dataset"]["train_jsonl"])
    ds = build_dataset(train_jsonl, tokenizer, cfg["train"].get("max_seq_len", 256), args.limit)

    # ---- training ----
    ckpt_dir = out_dir / "checkpoints"
    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=cfg["train"]["learning_rate"],
        max_steps=args.max_steps or cfg["train"]["max_steps"],
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, args.save_steps),
        save_total_limit=3,
        report_to=[],
        use_cpu=not use_cuda,
        remove_unused_columns=False,
    )
    resume = args.resume_from_checkpoint
    if resume is not None and not Path(resume).exists():
        print(f"WARNING: resume path not found ({resume}) — starting fresh", file=sys.stderr)
        resume = None
    trainer = Trainer(model=model, args=targs, train_dataset=ds)
    if args.drive_sync:
        from transformers import TrainerCallback

        drive_dir = Path(args.drive_sync)

        class DriveSync(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):  # noqa: ARG002
                src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if src.exists():
                    dst = drive_dir / f"checkpoint-{state.global_step}"
                    dst.mkdir(parents=True, exist_ok=True)
                    import shutil

                    for item in src.iterdir():
                        if item.is_dir():
                            shutil.copytree(item, dst / item.name, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dst / item.name)
                    print(f"[drive-sync] checkpoint-{state.global_step} -> {dst}")

        trainer.add_callback(DriveSync())
    trainer.train(resume_from_checkpoint=resume)

    # ---- save adapter ----
    adapter_dir = out_dir / "lora-adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"adapter saved -> {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

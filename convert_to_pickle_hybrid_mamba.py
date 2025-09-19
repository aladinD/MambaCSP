#!/usr/bin/env python3
"""
convert_to_pickle.py

Convert training checkpoints (state_dict or already-pickled) into a pickled nn.Module
that your evaluation script can load with:
    model = torch.load(PATH, map_location=device, weights_only=False).to(device).eval()

Supports:
- --backbone gpt2
- --backbone mamba (HF-pretrained OR compact mamba-ssm)
- Thin attention mixer (add_mixer, mixer_layers, mixer_heads, mixer_rank, dropouts)

Examples
--------
# GPT-2 checkpoint (state_dict) -> pickled model
python convert_to_pickle.py \
  --backbone gpt2 \
  --in  model_weights/train_acc/full_shot_tdd/U2U_LLM4CP.pth \
  --out model_weights/train_acc/full_shot_tdd/U2U_LLM4CP_pickled.pth

# Mamba HF-pretrained checkpoint (state_dict) -> pickled model
python convert_to_pickle.py \
  --backbone mamba --use-hf-mamba --hf-name state-spaces/mamba-130m-hf \
  --add-mixer --mixer-layers 1 --mixer-heads 2 --mixer-rank 128 \
  --prev-len 16 --pred-len 4 --K 48 --UQh 1 --UQv 1 --BQh 1 --BQv 1 \
  --in  model_weights/train_acc/full_shot_tdd/mamba/U2U_LLM4CP.pth \
  --out model_weights/train_acc/full_shot_tdd/mamba/U2U_LLM4CP_mamba_pickled.pth

# Mamba compact (mamba-ssm) checkpoint (state_dict) -> pickled
python convert_to_pickle.py \
  --backbone mamba \
  --d-model 768 --mamba-layers 6 --d-state 16 --d-conv 4 --expand 2 \
  --add-mixer --mixer-layers 1 --mixer-heads 2 --mixer-rank 128 \
  --in  model_weights/train_acc/full_shot_tdd/mamba/U2U_LLM4CP.pth \
  --out model_weights/train_acc/full_shot_tdd/mamba/U2U_LLM4CP_mamba_pickled.pth
"""

import argparse
import os
import torch
from collections import OrderedDict


# -------------------- utils --------------------
def _maybe_strip_module_prefix(state):
    """Remove 'module.' prefix added by DDP."""
    if isinstance(state, (dict, OrderedDict)) and any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _load_any(path):
    """Load object (dict or nn.Module). PyTorch>=2.6 defaults to weights_only=True; we must disable for pickled nn.Module."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # older torch without weights_only kwarg
        return torch.load(path, map_location="cpu")


# -------------------- builders --------------------
def _build_gpt(args):
    from models.GPT4CP import Model
    # Build on CPU: use_gpu=0 so we don't touch GPUs here.
    return Model(
        gpu_id=0, use_gpu=0,
        pred_len=args.pred_len, prev_len=args.prev_len,
        UQh=args.UQh, UQv=args.UQv, BQh=args.BQh, BQv=args.BQv
    )


def _build_mamba(args):
    from models.MAMBA import Model
    # Match the exact training config (HF vs compact + mixer settings!)
    return Model(
        use_hf=args.use_hf_mamba,
        hf_name=args.hf_name,
        d_model=args.d_model,
        mamba_layers=args.mamba_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        # thin attention mixer
        add_mixer=args.add_mixer,
        mixer_layers=args.mixer_layers,
        mixer_heads=args.mixer_heads,
        mixer_rank=args.mixer_rank,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        # task & dims
        pred_len=args.pred_len, prev_len=args.prev_len,
        use_gpu=0, gpu_id=0,        # build on CPU
        K=args.K, UQh=args.UQh, UQv=args.UQv, BQh=args.BQh, BQv=args.BQv
    )


def _build_model(args):
    if args.backbone == "gpt2":
        return _build_gpt(args)
    elif args.backbone == "mamba":
        return _build_mamba(args)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")


# -------------------- main convert --------------------
def convert(in_path, out_path, args):
    obj = _load_any(in_path)

    # A) already a pickled nn.Module -> just re-save (re-pickle under current env)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        torch.save(obj, out_path)
        print(f"[OK] Input was already a pickled model. Re-saved to: {out_path}")
        return

    # B) checkpoint dict with 'model_state'
    if isinstance(obj, dict) and "model_state" in obj:
        state = _maybe_strip_module_prefix(obj["model_state"])
        model = _build_model(args)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("  missing (first 10):", missing[:10])
            if unexpected:
                print("  unexpected (first 10):", unexpected[:10])
        model.eval()
        torch.save(model, out_path)
        print(f"[OK] Converted checkpoint -> pickled model saved to: {out_path}")
        return

    # C) raw state_dict (tensor map)
    if isinstance(obj, (dict, OrderedDict)) and all(torch.is_tensor(v) for v in obj.values()):
        state = _maybe_strip_module_prefix(obj)
        model = _build_model(args)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("  missing (first 10):", missing[:10])
            if unexpected:
                print("  unexpected (first 10):", unexpected[:10])
        model.eval()
        torch.save(model, out_path)
        print(f"[OK] Converted raw state_dict -> pickled model saved to: {out_path}")
        return

    raise TypeError(
        f"Unrecognized checkpoint at {in_path}. "
        f"Expected nn.Module OR dict with 'model_state' OR raw state_dict."
    )


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Convert LLM4CP checkpoints to pickled nn.Module for evaluation")
    p.add_argument("--backbone", type=str, default="mamba", choices=["mamba", "gpt2"])
    p.add_argument("--in",  dest="in_path",  required=True, help="Input .pth (pickled model or state_dict checkpoint)")
    p.add_argument("--out", dest="out_path", required=True, help="Output pickled model path (.pth)")

    # Common model hyperparams (must match training)
    p.add_argument("--prev-len", type=int, default=16)
    p.add_argument("--pred-len", type=int, default=4)
    p.add_argument("--UQh", type=int, default=1)
    p.add_argument("--UQv", type=int, default=1)
    p.add_argument("--BQh", type=int, default=1)
    p.add_argument("--BQv", type=int, default=1)
    p.add_argument("--K", type=int, default=48)

    # Mamba-specific (only used when --backbone mamba)
    p.add_argument("--use-hf-mamba", action="store_true",
                   help="Instantiate HF Mamba backbone; else compact custom mamba-ssm.")
    p.add_argument("--hf-name", type=str, default="state-spaces/mamba-130m-hf",
                   help="HF model id (if --use-hf-mamba).")
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--mamba-layers", type=int, default=6)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)

    # Thin attention mixer (match your training!)
    p.add_argument("--add-mixer", action="store_true", help="Enable thin attention mixer")
    p.add_argument("--mixer-layers", type=int, default=1)
    p.add_argument("--mixer-heads",  type=int, default=2)
    p.add_argument("--mixer-rank",   type=int, default=128)
    p.add_argument("--attn-dropout", type=float, default=0.0)
    p.add_argument("--proj-dropout", type=float, default=0.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    convert(args.in_path, args.out_path, args)

# Usage
# python convert_to_pickle_hybrid_mamba.py \
#   --backbone mamba \
#   --use-hf-mamba \
#   --hf-name state-spaces/mamba-130m-hf \
#   --add-mixer --mixer-layers 1 --mixer-heads 2 --mixer-rank 128 \
#   --prev-len 16 --pred-len 4 --K 48 --UQh 1 --UQv 1 --BQh 1 --BQv 1 \
#   --in model_weights/train_acc/full_shot_tdd/smaller_batch_mamba_test/U2U_LLM4CP.pth \
#   --out model_weights/train_acc/full_shot_tdd/smaller_batch_mamba_test/U2U_LLM4CP_mamba.pth


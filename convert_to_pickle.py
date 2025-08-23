# convert_to_pickle.py
# Usage examples:
#   python convert_to_pickle.py --in model_weights/train_acc/full_shot_tdd/U2U_LLM4CP.pth \
#                               --out model_weights/train_acc/full_shot_tdd/U2U_LLM4CP_pickled.pth
#   # (optional) override model hyperparams if you trained with non-defaults:
#   python convert_to_pickle.py --in ... --out ... --prev-len 16 --pred-len 4 --UQh 1 --UQv 1 --BQh 1 --BQv 1
#
# Notes:
# - Runs entirely on CPU; no GPU needed.
# - If the input is already a pickled model (nn.Module), it just re-saves it.
# - If the input is a checkpoint dict, it reconstructs Model(...) and loads model_state.

import argparse
import os
import torch
from collections import OrderedDict

# import your model class
from models.GPT4CP import Model

def _maybe_strip_module_prefix(state):
    """Remove 'module.' prefix that appears when saving from DDP."""
    if isinstance(state, (dict, OrderedDict)) and any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

def load_checkpoint_any(path):
    """Load whatever is at path on CPU (dict/OrderedDict or nn.Module)."""
    return torch.load(path, map_location="cpu")

def build_model_from_args(args):
    # gpu_id is irrelevant on CPU; set to 0
    return Model(
        gpu_id=0,
        pred_len=args.pred_len,
        prev_len=args.prev_len,
        UQh=args.UQh, UQv=args.UQv,
        BQh=args.BQh, BQv=args.BQv
    )

def convert(in_path, out_path, args):
    obj = load_checkpoint_any(in_path)

    # Case A: already a pickled model (nn.Module) -> just re-save (or copy)
    if isinstance(obj, torch.nn.Module):
        model = obj.eval()  # ensure eval() for inference use
        torch.save(model, out_path)
        print(f"[OK] Input was already a pickled model. Re-saved to: {out_path}")
        return

    # Case B: training checkpoint dict: expect key "model_state"
    if isinstance(obj, dict) and "model_state" in obj:
        state = obj["model_state"]
        state = _maybe_strip_module_prefix(state)

        model = build_model_from_args(args)
        # Be tolerant if tiny key diffs exist
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:   print("  missing keys (first 10):", missing[:10])
            if unexpected:print("  unexpected keys (first 10):", unexpected[:10])

        model.eval()
        torch.save(model, out_path)
        print(f"[OK] Converted checkpoint -> pickled model saved to: {out_path}")
        return

    # Case C: pure state_dict saved directly (no wrapper dict)
    if isinstance(obj, (dict, OrderedDict)) and all(torch.is_tensor(v) for v in obj.values()):
        state = _maybe_strip_module_prefix(obj)
        model = build_model_from_args(args)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:   print("  missing keys (first 10):", missing[:10])
            if unexpected:print("  unexpected keys (first 10):", unexpected[:10])

        model.eval()
        torch.save(model, out_path)
        print(f"[OK] Converted raw state_dict -> pickled model saved to: {out_path}")
        return

    raise TypeError(
        f"Unrecognized checkpoint format at {in_path}. "
        f"Expected nn.Module OR dict with 'model_state' OR raw state_dict."
    )

def parse_args():
    p = argparse.ArgumentParser(description="Convert LLM4CP state-dict checkpoint to pickled model")
    p.add_argument("--in",  dest="in_path",  required=True, help="Input checkpoint (.pth)")
    p.add_argument("--out", dest="out_path", required=True, help="Output pickled model path (.pth)")

    # Model hyperparams (must match how you trained)
    p.add_argument("--prev-len", type=int, default=16)
    p.add_argument("--pred-len", type=int, default=4)
    p.add_argument("--UQh", type=int, default=1)
    p.add_argument("--UQv", type=int, default=1)
    p.add_argument("--BQh", type=int, default=1)
    p.add_argument("--BQv", type=int, default=1)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    convert(args.in_path, args.out_path, args)

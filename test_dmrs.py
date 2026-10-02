"""
DMRS model evaluation (GPT-2 or Mamba).

For each configured pattern and mode (TDD/FDD), evaluate a single checkpoint
across 10 user speeds and save:
  - NMSE, SE, SE0, and SE/SE0 per speed.
"""

import os
from collections import OrderedDict
import argparse

import numpy as np
import torch
import torch.nn as nn
import tqdm
import hdf5storage
from einops import rearrange

from data import LoadBatch_ofdm_2, noise, Transform_TDD_FDD
from metrics import NMSELoss, SE_Loss


DEFAULT_PATTERNS = [
    "pattern1_type1_sparse",
    "pattern2_type2_densefreq",
    "pattern3_highmob_dense",
]

RESULTS_ROOT = "dmrs_results"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DMRS checkpoint on configured datasets.")
    p.add_argument("--backbone", type=str, default="gpt", choices=["gpt", "gpt2", "mamba"],
                   help="Backbone used for the checkpoint.")
    p.add_argument("--data-root", type=str, default="./dmrs_datasets",
                   help="Root directory for dmrs_datasets")
    p.add_argument("--weight-root", type=str, default="./dmrs_model_weights",
                   help="Root directory for model weight checkpoints.")
    p.add_argument("--results-root", type=str, default=RESULTS_ROOT,
                   help="Directory to save evaluation CSVs.")
    p.add_argument("--weight-subdir", type=str, default="",
                   help="Optional subdirectory under each <pattern>/<mode> weight folder.")
    p.add_argument("--weight-name-u2u", type=str, default="U2U_LLM4CP.pth",
                   help="Filename pattern for TDD checkpoints.")
    p.add_argument("--weight-name-u2d", type=str, default="U2D_LLM4CP.pth",
                   help="Filename pattern for FDD checkpoints.")
    p.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS,
                   help="Patterns to evaluate.")
    p.add_argument("--bs", type=int, default=64, help="Batch size for evaluation.")
    p.add_argument("--snr-db", type=float, default=18, help="AWGN SNR used for normalization test.")

    # Model/task dims (must match training)
    p.add_argument("--pred-len", type=int, default=4)
    p.add_argument("--prev-len", type=int, default=16)
    p.add_argument("--K", type=int, default=48)
    p.add_argument("--UQh", type=int, default=1)
    p.add_argument("--UQv", type=int, default=1)
    p.add_argument("--BQh", type=int, default=1)
    p.add_argument("--BQv", type=int, default=1)

    # Mamba options (only needed when --backbone mamba)
    p.add_argument("--use-hf-mamba", action="store_true",
                   help="Use HF-pretrained Mamba.")
    p.add_argument("--hf-name", type=str, default="state-spaces/mamba-370m-hf",
                   help="HF model id for Mamba backbone.")
    p.add_argument("--d-model", type=int, default=768, help="Backbone hidden size.")
    p.add_argument("--mamba-layers", type=int, default=6, help="Number of compact Mamba layers.")
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    return p.parse_args()


def normalize_backbone(value: str) -> str:
    val = (value or "gpt").lower()
    return "gpt" if val == "gpt2" else val


def write_csv(filename, rows):
    with open(filename, "w") as f:
        for row in rows:
            f.write(",".join(map(str, row)))
            f.write("\n")


def _load_any_checkpoint(path):
    """Load object from disk with compatibility for torch>=2.6."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _maybe_strip_module_prefix(state):
    if isinstance(state, (dict, OrderedDict)) and any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _build_gpt_model(args):
    from models.GPT4CP import Model
    return Model(
        gpu_id=0,
        use_gpu=0,
        pred_len=args.pred_len,
        prev_len=args.prev_len,
        UQh=args.UQh,
        UQv=args.UQv,
        BQh=args.BQh,
        BQv=args.BQv,
        K=args.K,
    )


def _build_mamba_model(args):
    from models.MAMBA import Model
    return Model(
        use_hf=args.use_hf_mamba,
        hf_name=args.hf_name,
        d_model=args.d_model,
        mamba_layers=args.mamba_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        pred_len=args.pred_len,
        prev_len=args.prev_len,
        use_gpu=0,
        gpu_id=0,
        K=args.K,
        UQh=args.UQh,
        UQv=args.UQv,
        BQh=args.BQh,
        BQv=args.BQv,
    )


def _build_model(backbone, args):
    if backbone == "mamba":
        return _build_mamba_model(args)
    return _build_gpt_model(args)


def _cache_path(weight_path, backbone, cache_dir):
    base = os.path.basename(weight_path)
    stem = base[:-4] if base.endswith(".pth") else base
    return os.path.join(cache_dir, f"{stem}_{backbone}_pickled.pth")


def load_or_convert_model(path, device, cache_dir, backbone, args):
    """
    Load a model checkpoint. If needed, instantiate the selected architecture,
    load state_dict and cache a pickled module for faster future runs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _cache_path(path, backbone, cache_dir)
    if os.path.exists(cache_file):
        return load_checkpoint(cache_file, device)

    obj = _load_any_checkpoint(path)

    # Already pickled module
    if isinstance(obj, nn.Module):
        obj.eval()
        torch.save(obj, cache_file)
        return obj.to(device)

    # Checkpoint formats
    if isinstance(obj, dict) and "model_state" in obj:
        state = obj["model_state"]
    elif isinstance(obj, (dict, OrderedDict)) and all(torch.is_tensor(v) for v in obj.values()):
        state = obj
    else:
        raise TypeError(
            f"Unrecognized checkpoint format at {path}. Expected nn.Module or state-dict content."
        )

    state = _maybe_strip_module_prefix(state)
    model = _build_model(backbone, args)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing (first 10):", missing[:10])
        if unexpected:
            print("  unexpected (first 10):", unexpected[:10])

    model.eval()
    torch.save(model, cache_file)
    print(f"[convert] Cached pickled model -> {cache_file}")
    return model.to(device)


def load_checkpoint(path, device):
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    return model.to(device).eval()


def build_weight_path(pattern, is_u2d, backbone, args):
    mode = "FDD" if is_u2d else "TDD"
    filename = args.weight_name_u2d if is_u2d else args.weight_name_u2u
    filename = filename.format(backbone=backbone)
    weight_dir = os.path.join(args.weight_root, pattern, mode)
    if args.weight_subdir:
        weight_dir = os.path.join(weight_dir, args.weight_subdir)
    return os.path.join(weight_dir, filename)


def build_dataset_paths(pattern, is_u2d, args):
    data_dir = os.path.join(args.data_root, pattern, "FDD" if is_u2d else "TDD")
    prev_path = os.path.join(data_dir, "H_U_his_test.mat")
    pred_path = os.path.join(data_dir, "H_D_pre_test.mat" if is_u2d else "H_U_pre_test.mat")
    return prev_path, pred_path, data_dir


def evaluate_config(cfg, device, args):
    name = cfg["name"]
    pattern = cfg["pattern"]
    is_u2d = int(cfg["is_U2D"])
    mode_str = "fdd" if is_u2d else "tdd"
    backbone = normalize_backbone(args.backbone)
    weight_path = build_weight_path(pattern, is_u2d, backbone, args)
    out_dir = os.path.join(args.results_root, name)
    os.makedirs(out_dir, exist_ok=True)

    prev_path, pred_path, data_dir = build_dataset_paths(pattern, is_u2d, args)

    missing = [p for p in [weight_path, prev_path, pred_path] if not os.path.exists(p)]
    if missing:
        print(f"[SKIP] {name} ({mode_str}): missing -> {missing}")
        return

    print(f"\n=== Evaluating {name} ({mode_str.upper()}) [{backbone.upper()}] ===")
    print(f"  weights: {weight_path}")
    print(f"  data: {data_dir}")

    criterion_nmse = NMSELoss()
    criterion_se = SE_Loss(snr=10, device=device)

    model = load_or_convert_model(
        weight_path,
        device=device,
        cache_dir=os.path.join(out_dir, "pickled_cache"),
        backbone=backbone,
        args=args,
    )

    test_data_prev_base = hdf5storage.loadmat(prev_path)["H_U_his_test"]
    test_data_pred_base = hdf5storage.loadmat(pred_path)["H_D_pre_test" if is_u2d else "H_U_pre_test"]

    NMSE = [[]]
    SE_pred = [[]]
    SE0_true = [[]]
    SE_ratio = [[]]

    Nt, Nr = 16, 1

    for speed in range(10):
        loss_nmse_stack, loss_se_stack, loss_se0_stack = [], [], []

        test_data_prev = test_data_prev_base[[speed], ...]
        test_data_pred = test_data_pred_base[[speed], ...]

        test_data_prev = rearrange(test_data_prev, "v b l k n m c -> (v b c) (n m) l (k)")
        test_data_pred = rearrange(test_data_pred, "v b l k n m c -> (v b c) (n m) l (k)")

        test_data_prev = noise(test_data_prev, args.snr_db)
        test_data_pred = noise(test_data_pred, args.snr_db)
        std = np.sqrt(np.std(np.abs(test_data_prev) ** 2))
        test_data_prev = test_data_prev / std
        test_data_pred = test_data_pred / std

        lens, _, _, _ = test_data_prev.shape
        prev_data = LoadBatch_ofdm_2(test_data_prev)
        pred_data = LoadBatch_ofdm_2(test_data_pred)
        cycle_times = lens // args.bs

        with torch.no_grad():
            pbar = tqdm.tqdm(range(cycle_times), desc=f"{name} | speed={speed}", leave=False)
            for cyt in pbar:
                prev = prev_data[cyt * args.bs : (cyt + 1) * args.bs, :, :, :].to(device)
                pred = pred_data[cyt * args.bs : (cyt + 1) * args.bs, :, :, :].to(device)

                prev = rearrange(prev, "b m l k -> (b m) l k")
                pred = rearrange(pred, "b m l k -> (b m) l k")

                out = model(prev, None, None, None)
                loss_nmse = criterion_nmse(out, pred)

                out_b = rearrange(out, "(b m) l k -> b l (k m)", b=args.bs)
                pred_b = rearrange(pred, "(b m) l k -> b l (k m)", b=args.bs)

                se, se0 = criterion_se(
                    h=Transform_TDD_FDD(out_b, Nt=Nt, Nr=Nr),
                    h0=Transform_TDD_FDD(pred_b, Nt=Nt, Nr=Nr),
                )

                loss_nmse_stack.append(loss_nmse.item())
                loss_se_stack.append(se.item())
                loss_se0_stack.append(se0.item())

        nmse_mean = np.nanmean(np.array(loss_nmse_stack))
        se_mean_pos = -np.nanmean(np.array(loss_se_stack))
        se0_mean_pos = -np.nanmean(np.array(loss_se0_stack))
        se_ratio = np.nanmean(np.array(loss_se_stack)) / np.nanmean(np.array(loss_se0_stack))

        print(
            f"speed {speed}: NMSE={nmse_mean:.6f} | SE={se_mean_pos:.6f} | "
            f"SE0={se0_mean_pos:.6f} | SE/SE0={se_ratio:.6f}"
        )

        NMSE[0].append(nmse_mean)
        SE_pred[0].append(se_mean_pos)
        SE0_true[0].append(se0_mean_pos)
        SE_ratio[0].append(se_ratio)

    prefix = os.path.join(out_dir, f"{name}_{mode_str}_{backbone}")

    write_csv(f"{prefix}_nmse.csv", NMSE)
    write_csv(f"{prefix}_se.csv", SE_pred)
    write_csv(f"{prefix}_se0.csv", SE0_true)
    write_csv(f"{prefix}_se_ratio.csv", SE_ratio)
    print(f"[Done] Saved CSVs under {out_dir} with prefix: {os.path.basename(prefix)}_*")


def build_config(patterns):
    configs = []
    for pattern in patterns:
        short_name = pattern.split("_", 1)[0]
        configs.append({"name": f"{short_name}_TDD", "pattern": pattern, "is_U2D": 0})
        configs.append({"name": f"{short_name}_FDD", "pattern": pattern, "is_U2D": 1})
    return configs


if __name__ == "__main__":
    args = parse_args()
    args.backbone = normalize_backbone(args.backbone)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for cfg in build_config(args.patterns):
        evaluate_config(cfg, device, args)

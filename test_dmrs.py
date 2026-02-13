"""
DMRS model evaluation (GPT-only).

For each configured DMRS pattern (TDD/FDD), evaluate a single GPT checkpoint
on its matching test set across 10 user speeds and export:
  - NMSE, SE, SE0, and SE/SE0 per speed (one row per model/config).

Only GPT baselines are considered; no transformer/cnn/gru/lstm/rnn/pad/pvec.
"""

import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import hdf5storage
import tqdm
from einops import rearrange

from data import LoadBatch_ofdm_2, noise, Transform_TDD_FDD
from metrics import NMSELoss, SE_Loss

# ----------------------- Config -----------------------
# Add or edit entries here to point to your DMRS GPT checkpoints and datasets.
MODEL_AND_DATA_CONFIG = [
    {
        "name": "pattern1_TDD",
        "is_U2D": 0,
        "weight_path": "./dmrs_model_weights/pattern1_type1_sparse/TDD/U2U_LLM4CP.pth",
        "data_dir": "./dmrs_datasets/pattern1_type1_sparse/TDD",
    },
    {
        "name": "pattern1_FDD",
        "is_U2D": 1,
        "weight_path": "./dmrs_model_weights/pattern1_type1_sparse/FDD/U2D_LLM4CP.pth",
        "data_dir": "./dmrs_datasets/pattern1_type1_sparse/FDD",
    },
    {
        "name": "pattern2_TDD",
        "is_U2D": 0,
        "weight_path": "./dmrs_model_weights/pattern2_type2_densefreq/TDD/U2U_LLM4CP.pth",
        "data_dir": "./dmrs_datasets/pattern2_type2_densefreq/TDD",
    },
    {
        "name": "pattern2_FDD",
        "is_U2D": 1,
        "weight_path": "./dmrs_model_weights/pattern2_type2_densefreq/FDD/U2D_LLM4CP.pth",
        "data_dir": "./dmrs_datasets/pattern2_type2_densefreq/FDD",
    },
    {
        "name": "pattern3_TDD",
        "is_U2D": 0,
        "weight_path": "./dmrs_model_weights/pattern3_highmob_dense/TDD/U2U_LLM4CP.pth",
        "data_dir": "./dmrs_datasets/pattern3_highmob_dense/TDD",
    },
    {
        "name": "pattern3_FDD",
        "is_U2D": 1,
        "weight_path": "./dmrs_model_weights/pattern3_highmob_dense/FDD/U2D_LLM4CP.pth",
        "data_dir": "./dmrs_datasets/pattern3_highmob_dense/FDD",
    },
]

RESULTS_ROOT = "dmrs_results"

# ----------------------- Helpers -----------------------
def write_csv(filename, rows):
    with open(filename, "w") as f:
        for row in rows:
            row = list(map(str, row))
            f.write(",".join(row))
            f.write("\n")


def load_checkpoint(path, device):
    """Load a pickled GPT model (torch.save(model, ...))."""
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    return model.to(device).eval()


# --- Auto-convert checkpoints to pickled GPT models (without touching originals) ---
def _maybe_strip_module_prefix(state):
    if isinstance(state, (dict, OrderedDict)) and any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

def _build_gpt_model(prev_len, pred_len, UQh, UQv, BQh, BQv, K):
    from models.GPT4CP import Model
    return Model(
        gpu_id=0, use_gpu=0,
        pred_len=pred_len, prev_len=prev_len,
        UQh=UQh, UQv=UQv, BQh=BQh, BQv=BQv,
        K=K,
    )

def load_or_convert_gpt(path, device, cache_dir, hparams):
    """
    Load a GPT model. If the path is a checkpoint (state_dict / dict with model_state),
    convert to a pickled nn.Module cached under cache_dir, leaving the original untouched.
    """
    os.makedirs(cache_dir, exist_ok=True)
    pickled_path = os.path.join(
        cache_dir, os.path.basename(path).replace(".pth", "_pickled.pth")
    )

    # Reuse cached pickled model if available
    if os.path.exists(pickled_path):
        return load_checkpoint(pickled_path, device)

    obj = torch.load(path, map_location="cpu")

    # Case A: already a pickled model
    if isinstance(obj, nn.Module):
        model = obj
    else:
        # Determine state dict
        if isinstance(obj, dict) and "model_state" in obj:
            state = obj["model_state"]
        elif isinstance(obj, (dict, OrderedDict)) and all(torch.is_tensor(v) for v in obj.values()):
            state = obj
        else:
            raise TypeError(
                f"Unrecognized checkpoint format at {path}. Expected nn.Module or state_dict-like content."
            )
        state = _maybe_strip_module_prefix(state)

        model = _build_gpt_model(
            prev_len=hparams["prev_len"],
            pred_len=hparams["pred_len"],
            UQh=hparams["UQh"],
            UQv=hparams["UQv"],
            BQh=hparams["BQh"],
            BQv=hparams["BQv"],
            K=hparams["K"],
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("  missing (first 10):", missing[:10])
            if unexpected:
                print("  unexpected (first 10):", unexpected[:10])

    model.eval()
    torch.save(model, pickled_path)
    print(f"[convert] Cached pickled model -> {pickled_path}")
    return model.to(device)


def build_paths(data_dir, is_u2d):
    prev_path = os.path.join(data_dir, "H_U_his_test.mat")
    tgt_name = "H_D_pre_test.mat" if is_u2d else "H_U_pre_test.mat"
    pred_path = os.path.join(data_dir, tgt_name)
    return prev_path, pred_path


# ----------------------- Core evaluation -----------------------
def evaluate_config(cfg, device, bs=64, snr_db=18):
    name = cfg["name"]
    is_u2d = int(cfg["is_U2D"])
    mode_str = "fdd" if is_u2d else "tdd"
    weight_path = cfg["weight_path"]
    data_dir = cfg["data_dir"]
    out_dir = os.path.join(RESULTS_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    prev_path, pred_path = build_paths(data_dir, is_u2d)

    # Basic existence checks
    missing = [p for p in [weight_path, prev_path, pred_path] if not os.path.exists(p)]
    if missing:
        print(f"[SKIP] {name}: missing paths -> {missing}")
        return

    print(f"\n=== Evaluating {name} ({mode_str.upper()}) ===")
    print(f"  weights: {weight_path}")
    print(f"  data   : {data_dir}")

    # Model hyperparams (should match training; adjust here if different)
    hparams = {
        "prev_len": 16,
        "pred_len": 4,
        "UQh": 1,
        "UQv": 1,
        "BQh": 1,
        "BQv": 1,
        "K": 48,
    }

    criterion_nmse = NMSELoss()
    criterion_se = SE_Loss(snr=10, device=device)
    model = load_or_convert_gpt(
        weight_path, device,
        cache_dir=os.path.join(out_dir, "pickled_cache"),
        hparams=hparams,
    )

    test_data_prev_base = hdf5storage.loadmat(prev_path)["H_U_his_test"]
    test_data_pred_base = hdf5storage.loadmat(pred_path)["H_D_pre_test" if is_u2d else "H_U_pre_test"]

    NMSE = [[]]  # one row (GPT) with 10 speeds
    SE_pred = [[]]
    SE0_true = [[]]
    SE_ratio = [[]]

    K, Nt, Nr = hparams["K"], 16, 1
    prev_len = hparams["prev_len"]
    pred_len = hparams["pred_len"]

    for speed in range(10):
        loss_nmse_stack, loss_se_stack, loss_se0_stack = [], [], []

        test_data_prev = test_data_prev_base[[speed], ...]
        test_data_pred = test_data_pred_base[[speed], ...]

        # (v b l k n m c) -> (batch, antennas, time, subcarriers)
        test_data_prev = rearrange(test_data_prev, "v b l k n m c -> (v b c) (n m) l (k)")
        test_data_pred = rearrange(test_data_pred, "v b l k n m c -> (v b c) (n m) l (k)")

        # Add noise and normalize
        test_data_prev = noise(test_data_prev, snr_db)
        test_data_pred = noise(test_data_pred, snr_db)
        std = np.sqrt(np.std(np.abs(test_data_prev) ** 2))
        test_data_prev = test_data_prev / std
        test_data_pred = test_data_pred / std

        lens, _, _, _ = test_data_prev.shape
        prev_data = LoadBatch_ofdm_2(test_data_prev)
        pred_data = LoadBatch_ofdm_2(test_data_pred)
        cycle_times = lens // bs

        with torch.no_grad():
            pbar = tqdm.tqdm(range(cycle_times), desc=f"{name} | speed={speed}", leave=False)
            for cyt in pbar:
                prev = prev_data[cyt * bs : (cyt + 1) * bs, :, :, :].to(device)
                pred = pred_data[cyt * bs : (cyt + 1) * bs, :, :, :].to(device)

                prev = rearrange(prev, "b m l k -> (b m) l k")
                pred = rearrange(pred, "b m l k -> (b m) l k")

                out = model(prev, None, None, None)

                loss_nmse = criterion_nmse(out, pred)

                out_b = rearrange(out, "(b m) l k -> b l (k m)", b=bs)
                pred_b = rearrange(pred, "(b m) l k -> b l (k m)", b=bs)

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

    # ---- CSV export (one row per model; here only GPT) ----
    prefix = os.path.join(out_dir, f"{name}_{mode_str}")

    write_csv(f"{prefix}_nmse.csv", NMSE)
    write_csv(f"{prefix}_se.csv", SE_pred)
    write_csv(f"{prefix}_se0.csv", SE0_true)
    write_csv(f"{prefix}_se_ratio.csv", SE_ratio)

    print(f"[Done] Saved CSVs under {out_dir} with prefix: {os.path.basename(prefix)}_*")


# ----------------------- Entry point -----------------------
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for cfg in MODEL_AND_DATA_CONFIG:
        evaluate_config(cfg, device)

#!/usr/bin/env python3
# train_mamba_or_gpt.py
"""
Train LLM4CP with a selectable backbone:
  - GPT-2  (original, from models/GPT4CP.py)
  - Mamba  (new, from models/MAMBA.py; either HF-pretrained or compact custom)

Examples
--------
# Train TDD with Mamba (compact custom backbone):
python train_mamba_or_gpt.py \
  --backbone mamba \
  --save-path model_weights/U2U_MAMBA.pth

# Train TDD with Mamba (HF-pretrained backbone):
python train_mamba_or_gpt.py \
  --backbone mamba --use-hf-mamba \
  --hf-name state-spaces/mamba-370m-hf \
  --save-path model_weights/U2U_MAMBA_HF.pth

python train_mamba.py \
  --backbone mamba --use-hf-mamba \
  --hf-name state-spaces/mamba-130m-hf \
  --save-path model_weights/U2U_MAMBA_HF.pth

# Train TDD with original GPT-2:
python train_mamba_or_gpt.py \
  --backbone gpt2 \
  --save-path model_weights/U2U_LLM4CP_gpt2.pth
"""

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import Dataset_Pro
from metrics import NMSELoss

# ----------------- Args -----------------
def parse_args():
    p = argparse.ArgumentParser()
    # Data / task
    p.add_argument("--train-his", type=str, default="./data/dataset/train/H_U_his_train.mat")
    p.add_argument("--train-tgt", type=str, default="./data/dataset/train/H_U_pre_train.mat")
    p.add_argument("--u2d", type=int, default=0, help="1: U->D (FDD), 0: U->U (TDD)")
    p.add_argument("--few", type=int, default=0, help="1: few-shot split")

    # Backbone choice
    p.add_argument("--backbone", type=str, default="mamba", choices=["mamba", "gpt2"],
                   help="Backbone type: 'mamba' or 'gpt2'")

    # Mamba (route A/B)
    p.add_argument("--use-hf-mamba", action="store_true",
                   help="If set, load a HF pretrained Mamba and feed inputs_embeds; else use compact custom Mamba.")
    p.add_argument("--hf-name", type=str, default="state-spaces/mamba-370m-hf",
                   help="HF model id for Mamba backbone (used only if --use-hf-mamba).")
    p.add_argument("--d-model", type=int, default=768, help="Backbone hidden size for CSI embeddings")
    p.add_argument("--mamba-layers", type=int, default=6, help="Number of Mamba layers (compact custom)")
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)

    # Model/task dims
    p.add_argument("--pred-len", type=int, default=4)
    p.add_argument("--prev-len", type=int, default=16)
    p.add_argument("--K", type=int, default=48)
    p.add_argument("--UQh", type=int, default=1)
    p.add_argument("--UQv", type=int, default=1)
    p.add_argument("--BQh", type=int, default=1)
    p.add_argument("--BQv", type=int, default=1)

    # Training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--save-path", type=str, default="model_weights/U2U_LLM4CP.pth")
    return p.parse_args()


# ----------------- Train / Val loops -----------------
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    losses = []
    for batch in loader:
        target, prev = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(prev, None, None, None)
        loss = criterion(pred, target)
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()
    return float(np.nanmean(np.array(losses))) if losses else np.inf


@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    losses = []
    for batch in loader:
        target, prev = batch[0].to(device), batch[1].to(device)
        pred = model(prev, None, None, None)
        loss = criterion(pred, target)
        losses.append(float(loss.item()))
    return float(np.nanmean(np.array(losses))) if losses else np.inf


# ----------------- Main -----------------
if __name__ == "__main__":
    args = parse_args()

    # Device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Datasets
    train_set = Dataset_Pro(args.train_his, args.train_tgt, is_train=1, is_U2D=args.u2d, is_few=args.few)
    val_set   = Dataset_Pro(args.train_his, args.train_tgt, is_train=0, is_U2D=args.u2d)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True, drop_last=False)

    # Model selection
    if args.backbone == "gpt2":
        # Original GPT-2 model
        from models.GPT4CP import Model as GPTModel
        model = GPTModel(
            gpu_id=args.gpu_id,
            pred_len=args.pred_len, prev_len=args.prev_len,
            UQh=args.UQh, UQv=args.UQv, BQh=args.BQh, BQv=args.BQv,
        ).to(device)
        print("[Backbone] GPT-2 (original)")

    else:
        # Mamba model (our drop-in replacement)
        from models.MAMBA import Model as MambaModel
        model = MambaModel(
            use_hf=args.use_hf_mamba,
            hf_name=args.hf_name,
            d_model=args.d_model,
            mamba_layers=args.mamba_layers,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            pred_len=args.pred_len, prev_len=args.prev_len,
            use_gpu=1, gpu_id=args.gpu_id,
            K=args.K, UQh=args.UQh, UQv=args.UQv, BQh=args.BQh, BQv=args.BQv
        ).to(device)
        print(f"[Backbone] Mamba | use_hf={args.use_hf_mamba} | hf_name={args.hf_name if args.use_hf_mamba else 'custom-compact'}")

    # Info
    total = sum(p.numel() for p in model.parameters())
    learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameter: {total/1e6:.5f}M")
    print(f"Number of learnable parameter: {learn/1e6:.5f}M")

    # (Optional) resume if a pickled model exists at save_path
    if os.path.exists(args.save_path):
        try:
            model = torch.load(args.save_path, map_location=device)
            print(f"[Resume] Loaded pickled model from {args.save_path}")
        except Exception as e:
            print(f"[Resume WARN] Could not load pickled model from {args.save_path}: {e}")

    # Optimizer / Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    criterion = NMSELoss().to(device)

    # Train
    best_val = float("inf")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    print("Start training...")

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val = validate(model, val_loader, device, criterion)

        print(f"Epoch: {epoch+1}/{args.epochs} | train NMSE: {tr:.7f} | val NMSE: {val:.7f}")

        if val < best_val:
            best_val = val
            torch.save(model, args.save_path)  # keep same pickled format as original script
            print(f"[Checkpoint] best val {best_val:.7f} -> saved to {args.save_path}")

    # Final counts
    total = sum(p.numel() for p in model.parameters())
    learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameter: {total/1e6:.5f}M")
    print(f"Number of learnable parameter: {learn/1e6:.5f}M")

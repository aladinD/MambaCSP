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

from data import Dataset_Pro# train.py
"""
LLM4CP Training Script (Patched for DDP + AMP + Memory Safety)

Key fixes vs OOM-on-GPU0:
- Set device BEFORE any GPU allocations.
- Create & load model on CPU first, THEN move to the current GPU, THEN DDP-wrap.
- Ensure dataset stays on CPU; move to GPU only inside the training loop.
- Use per-GPU batch size (default smaller) + optional gradient accumulation for large global batch.
- Clean state_dict checkpoints. AMP enabled. Proper DistributedSampler usage.
"""

import os
from datetime import timedelta
import argparse
import math
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

# ---- Environment knobs to reduce fragmentation / contention (set early) ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

# ---- Project imports (must NOT allocate on GPU) ----
from models.GPT4CP import Model
from data import Dataset_Pro
from metrics import NMSELoss


# --------------------- DDP helpers ---------------------
def ddp_available() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if ddp_available() else 0

def get_world_size() -> int:
    return dist.get_world_size() if ddp_available() else 1

def is_main() -> bool:
    return get_rank() == 0

def setup_ddp_if_launched():
    """
    Initialize DDP if launched with torchrun. Return (device, local_rank, ddp_enabled).
    """
    ddp_enabled = "LOCAL_RANK" in os.environ
    if ddp_enabled:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        local_rank = int(os.environ["LOCAL_RANK"])
        # IMPORTANT: pin this process to its GPU BEFORE any CUDA tensors/models are created.
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, local_rank, ddp_enabled

def barrier():
    if ddp_available():
        dist.barrier()


# --------------------- Argparse ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Data / task
    p.add_argument("--train-his", type=str, default="./data/dataset/train/H_U_his_train.mat",
                   help="Path to historical CSI .mat file")
    p.add_argument("--train-tgt", type=str, default="./data/dataset/train/H_U_pre_train.mat",
                   help="Path to target (future) CSI .mat file")
    p.add_argument("--u2d", type=int, default=0, help="1 for Uplink->Downlink (FDD), 0 for Uplink->Uplink (TDD)")
    p.add_argument("--few", type=int, default=0, help="1 for few-shot dataset, else 0")

    # Model hyperparams (match paper defaults)
    p.add_argument("--pred-len", type=int, default=4)
    p.add_argument("--prev-len", type=int, default=16)
    p.add_argument("--UQh", type=int, default=1)
    p.add_argument("--UQv", type=int, default=1)
    p.add_argument("--BQh", type=int, default=1)
    p.add_argument("--BQv", type=int, default=1)

    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--per-gpu-batch-size", type=int, default=64,
                   help="Per-process batch size (NOT global).")
    p.add_argument("--global-batch-size", type=int, default=0,
                   help="Optional: set a desired global batch and we will compute per-GPU + accumulation.")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = per_gpu * world_size * accum_steps).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-path", type=str, default="model_weights/U2U_LLM4CP.pth")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--compile", action="store_true", help="Use torch.compile for extra speed (optional)")
    return p.parse_args()


# --------------------- Checkpoint helpers ---------------------
def save_checkpoint(path, model, optimizer, epoch, best_loss, ddp_enabled):
    if not is_main():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state": (model.module if ddp_enabled else model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }
    torch.save(state, path)

def load_checkpoint_cpu_if_any(path, model):
    """
    Load model weights to CPU model (safe for memory).
    Return (start_epoch, best_loss, has_ckpt).
    """
    start_epoch, best_loss, has_ckpt = 0, float("inf"), False
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", best_loss)
        has_ckpt = True
    return start_epoch, best_loss, has_ckpt


# --------------------- Training / Validation ---------------------
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, learn

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch, ddp_enabled, autocast_dtype, accum_steps):
    model.train()
    if ddp_enabled and isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    running_loss = 0.0
    running_count = 0
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        # Dataset_Pro should return CPU tensors; we move to GPU here only:
        target = batch[0].to(device, non_blocking=True)  # future CSI
        prev   = batch[1].to(device, non_blocking=True)  # history CSI

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            pred = model(prev, None, None, None)
            loss = criterion(pred, target)

        # gradient accumulation
        loss_to_backprop = loss / accum_steps
        scaler.scale(loss_to_backprop).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # weighted tracking by batch size
        bsz = target.shape[0]
        running_loss += loss.detach().float().item() * bsz
        running_count += bsz

    # flush pending grads if last step not hit
    # (optional; usually not needed if dataset size is divisible)
    # if (i + 1) % accum_steps != 0:
    #     scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

    # reduce across ranks
    world = get_world_size()
    device_cpu = torch.device("cpu")
    loss_tensor = torch.tensor(running_loss, dtype=torch.float32, device=device)
    cnt_tensor = torch.tensor(running_count, dtype=torch.float32, device=device)
    if world > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)
    epoch_loss = (loss_tensor / cnt_tensor).to(device_cpu).item()
    return epoch_loss

@torch.no_grad()
def validate(model, loader, criterion, device, autocast_dtype):
    model.eval()
    running_loss = 0.0
    running_count = 0

    for batch in loader:
        target = batch[0].to(device, non_blocking=True)
        prev   = batch[1].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            pred = model(prev, None, None, None)
            loss = criterion(pred, target)
        bsz = target.shape[0]
        running_loss += loss.detach().float().item() * bsz
        running_count += bsz

    world = get_world_size()
    device_cpu = torch.device("cpu")
    loss_tensor = torch.tensor(running_loss, dtype=torch.float32, device=device)
    cnt_tensor = torch.tensor(running_count, dtype=torch.float32, device=device)
    if world > 1:
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)
    val_loss = (loss_tensor / cnt_tensor).to(device_cpu).item()
    return val_loss


# --------------------- Main ---------------------
def main():
    args = parse_args()

    # Sanity message so we don't mix up files
    if args.u2d:
        if "D_pre" not in os.path.basename(args.train_tgt):
            if is_main():
                print(f"[WARN] --u2d=1 but train-tgt looks uplink-ish: {args.train_tgt}")
                print("       Make sure this points to H_D_pre_train.mat (downlink).")
    else:
        if "U_pre" not in os.path.basename(args.train_tgt):
            if is_main():
                print(f"[WARN] --u2d=0 but train-tgt doesn't look like H_U_pre_*.mat: {args.train_tgt}")


    # Perf knobs (safe when shapes are static-ish)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Repro
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # DDP setup FIRST, then pin device
    device, local_rank, ddp_enabled = setup_ddp_if_launched()
    if is_main():
        print(f"DDP: {ddp_enabled} | device: {device} | local_rank: {local_rank}")

    # Compute per-GPU batch size & accumulation if global batch specified
    per_gpu_bs = args.per_gpu_batch_size
    accum_steps = args.accum_steps
    world = get_world_size()
    if args.global_batch_size > 0:
        # try to honor desired global batch ~= per_gpu_bs * world * accum_steps
        per_gpu_bs = max(1, args.global_batch_size // max(1, world * accum_steps))
        if is_main():
            print(f"[BatchSizing] Requested global {args.global_batch_size} -> per_gpu {per_gpu_bs} with accum {accum_steps}")
    eff_global = per_gpu_bs * max(1, world) * max(1, accum_steps)
    if is_main():
        print(f"Effective global batch size: {eff_global} (per_gpu={per_gpu_bs}, world={world}, accum={accum_steps})")

    # Datasets (keep on CPU)
    train_set = Dataset_Pro(args.train_his, args.train_tgt, is_train=1, is_U2D=args.u2d, is_few=args.few)
    val_set   = Dataset_Pro(args.train_his, args.train_tgt, is_train=0, is_U2D=args.u2d)

    # Samplers / Loaders
    if ddp_enabled:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(val_set, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=per_gpu_bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=per_gpu_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    # ----- Build model on CPU first (avoid touching GPU:0), load ckpt on CPU, THEN move to GPU -----
    model_cpu = Model(
        gpu_id=0,  # this arg is internal to your Model; passing 0 is fine since we aren't allocating yet
        pred_len=args.pred_len, prev_len=args.prev_len,
        UQh=args.UQh, UQv=args.UQv, BQh=args.BQh, BQv=args.BQv
    )  # still on CPU

    start_epoch, best_loss, has_ckpt = load_checkpoint_cpu_if_any(args.save_path, model_cpu)
    if is_main():
        if has_ckpt:
            print(f"[Resume] Loaded CPU checkpoint '{args.save_path}' at epoch {start_epoch}, best_loss={best_loss:.7f}")
        else:
            print(f"[Resume] No checkpoint at '{args.save_path}'. Starting fresh.")

    # Optional compile BEFORE moving to GPU/DDP (PyTorch 2.1+)
    if args.compile and hasattr(torch, "compile"):
        model_cpu = torch.compile(model_cpu, mode="max-autotune")

    # Now move model to this rank's GPU
    model = model_cpu.to(device, non_blocking=True)

    # Optimizer (create AFTER model is on the right device)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

    # AMP dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
        use_scaler = False
    else:
        autocast_dtype = torch.float16
        use_scaler = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # DDP wrap (AFTER moving to device, AFTER optimizer created)
    if ddp_enabled:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,   # IMPORTANT
        )


    # Loss
    criterion = NMSELoss().to(device)

    barrier()

    # Report params
    if is_main():
        total, learn = count_params(model.module if ddp_enabled else model)
        print(f"Number of parameters: {total/1e6:.5f}M | Learnable: {learn/1e6:.5f}M")

    # ---------- Train loop ----------
    epochs = args.epochs
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion,
            device, epoch, ddp_enabled, autocast_dtype, accum_steps
        )
        val_loss = validate(model, val_loader, criterion, device, autocast_dtype)

        if is_main():
            print(f"Epoch {epoch+1}/{epochs} | train NMSE: {train_loss:.7f} | val NMSE: {val_loss:.7f}")

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(args.save_path, model, optimizer, epoch, best_loss, ddp_enabled)
            if is_main():
                print(f"[Checkpoint] New best val NMSE: {best_loss:.7f} -> saved to {args.save_path}")

        scheduler.step()

    if is_main():
        total, learn = count_params(model.module if ddp_enabled else model)
        print(f"FINAL | Number of parameters: {total/1e6:.5f}M | Learnable: {learn/1e6:.5f}M")

    if ddp_available():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


# Train for TDD
# torchrun --nproc_per_node=8 train_acc.py --save-path model_weights/train_acc/full_shot_tdd/SOME_FOLDER/U2U_LLM4CP.pth
# torchrun --nproc_per_node=8 train_acc.py \
#   --u2d 0 \
#   --train-his ./data/dataset/train/H_U_his_train.mat \
#   --train-tgt ./data/dataset/train/H_U_pre_train.mat \
#   --save-path model_weights/train_acc/full_shot_tdd/U2U_LLM4CP.pth

# Train for FDD
# torchrun --nproc_per_node=8 train_acc.py \
#   --u2d 1 \
#   --train-his ./data/dataset/train/H_U_his_train.mat \
#   --train-tgt ./data/dataset/train/H_D_pre_train.mat \
#   --save-path model_weights/train_acc/full_shot_fdd/U2D_LLM4CP.pth

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

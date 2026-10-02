#!/usr/bin/env python3
"""
LLM4CP DMRS Trainer (DDP + AMP + Memory-Safe Init)

Single training entrypoint for DMRS experiments with a selectable backbone:
- `--backbone gpt`    -> original GPT-2 model (`models.GPT4CP.Model`)
- `--backbone mamba`  -> Mamba replacement (`models.MAMBA.Model`)

Behavior:
- Selects device before model build.
- Builds model on CPU first, loads checkpoint on CPU, then moves to rank GPU.
- Uses per-GPU batch size + optional gradient accumulation.
- Persists checkpoints as state_dict dict for DDP-safe resume.
"""

import os
from datetime import timedelta
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler


# ---- Environment knobs to reduce fragmentation / contention (set early) ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:512")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")


# ---- Project imports (must NOT allocate on GPU) ----
from data import Dataset_Pro
from metrics import NMSELoss


# --------------------- DDP helpers ---------------------
def ddp_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if ddp_available() else 0


def get_world_size() -> int:
    return dist.get_world_size() if ddp_available() else 1


def is_main() -> int:
    return get_rank() == 0


def setup_ddp_if_launched():
    """
    Initialize DDP if launched with torchrun.
    Returns (device, local_rank, ddp_enabled).
    """
    ddp_enabled = "LOCAL_RANK" in os.environ
    if ddp_enabled:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, local_rank, ddp_enabled


def barrier():
    if ddp_available():
        dist.barrier()


# --------------------- Helpers ---------------------
def normalize_backbone(value: str) -> str:
    """Accept legacy aliases while keeping simple public names."""
    val = (value or "gpt").lower()
    return "gpt" if val == "gpt2" else val


def parse_args():
    p = argparse.ArgumentParser(
        description="Train DMRS predictor with either GPT-2 or Mamba backbone."
    )
    # Data / task
    p.add_argument("--train-his", type=str, default="./data/dataset/train/H_U_his_train.mat",
                   help="Path to historical CSI .mat file")
    p.add_argument("--train-tgt", type=str, default="./data/dataset/train/H_U_pre_train.mat",
                   help="Path to target (future) CSI .mat file")
    p.add_argument("--u2d", type=int, default=0,
                   help="1 for Uplink->Downlink (FDD), 0 for Uplink->Uplink (TDD)")
    p.add_argument("--few", type=int, default=0, help="1 for few-shot dataset, else 0")

    # Backbone choice
    p.add_argument("--backbone", type=str, default="gpt", choices=["gpt", "gpt2", "mamba"],
                   help="Backbone type: gpt (GPT-2) or mamba")

    # Mamba backbone options
    p.add_argument("--use-hf-mamba", action="store_true",
                   help="Use pretrained HF Mamba backbone (requires transformers + model download)")
    p.add_argument("--hf-name", type=str, default="state-spaces/mamba-370m-hf",
                   help="HF model id for Mamba backbone (used only if --use-hf-mamba)")
    p.add_argument("--d-model", type=int, default=768,
                   help="Backbone hidden size for CSI embeddings")
    p.add_argument("--mamba-layers", type=int, default=6, help="Number of compact Mamba blocks")
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--K", type=int, default=48)

    # Model hyperparams
    p.add_argument("--pred-len", type=int, default=4)
    p.add_argument("--prev-len", type=int, default=16)
    p.add_argument("--UQh", type=int, default=1)
    p.add_argument("--UQv", type=int, default=1)
    p.add_argument("--BQh", type=int, default=1)
    p.add_argument("--BQv", type=int, default=1)

    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--per-gpu-batch-size", type=int, default=64,
                   help="Per-process batch size (NOT global)")
    p.add_argument("--global-batch-size", type=int, default=0,
                   help="Optional target global batch: per_gpu = global / (world * accum)")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = per_gpu * world * accum)")
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {
        "model_state": (model.module if ddp_enabled else model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }
    torch.save(state, path)


def load_checkpoint_cpu_if_any(path, model):
    """
    Load state-dict checkpoint onto CPU model.
    Returns (start_epoch, best_loss, has_ckpt).
    """
    def _strip_module_prefix(state):
        if any(k.startswith("module.") for k in state.keys()):
            return {k.replace("module.", "", 1): v for k, v in state.items()}
        return state

    start_epoch, best_loss, has_ckpt = 0, float("inf"), False
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = _strip_module_prefix(ckpt["model_state"])
            model.load_state_dict(state)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss = ckpt.get("best_loss", best_loss)
        elif isinstance(ckpt, dict):
            # Compatibility with raw state_dict saves.
            ckpt = _strip_module_prefix(ckpt)
            model.load_state_dict(ckpt)
            start_epoch = 0
        else:
            return start_epoch, best_loss, has_ckpt
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
        target = batch[0].to(device, non_blocking=True)  # future CSI
        prev = batch[1].to(device, non_blocking=True)    # history CSI

        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            pred = model(prev, None, None, None)
            loss = criterion(pred, target)

        loss_to_backprop = loss / accum_steps
        scaler.scale(loss_to_backprop).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

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
    return (loss_tensor / cnt_tensor).to(device_cpu).item()


@torch.no_grad()
def validate(model, loader, criterion, device, autocast_dtype):
    model.eval()
    running_loss = 0.0
    running_count = 0

    for batch in loader:
        target = batch[0].to(device, non_blocking=True)
        prev = batch[1].to(device, non_blocking=True)
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
    return (loss_tensor / cnt_tensor).to(device_cpu).item()


def build_backbone_model(args, local_rank):
    """
    Build either GPT or Mamba model on CPU.
    """
    backbone = normalize_backbone(args.backbone)
    if backbone == "mamba":
        from models.MAMBA import Model as BackboneModel
        model_cpu = BackboneModel(
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
            gpu_id=local_rank,
            K=args.K,
            UQh=args.UQh, UQv=args.UQv, BQh=args.BQh, BQv=args.BQv
        )
        return model_cpu, "mamba"

    from models.GPT4CP import Model as BackboneModel
    model_cpu = BackboneModel(
        pred_len=args.pred_len,
        prev_len=args.prev_len,
        UQh=args.UQh, UQv=args.UQv, BQh=args.BQh, BQv=args.BQv,
        use_gpu=0,
        gpu_id=0,
        K=args.K,
    )
    return model_cpu, "gpt"


# --------------------- Main ---------------------
def main():
    args = parse_args()
    args.backbone = normalize_backbone(args.backbone)

    # Sanity message so we do not mix up file types
    if args.u2d:
        if "D_pre" not in os.path.basename(args.train_tgt) and is_main():
            print(f"[WARN] --u2d=1 but train-tgt looks uplink-ish: {args.train_tgt}")
            print("       Make sure this points to H_D_pre_train.mat (downlink).")
    else:
        if "U_pre" not in os.path.basename(args.train_tgt) and is_main():
            print(f"[WARN] --u2d=0 but train-tgt doesn't look like H_U_pre_*.mat: {args.train_tgt}")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device, local_rank, ddp_enabled = setup_ddp_if_launched()
    if is_main():
        print(f"DDP: {ddp_enabled} | device: {device} | local_rank: {local_rank}")
        print(f"Backbone: {args.backbone} | args: {vars(args)}")

    # Batch sizing
    per_gpu_bs = args.per_gpu_batch_size
    accum_steps = args.accum_steps
    world = get_world_size()
    if args.global_batch_size > 0:
        per_gpu_bs = max(1, args.global_batch_size // max(1, world * accum_steps))
        if is_main():
            print(f"[BatchSizing] Requested global {args.global_batch_size} -> per_gpu {per_gpu_bs} with accum {accum_steps}")
    eff_global = per_gpu_bs * max(1, world) * max(1, accum_steps)
    if is_main():
        print(f"Effective global batch size: {eff_global} (per_gpu={per_gpu_bs}, world={world}, accum={accum_steps})")

    # Datasets (kept on CPU)
    train_set = Dataset_Pro(args.train_his, args.train_tgt, is_train=1, is_U2D=args.u2d, is_few=args.few)
    val_set = Dataset_Pro(args.train_his, args.train_tgt, is_train=0, is_U2D=args.u2d)

    if ddp_enabled:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

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

    # Build and restore checkpoint
    model_cpu, backbone = build_backbone_model(args, local_rank)
    if backbone == "mamba":
        print(f"[Backbone] Mamba | use_hf={args.use_hf_mamba} | hf_name={args.hf_name if args.use_hf_mamba else 'custom-compact'}")
    else:
        print("[Backbone] GPT-2 (original)")

    start_epoch, best_loss, has_ckpt = load_checkpoint_cpu_if_any(args.save_path, model_cpu)
    if is_main():
        if has_ckpt:
            print(f"[Resume] Loaded CPU checkpoint '{args.save_path}' at epoch {start_epoch}, best_loss={best_loss:.7f}")
        else:
            print(f"[Resume] No checkpoint at '{args.save_path}'. Starting fresh.")

    if args.compile and hasattr(torch, "compile"):
        model_cpu = torch.compile(model_cpu, mode="max-autotune")

    model = model_cpu.to(device, non_blocking=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
        use_scaler = False
    else:
        autocast_dtype = torch.float16
        use_scaler = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    if ddp_enabled:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    criterion = NMSELoss().to(device)
    barrier()

    if is_main():
        total, learn = count_params(model.module if ddp_enabled else model)
        print(f"Number of parameters: {total/1e6:.5f}M | Learnable: {learn/1e6:.5f}M")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            epoch, ddp_enabled, autocast_dtype, accum_steps
        )
        val_loss = validate(model, val_loader, criterion, device, autocast_dtype)

        if is_main():
            print(
                f"Epoch {epoch+1}/{args.epochs} | train NMSE: {train_loss:.7f} | "
                f"val NMSE: {val_loss:.7f} | lr: {optimizer.param_groups[0]['lr']:.6g}"
            )

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


# Example (TDD, GPT):
# torchrun --nproc_per_node=8 train_dmrs.py \
#   --backbone gpt \
#   --u2d 0 \
#   --train-his ./dmrs_datasets/pattern1_type1_sparse/TDD/H_U_his_train.mat \
#   --train-tgt ./dmrs_datasets/pattern1_type1_sparse/TDD/H_U_pre_train.mat \
#   --save-path dmrs_model_weights/pattern1_type1_sparse/TDD/U2U_LLM4CP.pth

# Example (FDD, Mamba):
# torchrun --nproc_per_node=8 train_dmrs.py \
#   --backbone mamba \
#   --use-hf-mamba \
#   --hf-name state-spaces/mamba-370m-hf \
#   --u2d 1 \
#   --train-his ./dmrs_datasets/pattern1_type1_sparse/FDD/H_U_his_train.mat \
#   --train-tgt ./dmrs_datasets/pattern1_type1_sparse/FDD/H_D_pre_train.mat \
#   --save-path dmrs_model_weights/pattern1_type1_sparse/FDD/U2D_LLM4CP.pth

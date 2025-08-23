"""
Evaluate several channel-prediction models on a held-out test set across 10 user speeds.
For each model & speed, compute:
  - NMSE between predicted and ground-truth CSI sequences,
  - Spectral Efficiency (SE) of predicted CSI,
  - Spectral Efficiency with perfect CSI (SE0),
  - SE ratio = SE / SE0.

Adds:
  - tqdm progress bars around inner loops (ETA per model/speed),
  - CSV export for NMSE, SE, SE0, and SE ratio.

Data layout (from .mat files)
-----------------------------
"v b l k n m c"
  v: velocity index (10 speeds)
  b: per-speed batch index
  l: time length (history + future)
  k: subcarriers (K=48)
  n, m: antenna grid dims (so n*m = Nt)
  c: polarization (or similar)

We reshape to:
  (batch, antennas, time, subcarriers) = (lens, Nt, L, K)
Then process each antenna stream in parallel.
"""

import time
import torch
import numpy as np
from data import LoadBatch_ofdm_1, LoadBatch_ofdm_2, noise, Transform_TDD_FDD
from metrics import NMSELoss, SE_Loss
from einops import rearrange
import hdf5storage
import tqdm
from pvec import pronyvec
from PAD import PAD3

if __name__ == "__main__":
    # ----------------------- Configuration -----------------------
    # Which GPU to use for testing
    device = torch.device('cuda:0')  # change to 'cuda:0' or another device as needed

    # Mode: 0 = TDD (uplink->uplink), 1 = FDD (uplink->downlink)
    is_U2D = 0
    mode_str = "fdd" if is_U2D else "tdd"

    # Test data (.mat) paths
    # Expected keys in files:
    #   H_U_his_test, H_U_pre_test, H_D_pre_test
    prev_path = "./data/dataset/test/H_U_his_test.mat"  # historical UL CSI
    pred_path = "./data/dataset/test/H_U_pre_test.mat"  # future UL CSI (TDD)
    pred_path_fdd = "./data/dataset/test/H_D_pre_test.mat"  # future DL CSI (FDD)

    # Pretrained model checkpoints for each NN baseline
    model_path = {
        'gpt': './data/model/weights/full_shot_tdd/U2U_LLM4CP.pth',
        # 'gpt': './model_weights/train_acc/full_shot_tdd/U2U_LLM4CP.pth',
        'transformer': './data/model/weights/full_shot_tdd/U2U_trans.pth',
        'cnn': './data/model/weights/full_shot_tdd/U2U_cnn.pth',
        'gru': './data/model/weights/full_shot_tdd/U2U_gru.pth',
        'lstm': './data/model/weights/full_shot_tdd/U2U_lstm.pth',
        'rnn': './data/model/weights/full_shot_tdd/U2U_rnn.pth'
    }

    # Which models to evaluate
    # 'np'   = naive baseline (repeat last observed frame)
    # 'pad'  = Prony-based angular-delay model
    # 'pvec' = Prony in frequency (supported but not enabled by default here)
    model_test_enable = ['gpt', 'transformer', 'cnn', 'gru', 'lstm', 'rnn', 'np', 'pad']

    # Window sizes and system dims (must match your dataset/model)
    prev_len = 16      # length of history window
    label_len = 12     # transformer decoder warm-start length
    pred_len = 4       # prediction horizon
    K, Nt, Nr, SR = (48, 16, 1, 1)  # Nt = 4*4 = 16 antennas, K=48 subcarriers

    # Mini-batch size for evaluation
    bs = 64

    print("Total model nums:", len(model_test_enable))

    # ----------------------- Metrics & data load -----------------------
    # Losses:
    # - NMSELoss: normalized MSE between predicted and ground-truth CSI (lower is better)
    # - SE_Loss(snr=10 dB): returns negative SE as a "loss" (so we negate it to report positive SE)
    criterion_nmse = NMSELoss()
    criterion_se = SE_Loss(snr=10, device=device)

    # Load test datasets (NumPy arrays via hdf5storage)
    test_data_prev_base = hdf5storage.loadmat(prev_path)['H_U_his_test']  # shape: (v b l k n m c)
    if is_U2D:
        test_data_pred_base = hdf5storage.loadmat(pred_path_fdd)['H_D_pre_test']
    else:
        test_data_pred_base = hdf5storage.loadmat(pred_path)['H_U_pre_test']

    # Storage for per-model metrics across 10 speeds
    NMSE = [[] for _ in model_test_enable]
    SE_ratio = [[] for _ in model_test_enable]   # SE / SE0
    SE_pred = [[] for _ in model_test_enable]    # positive SE (predicted)
    SE0_true = [[] for _ in model_test_enable]   # positive SE0 (oracle / perfect CSI)

    # ----------------------- Evaluation loop -----------------------
    for i, name in enumerate(model_test_enable):
        print("---------------------------------------------------------------")
        print(f"Loading {i+1}-th model ...... {name}")

        # Load neural-network models (skip for classical baselines 'pad'/'pvec' and naive 'np')
        if name not in ['pad', 'pvec', 'np']:
            # NOTE: these checkpoints are full pickled models (torch.save(model, ...)); class must be importable
            model = torch.load(model_path[name], map_location=device).to(device)
            model.eval()

        # Evaluate across 10 user speeds (0..9)
        for speed in range(0, 10):
            test_loss_stack_nmse = []  # list of scalar NMSEs for this model+speed
            test_loss_stack_se = []    # list of (negative) SE values for this model+speed
            test_loss_stack_se0 = []   # list of (negative) SE0 values for this model+speed

            # Slice out this speed and reshape
            # From (v b l k n m c) -> ( (v b c) , (n m) , l , k )
            test_data_prev = test_data_prev_base[[speed], ...]
            test_data_pred = test_data_pred_base[[speed], ...]
            test_data_prev = rearrange(test_data_prev, 'v b l k n m c -> (v b c) (n m) l (k)')
            test_data_pred = rearrange(test_data_pred, 'v b l k n m c -> (v b c) (n m) l (k)')

            # Add evaluation noise ~ 18 dB SNR to both input and target
            test_data_prev = noise(test_data_prev, 18)
            test_data_pred = noise(test_data_pred, 18)

            # Normalize by RMS-like scale of the input (scale-invariant evaluation)
            std = np.sqrt(np.std(np.abs(test_data_prev) ** 2))
            test_data_prev = test_data_prev / std
            test_data_pred = test_data_pred / std

            # lens = number of batches for this speed (v*b*c collapsed)
            lens, _, _, _ = test_data_prev.shape

            if name in ['gpt', 'transformer', 'rnn', 'lstm', 'gru', 'cnn', 'np']:
                # Convert numpy -> torch tensors for NN baselines
                prev_data = LoadBatch_ofdm_2(test_data_prev)  # shape: (lens, Nt, L, K) as torch
                pred_data = LoadBatch_ofdm_2(test_data_pred)

                # How many full mini-batches of size 'bs' we have (drop remainder)
                cycle_times = lens // bs

                # tqdm progress bar for inner loop (ETA per model/speed)
                with torch.no_grad():
                    pbar_desc = f"{name} | speed={speed}"
                    for cyt in tqdm.tqdm(range(cycle_times), desc=pbar_desc, leave=False):
                        # Take a mini-batch of size 'bs'
                        prev = prev_data[cyt * bs:(cyt + 1) * bs, :, :, :].to(device)
                        pred = pred_data[cyt * bs:(cyt + 1) * bs, :, :, :].to(device)

                        # Flatten antenna dimension into batch: (b, m, l, k) -> ((b*m), l, k)
                        prev = rearrange(prev, 'b m l k -> (b m) l k')
                        pred = rearrange(pred, 'b m l k -> (b m) l k')

                        # Forward pass depends on model type
                        if name == 'gpt':
                            out = model(prev, None, None, None)
                        elif name == 'transformer':
                            encoder_input = prev  # ((b*m), l, k)
                            dec_inp = torch.zeros_like(encoder_input[:, -pred_len:, :]).to(device)
                            decoder_input = torch.cat(
                                [encoder_input[:, prev_len - label_len:prev_len, :], dec_inp],
                                dim=1
                            )  # teacher-forcing warm start
                            out = model(encoder_input, decoder_input)  # -> ((b*m), pred_len, k)
                        elif name in ['lstm', 'rnn', 'gru']:
                            out = model(prev, pred_len, device)
                        elif name == 'cnn':
                            out = model(prev)
                        elif name == 'np':
                            # naive baseline: repeat the last observed step pred_len times
                            out = prev[:, [-1], :].repeat([1, pred_len, 1])

                        # NMSE on the flattened ((b*m), l, k) view
                        loss_nmse = criterion_nmse(out, pred)

                        # For SE: reshape back to (b, l, k*m) to build MISO H matrices
                        out_b = rearrange(out, '(b m) l k -> b l (k m)', b=bs)
                        pred_b = rearrange(pred, '(b m) l k -> b l (k m)', b=bs)

                        # Transform to physical channel shape & compute SE (negative as "loss")
                        # Nt=4*4=16, Nr=1
                        se, se0 = criterion_se(
                            h=Transform_TDD_FDD(out_b, Nt=4*4, Nr=1),
                            h0=Transform_TDD_FDD(pred_b, Nt=4*4, Nr=1)
                        )

                        # Collect scalars
                        test_loss_stack_nmse.append(loss_nmse.item())
                        test_loss_stack_se.append(se.item())     # negative SE
                        test_loss_stack_se0.append(se0.item())   # negative SE0

                # Aggregate across mini-batches for this speed
                nmse_mean = np.nanmean(np.array(test_loss_stack_nmse))
                # Convert to positive SE, SE0 for reporting:
                se_mean_pos = -np.nanmean(np.array(test_loss_stack_se))
                se0_mean_pos = -np.nanmean(np.array(test_loss_stack_se0))
                # Ratio uses negative values in numerator/denominator; signs cancel
                se_ratio = np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0))

                print(f"speed {speed}:  NMSE: {nmse_mean:.6f} | "
                      f"SE: {se_mean_pos:.6f} | SE0: {se0_mean_pos:.6f} | "
                      f"SE_per: {se_ratio:.6f}")

                # Store per-speed metrics
                NMSE[i].append(nmse_mean)
                SE_pred[i].append(se_mean_pos)
                SE0_true[i].append(se0_mean_pos)
                SE_ratio[i].append(se_ratio)

            elif name in ['pad', 'pvec']:
                # Classical baselines operate per-sample (no torch model)
                cycle_times = lens
                pbar_desc = f"{name} | speed={speed}"
                for cyt in tqdm.tqdm(range(cycle_times), desc=pbar_desc, leave=False):
                    # Take sample 'cyt' and convert to (k, l, m) for PAD/pvec functions
                    prev = test_data_prev[cyt, :, :, :]   # (m, l, k)
                    prev = rearrange(prev, 'm l k -> k l m', k=K)  # -> (k, l, m)
                    pred = test_data_pred[cyt, :, :, :]   # (m, l, k)
                    pred = rearrange(pred, 'm l k -> k l m', k=K)  # -> (k, l, m)

                    # Run the predictor
                    if name == 'pad':
                        # PAD in delay domain
                        out = PAD3(prev, p=8, startidx=prev_len, subcarriernum=K, Nr=Nr, Nt=Nt, pre_len=pred_len)
                    elif name == 'pvec':
                        # Prony in frequency domain
                        out = pronyvec(prev, p=8, startidx=prev_len, subcarriernum=K, Nr=Nr, Nt=Nt, pre_len=pred_len)

                    # Convert to torch format for metrics
                    out_t = LoadBatch_ofdm_1(out)    # -> (1, m, l, k) torch
                    pred_t = LoadBatch_ofdm_1(pred)  # -> (1, m, l, k) torch

                    # NMSE in torch space
                    loss_nmse = criterion_nmse(out_t, pred_t)

                    # For SE: turn (1, m, l, k) -> (1, l, k*m)
                    out_blkm = rearrange(out_t, 'b m l k -> b l (k m)')
                    pred_blkm = rearrange(pred_t, 'b m l k -> b l (k m)')

                    se, se0 = criterion_se(
                        h=Transform_TDD_FDD(out_blkm, Nt=4*4, Nr=1),
                        h0=Transform_TDD_FDD(pred_blkm, Nt=4*4, Nr=1)
                    )

                    test_loss_stack_nmse.append(loss_nmse.item())
                    test_loss_stack_se.append(se.item())
                    test_loss_stack_se0.append(se0.item())

                # Aggregate across samples for this speed
                nmse_mean = np.nanmean(np.array(test_loss_stack_nmse))
                se_mean_pos = -np.nanmean(np.array(test_loss_stack_se))
                se0_mean_pos = -np.nanmean(np.array(test_loss_stack_se0))
                se_ratio = np.nanmean(np.array(test_loss_stack_se)) / np.nanmean(np.array(test_loss_stack_se0))

                print(f"speed {speed}:  NMSE: {nmse_mean:.6f} | "
                      f"SE: {se_mean_pos:.6f} | SE0: {se0_mean_pos:.6f} | "
                      f"SE_per: {se_ratio:.6f}")

                NMSE[i].append(nmse_mean)
                SE_pred[i].append(se_mean_pos)
                SE0_true[i].append(se0_mean_pos)
                SE_ratio[i].append(se_ratio)

    # ----------------------- CSV export -----------------------
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    suffix = f"{mode_str}_full"  # matches previous naming style

    # Helper to write rows (one model per row, 10 comma-separated speeds)
    def write_csv(filename, rows):
        with open(filename, "w") as f:
            for row in rows:
                row = list(map(str, row))
                f.write(','.join(row))
                f.write('\n')

    # NMSE (as in original script)
    nmse_file = f"{ts}_data_nmse_{suffix}.csv"
    write_csv(nmse_file, NMSE)
    print(f"Saved NMSE CSV -> {nmse_file}")

    # NEW: SE (positive predicted SE)
    se_file = f"{ts}_data_se_{suffix}.csv"
    write_csv(se_file, SE_pred)
    print(f"Saved SE CSV -> {se_file}")

    # NEW: SE0 (positive oracle SE with perfect CSI)
    se0_file = f"{ts}_data_se0_{suffix}.csv"
    write_csv(se0_file, SE0_true)
    print(f"Saved SE0 CSV -> {se0_file}")

    # NEW: SE ratio (SE / SE0)
    ratio_file = f"{ts}_data_se_ratio_{suffix}.csv"
    write_csv(ratio_file, SE_ratio)
    print(f"Saved SE ratio CSV -> {ratio_file}")
    print("All done.")
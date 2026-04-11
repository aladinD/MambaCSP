# MambaCSP

## Title
**MambaCSP: A Mamba-based CSI Prediction Framework for DMRS-based Channel Reconstruction**

## Abstract
This repository provides the MambaCSP implementation for DMRS channel prediction on time-frequency CSI tensors.  
It includes dataset generation, unified training, and unified evaluation for both GPT-2 and Mamba backbones under TDD/FDD settings.  
The code is organized for reproducible end-to-end experiments: generate data with QuaDRiGa, train checkpoints, and evaluate NMSE/SE metrics.

## Environment
- Tested with Python 3.10
- NVIDIA GPU + CUDA
- Requirements from `requirements.txt`
- MATLAB with [QuaDRiGa](https://quadriga-channel-model.de/) for dataset generation

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Generation (DMRS)
`dataset_generation.m` is the generator entrypoint.

1. Open MATLAB and add QuaDRiGa/toolbox dependencies.
2. Edit `dataset_generation.m` config:
   - `cfg.pattern_id = 1|2|3`
   - `cfg.duplex = 'TDD' | 'FDD'`
   - `cfg.out_root = './dmrs_datasets'` (default output root)
3. Run in MATLAB:
   ```matlab
   >> run('dataset_generation.m')
   ```
4. Repeat for the three patterns and both duplex modes (or keep the script with your preferred `pattern_id`/`duplex` values).

### Output layout
By default, each run writes one folder:
`dmrs_datasets/<pattern_name>/<TDD|FDD>/`

Required files (names must match training/testing code):
- `H_U_his_train.mat` (`H_U_his_train`)
- `H_U_pre_train.mat` (`H_U_pre_train`)
- `H_D_pre_train.mat` (`H_D_pre_train`)
- `H_U_his_test.mat` (`H_U_his_test`)
- `H_U_pre_test.mat` (`H_U_pre_test`)
- `H_D_pre_test.mat` (`H_D_pre_test`)
- optional metadata: `meta.mat`

Pattern names used by default in this repo:
- `pattern1_type1_sparse`
- `pattern2_type2_densefreq`
- `pattern3_highmob_dense`

`run_training.sh` and `test_dmrs.py` expect this exact folder layout.

## Train
DMRS training uses:
- `train_dmrs.py` (DDP + AMP, unified GPT/Mamba entrypoint)
- `run_training.sh` (batch launcher for all default pattern+mode combinations)

### 1) Batch training (recommended)
```bash
chmod +x run_training.sh
./run_training.sh
```
Defaults:
- 8 GPUs (`NPROC=8`)
- backbone: GPT (`BACKBONE=gpt`)
- all three default patterns
- both TDD and FDD for each pattern

### 2) Select backbone
- GPT:
  ```bash
  BACKBONE=gpt ./run_training.sh
  ```
- Mamba (compact local Mamba):
  ```bash
  BACKBONE=mamba ./run_training.sh
  ```
- Mamba with HF checkpoint:
  ```bash
  BACKBONE=mamba USE_HF_MAMBA=1 HF_NAME=state-spaces/mamba-370m-hf ./run_training.sh
  ```

### 3) Single command (one experiment)
```bash
torchrun --nproc_per_node 8 train_dmrs.py \
  --backbone gpt \
  --u2d 0 \
  --train-his ./dmrs_datasets/pattern1_type1_sparse/TDD/H_U_his_train.mat \
  --train-tgt ./dmrs_datasets/pattern1_type1_sparse/TDD/H_U_pre_train.mat \
  --save-path ./dmrs_model_weights/pattern1_type1_sparse/TDD/U2U_LLM4CP.pth
```
To train FDD, set `--u2d 1` and point `--train-tgt` to `H_D_pre_train.mat`.

### 4) Few-shot mode
Set `--few 1` to enable few-shot splits from `Dataset_Pro`.

Checkpoints default to:
`dmrs_model_weights/<pattern>/<TDD|FDD>/<U2U|U2D>_LLM4CP.pth`

### 5) Notes on DDP and AMP
- **DDP (Distributed Data Parallel)**
  - `train_dmrs.py` is DDP-ready and is launched with `torchrun`.
  - With `torchrun --nproc_per_node N`, each process handles one GPU (`LOCAL_RANK`), and DDP synchronizes model gradients across processes.
  - `DistributedSampler` is used so each process sees a different minibatch shard.
  - Validation loss is reduced across ranks, and only rank 0 writes checkpoints.
  - In `run_training.sh`, `NPROC` controls how many processes (typically GPUs) are used.

- **AMP (Automatic Mixed Precision)**
  - Forward and loss are wrapped with `torch.cuda.amp.autocast(...)` to run in lower precision when safe.
  - `train_dmrs.py` uses `GradScaler` when BF16 is not used.
  - On BF16-capable GPUs, it uses BF16 autocast with `scaler` disabled; otherwise it falls back to FP16 + scaler.
  - AMP reduces memory pressure and often improves throughput, which is useful for long DMRS training runs.

## Test
Evaluation uses:
- `test_dmrs.py`

Example:
```bash
python test_dmrs.py \
  --backbone gpt \
  --data-root ./dmrs_datasets \
  --weight-root ./dmrs_model_weights \
  --results-root ./dmrs_results \
  --patterns pattern1_type1_sparse pattern2_type2_densefreq
```

For Mamba checkpoints trained with HF/local Mamba, use the same command with `--backbone mamba` and optional Mamba flags:
```bash
python test_dmrs.py \
  --backbone mamba \
  --use-hf-mamba \
  --hf-name state-spaces/mamba-370m-hf \
  --weight-root ./dmrs_model_weights \
  --weight-subdir mamba
```

Outputs are saved under `dmrs_results/`:
- `<name>_tdd_<backbone>_nmse.csv`
- `<name>_tdd_<backbone>_se.csv`
- `<name>_tdd_<backbone>_se0.csv`
- `<name>_tdd_<backbone>_se_ratio.csv`
and similarly for `fdd`.

`test_dmrs.py` defaults to evaluating all 10 user speeds and both TDD/FDD.

## Citation
If you use this repository, cite the relevant works:

1. Original baseline codebase:
```bibtex
@article{liu2024llm4cp,
  title={LLM4CP: Adapting Large Language Models for Channel Prediction},
  author={Liu, Boxun and Liu, Xuanyu and Gao, Shijian and Cheng, Xiang and Yang, Liuqing},
  journal={arXiv preprint arXiv:2406.14440},
  year={2024}
}
```

2. MambaCSP manuscript:
   - Replace with the final BibTeX entry of your paper.

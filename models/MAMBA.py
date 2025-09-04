import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# same embedding module used in GPT4CP
from Embed import DataEmbedding

# optional: mirror their mirror endpoint behavior
# os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# ----- Optional custom Mamba backbone (route A) -----
try:
    from mamba_ssm import Mamba2  # pip install mamba-ssm
    HAS_MAMBA_SSM = True
except Exception:
    HAS_MAMBA_SSM = False


# ---------------- CSI attention blocks (as in GPT4CP) ----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class Res_block(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv1(x))
        rs1 = self.conv2(rs1)
        out = self.ca(rs1) * rs1
        return torch.add(x, out)


# ---------------- Small Mamba stack (route A) ----------------
class _MambaBlock(nn.Module):
    def __init__(self, d_model=768, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not HAS_MAMBA_SSM:
            raise ImportError("mamba-ssm not installed. Install or use use_hf=True to load HF Mamba.")
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):           # x: (B, L, F)
        return x + self.mamba(self.norm(x))


class _MambaBackbone(nn.Module):
    def __init__(self, d_model=768, n_layers=6, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            _MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):           # (B, L, F)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)    # (B, L, F)


# ---------------- Mamba model (drop-in for GPT4CP.Model) ----------------
class Model(nn.Module):
    """
    Drop-in replacement for GPT4CP.Model but with a Mamba backbone.

    Two backends:
      - use_hf=True  : load a pretrained HF Mamba (pass hf_name), freeze most weights, feed inputs_embeds
      - use_hf=False : build a compact Mamba stack with mamba-ssm (train from scratch or partially freeze)

    Other parts (preproc/embedding/head) follow GPT4CP.
    """
    def __init__(self,
                 # backbone config
                 use_hf=False,
                 hf_name: str = None,           # e.g., "state-spaces/mamba-370m-hf"
                 d_model=768,                   # CSI embedding width (matches GPT2 small regime)
                 mamba_layers=6, d_state=16, d_conv=4, expand=2,
                 # task config
                 pred_len=4, prev_len=16,
                 use_gpu=1, gpu_id=0,
                 # antenna/subcarrier config
                 K=48, UQh=1, UQv=1, BQh=1, BQv=1,
                 # patching/embedding/head config
                 patch_size=4, stride=1, res_layers=4, res_dim=64,
                 embed='timeF', freq='h', dropout=0.1):
        super().__init__()

        self.device = torch.device(f'cuda:{gpu_id}' if use_gpu else 'cpu')
        self.pred_len = pred_len
        self.prev_len = prev_len
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model

        self.K = K
        self.UQh = UQh
        self.UQv = UQv
        self.BQh = BQh
        self.BQv = BQv
        self.Nt = UQh * UQv
        self.Nr = BQh * BQv
        self.enc_in = K * self.Nt * self.Nr
        self.c_out = self.enc_in  # output channels (real+imag handled later)

        # Embedding to d_model (same as GPT4CP)
        self.enc_embedding1 = DataEmbedding(2 * self.enc_in, self.d_model, embed, freq, dropout)

        # ----- Backbone selection -----
        self.use_hf = use_hf
        if use_hf:
            # route B: load a pretrained HF Mamba and feed inputs_embeds
            from transformers import AutoModel  # works for most HF mambas
            self.hf_mamba = AutoModel.from_pretrained(
                hf_name or "state-spaces/mamba-370m-hf",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                output_hidden_states=False
            )
            self.mamba_dim = int(getattr(self.hf_mamba.config, "hidden_size", self.d_model))

            # projection layers to/from HF hidden size
            self.to_mamba = nn.Linear(self.d_model, self.mamba_dim, bias=False)
            self.from_mamba = nn.Linear(self.mamba_dim, self.d_model, bias=False)

            # optional: checkpointing
            # if hasattr(self.hf_mamba, "gradient_checkpointing_enable"):
            #     self.hf_mamba.gradient_checkpointing_enable()

            # freeze most weights, keep norms trainable (mirrors GPT-2 policy)
            for n, p in self.hf_mamba.named_parameters():
                p.requires_grad = False
                low = n.lower()
                if any(k in low for k in ["norm", "layernorm", "ln"]):
                    p.requires_grad = True

            if use_gpu:
                self.hf_mamba.to(self.device)

        else:
            # route A: compact custom Mamba (trainable/backbone init)
            self.hf_mamba = None
            self.mamba_dim = self.d_model
            self.backbone = _MambaBackbone(
                d_model=self.d_model, n_layers=mamba_layers,
                d_state=d_state, d_conv=d_conv, expand=expand
            )
            if use_gpu:
                self.backbone.to(self.device)

        # small linear along time (as in GPT4CP)
        self.predict_linear_pre = nn.Linear(self.prev_len, self.prev_len)

        # head: map features to (2*c_out) per time step, then prev_len -> pred_len
        self.out_layer_dim = nn.Linear(self.d_model, self.c_out * 2)
        self.output_layer_time = nn.Linear(self.prev_len, self.pred_len)

        # CSI attention encoder in delay & frequency domains (same as GPT4CP)
        self.RB_e = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        self.RB_f = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        for _ in range(res_layers):
            self.RB_e.append(Res_block(res_dim))
            self.RB_f.append(Res_block(res_dim))
        self.RB_e.append(nn.Conv2d(res_dim, 2, 3, 1, 1))
        self.RB_f.append(nn.Conv2d(res_dim, 2, 3, 1, 1))

        # simple temporal "patch" mixing (kept to mirror GPT4CP)
        self.patch_layer = nn.Linear(self.patch_size, self.patch_size)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # ----- normalize -----
        mean = torch.mean(x_enc)
        std = torch.std(x_enc)
        x_enc = (x_enc - mean) / (std + 1e-6)

        B, L, D = x_enc.shape  # D = 2 * enc_in (real+imag packed)

        # ----- delay-domain path (IFFT over subcarriers) -----
        x_r = rearrange(x_enc, 'b l (k o) -> b l k o', o=2)
        x_complex = torch.complex(x_r[..., 0], x_r[..., 1])          # (B, L, K)
        x_delay = torch.fft.ifft(x_complex, dim=2)                    # (B, L, K) complex
        x_delay = torch.cat([x_delay.real, x_delay.imag], dim=2)      # (B, L, 2K)

        # ----- simple temporal patch mixing (same as GPT4CP) -----
        x_delay = x_delay.reshape(B, L // self.patch_size, self.patch_size, D)
        x_delay = self.patch_layer(x_delay.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_delay = x_delay.reshape(B, L, D)
        x_delay = rearrange(x_delay, 'b l (k o) -> b o l k', o=2)     # (B, 2, L, K)
        x_delay = self.RB_f(x_delay)

        # ----- frequency path -----
        x_fre = x_enc.reshape(B, L // self.patch_size, self.patch_size, D)
        x_fre = self.patch_layer(x_fre.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x_fre = x_fre.reshape(B, L, D)
        x_fre = rearrange(x_fre, 'b l (k o) -> b o l k', o=2)         # (B, 2, L, K)
        x_fre = self.RB_e(x_fre)

        # fuse domains & go back to (B,L,D)
        x_enc = x_fre + x_delay
        x_enc = rearrange(x_enc, 'b o l k -> b l (k o)', o=2)

        # ----- embed to d_model -----
        enc_out = self.enc_embedding1(x_enc, x_mark_enc)              # (B, L, d_model)

        # optional linear along time (kept from GPT4CP)
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # (B,L,d_model)

        # ----- Mamba backbone -----
        if self.use_hf:
            # project to HF hidden size and feed as inputs_embeds
            inp = self.to_mamba(enc_out)                              # (B, L, mamba_dim)
            m_out = self.hf_mamba(inputs_embeds=inp, use_cache=False).last_hidden_state  # (B, L, mamba_dim)
            dec_out = self.from_mamba(m_out)                          # (B, L, d_model)
        else:
            dec_out = self.backbone(enc_out)                          # (B, L, d_model)

        # ----- head: per-timestep map to 2*c_out, then time map prev_len->pred_len -----
        dec_out = self.out_layer_dim(dec_out)                         # (B, L, 2*c_out)
        dec_out = self.output_layer_time(dec_out.permute(0, 2, 1)).permute(0, 2, 1)  # (B, pred_len, 2*c_out)

        # denormalize & return last pred_len time steps
        dec_out = dec_out * std + mean
        return dec_out[:, -self.pred_len:, :]                         # (B, pred_len, 2*c_out)


if __name__ == "__main__":
    # quick shape test (mirrors GPT4CP __main__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl = Model(UQh=1, UQv=1, BQh=1, BQv=1, use_hf=False, d_model=768).to(device)
    x = torch.rand(3, 16, 96, device=device)  # (B, prev_len, 2*K*Nt*Nr) with K=48, Nt=1, Nr=1 -> 2*48=96
    y = mdl(x, None, None, None)
    print("out shape:", y.shape)
    total = sum(p.numel() for p in mdl.parameters())
    learn = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"Number of parameter: {total/1e6:.5f}M | Learnable: {learn/1e6:.5f}M")
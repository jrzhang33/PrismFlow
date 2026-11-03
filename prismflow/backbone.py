import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        return self.dropout(x + self.pe)


class Transpose(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)


class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)


class GELU2(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        return self.layernorm(x) * (1 + scale) + shift


class FullAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class CrossAttention(nn.Module):
    def __init__(self, n_embd, condition_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output):
        B, T, C = x.size()
        T_E = encoder_output.size(1)
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


def _mlp(n_embd, mlp_hidden_times, activate, resid_pdrop):
    assert activate in ['GELU', 'GELU2']
    act = nn.GELU() if activate == 'GELU' else GELU2()
    return nn.Sequential(
        nn.Linear(n_embd, mlp_hidden_times * n_embd),
        act,
        nn.Linear(mlp_hidden_times * n_embd, n_embd),
        nn.Dropout(resid_pdrop),
    )


class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times, activate):
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = _mlp(n_embd, mlp_hidden_times, activate, resid_pdrop)

    def forward(self, x, timestep):
        x = x + self.attn(self.ln1(x, timestep))
        return x + self.mlp(self.ln2(x))


class Encoder(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate):
        super().__init__()
        self.blocks = nn.Sequential(*[
            EncoderBlock(n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
            for _ in range(n_layer)
        ])

    def forward(self, x, t):
        for block in self.blocks:
            x = block(x, t)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times, activate, condition_dim):
        super().__init__()
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn1 = FullAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn2 = CrossAttention(n_embd, condition_dim, n_head, attn_pdrop, resid_pdrop)
        self.ln_cross = AdaLayerNorm(n_embd)
        self.mlp = _mlp(n_embd, mlp_hidden_times, activate, resid_pdrop)

    def forward(self, x, encoder_output, timestep):
        x = x + self.attn1(self.ln1(x, timestep))
        x = x + self.attn2(self.ln_cross(x, timestep), encoder_output)
        return x + self.mlp(self.ln2(x))


class Decoder(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate,
                 condition_dim):
        super().__init__()
        self.blocks = nn.Sequential(*[
            DecoderBlock(n_embd, n_head, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate, condition_dim)
            for _ in range(n_layer)
        ])

    def forward(self, x, t, enc):
        for block in self.blocks:
            x = block(x, enc, t)
        return x


class TransformerBackbone(nn.Module):
    def __init__(self, n_feat, n_layer_enc=5, n_layer_dec=14, n_embd=1024, n_heads=16, attn_pdrop=0.1,
                 resid_pdrop=0.1, mlp_hidden_times=4, block_activate='GELU', max_len=2048):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)
        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)
        self.decoder = Decoder(n_layer_dec, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

    def encode(self, x, t):
        return self.encoder(self.pos_enc(self.emb(x)), t)

    def forward(self, x, t, return_enc=False):
        emb = self.emb(x)
        enc = self.encoder(self.pos_enc(emb), t)
        out = self.inverse(self.decoder(self.pos_dec(emb), t, enc))
        return (out, enc) if return_enc else out

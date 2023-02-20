import torch
from torch import nn, einsum
import torch.nn.functional as F

import math
from einops import rearrange, repeat, reduce

import matplotlib.pyplot as plt


def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")


class AbsolutePositionalEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len, l2nord_embed=False) -> None:
        super().__init__()
        self.scale = dim**-0.5 if not l2nord_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2nord_embed
        self.embed = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f"you are passing a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}"

        if not pos:
            pos = torch.arange(seq_len, device=device)
        pos_embed = self.embed(pos)
        pos_embed = pos_embed * self.scale
        return l2norm(pos_embed) if self.l2norm_embed else pos_embed


class ScaledSinusoidalEmbedding(nn.Module):

    def __init__(self, dim, theta=10000) -> None:
        super().__init__()
        assert (dim % 2) == 0
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta**-freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x, pos=None):
        seq_len, device = x.shape[1], x.device

        if not pos:
            pos = torch.arange(seq_len, device=device)

        embd = einsum("i, j -> i j", pos, self.inv_freq)
        embd = torch.cat([embd.sin(), embd.cos()], dim=-1)

        return embd * self.scale


class SinusoidalPositionalEmbdedding(nn.Module):

    def __init__(self, dim, max_seq_len) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float)) *
                             (-math.log(10000) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        assert x.shape[
            2] == self.dim, f"Channel of input of SinePosEmbedding must match dim, but got {x.shape[2]} vs {self.dim}"
        x += self.pe[:, :x.shape[1]]
        return x


if __name__ == '__main__':
    b, s, n = 32, 1024, 256
    x = torch.zeros(b, s, n)

    pos_embd1 = AbsolutePositionalEmbedding(n, s)
    pos_embd2 = ScaledSinusoidalEmbedding(n)
    pos_embd3 = SinusoidalPositionalEmbdedding(n, s)

    print(pos_embd1(x).shape)
    print(pos_embd2(x).shape)
    print(pos_embd3(x).shape)

    import seaborn as sns

    sns.heatmap(pos_embd3.get_buffer('pe').squeeze().numpy())
    plt.show()

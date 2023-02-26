import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

from torch.nn import Transformer


def get(var, default):
    assert default is not None, "default mustn't be None."
    if var is not None:
        return default
    else:
        return var


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        dim,
        dim_head=64,
        num_heads=8,
        dropout=0.,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim = dim
        self.embed_dim = num_heads * dim_head

        self.to_q = nn.Linear(dim, self.embed_dim)
        self.to_k = nn.Linear(dim, self.embed_dim)
        self.to_v = nn.Linear(dim, self.embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Linear(self.embed_dim, dim)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor = None,
                v: torch.Tensor = None):
        b, s, n = q.shape
        k = get(k, q)
        v = get(v, q)

        # scaled dot-product attention
        # project inputs to q, k, v
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # rearrange q, k, v to match num_heads
        q = rearrange(q, "b i (h d) -> b h i d", h=self.num_heads)
        k = rearrange(k, "b j (h d) -> b h j d", h=self.num_heads)
        v = rearrange(v, "b j (h d) -> b h j d", h=self.num_heads)

        # compute attention scores accross different keys
        scale = 1 / torch.sqrt(torch.tensor(self.dim_head,
                                            dtype=torch.float32))
        atten_scores = einsum(f'b h i d, b h j d -> b h i j', q, k) * scale
        atten_probs = torch.softmax(atten_scores, dim=-1)
        atten_out = einsum(f'b h i j, b h j d -> b h i d', atten_probs, v)

        # concate and linear project
        atten_out = rearrange('b h i d -> b i (h d)', atten_out)

        out = self.to_out(atten_out)

        return out


class FeedForward(nn.Module):

    def __init__(self,
                 dim,
                 dim_out=None,
                 dim_ff=None,
                 activation=F.gelu,
                 dropout=0.,
                 bias=True,
                 post_norm_act=False) -> None:
        super().__init__()

        dim_out = get(dim_out, dim)
        dim_ff = get(dim_ff, 4 * dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_ff, bias=bias), activation,
            nn.LayerNorm(dim_ff) if post_norm_act else nn.Identity(),
            nn.Dropout(dropout), nn.Linear(dim_ff, dim_out))

    def forwar(self, x):
        return self.ff(x)


class EncoderLayer(nn.Moduel):

    def __init__(self,
                 dim,
                 dim_head=64,
                 num_heads=8,
                 dropout=0.,
                 dim_out=None,
                 dim_ff=None,
                 activation=F.gelu,
                 bias=True,
                 post_norm_act=False,
                 norm_first=False) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.atten_layer = MultiHeadAttention(dim, dim_head, num_heads,
                                              dropout)
        self.ff = FeedForward(dim, dim_out, dim_ff, activation, dropout, bias,
                              post_norm_act)

        self.atten_norm = nn.LayerNorm(dim)
        if norm_first:
            self.ff_norm = nn.LayerNorm(dim)
        else:
            self.ff_norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        if self.norm_first:
            x += self.atten_layer(self.atten_norm(x))
            x += self.ff(self.ff_norm(x))
        else:
            x = self.atten_norm(x + self.atten_layer(x))
            x = self.ff_norm(x + self.ff(x))
        return x


class DecoderLayer(nn.Moduel):

    def __init__(self,
                 dim,
                 dim_head=64,
                 num_heads=8,
                 dropout=0.,
                 dim_out=None,
                 dim_ff=None,
                 activation=F.gelu,
                 bias=True,
                 post_norm_act=False,
                 norm_first=False) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.atten_layer = MultiHeadAttention(dim, dim_head, num_heads,
                                              dropout)
        self.cross_atten_layer = MultiHeadAttention(dim, dim_head, num_heads,
                                                    dropout)
        self.ff = FeedForward(dim, dim_out, dim_ff, activation, dropout, bias,
                              post_norm_act)

        self.atten_norm = nn.LayerNorm(dim)
        self.cross_atten_norm = nn.LayerNorm(dim)
        if norm_first:
            self.ff_norm = nn.LayerNorm(dim)
        else:
            self.ff_norm = nn.LayerNorm(dim_out)

    def forward(self, x, k=None, v=None):
        if self.norm_first:
            x += self.atten_layer(self.atten_norm(x))
            x += self.cross_atten_layer(self.cross_atten_norm(x), k, v)
            x += self.ff(self.ff_norm(x))
        else:
            x = self.atten_norm(x + self.atten_layer(x))
            x = self.cross_atten_norm(x + self.cross_atten_layer(x, k, v))
            x = self.ff_norm(x + self.ff(x))
        return x


class Encoder(nn.Module):

    def __init__(self,
                 dim,
                 dim_head=64,
                 num_heads=8,
                 dropout=0.,
                 dim_out=None,
                 dim_ff=None,
                 activation=F.gelu,
                 bias=True,
                 post_norm_act=False,
                 norm_first=False,
                 num_layers=1,
                 norm_after_layer=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(
                    EncoderLayer(dim, dim_head, num_heads, dropout, dim,
                                 dim_ff, activation, bias, post_norm_act,
                                 norm_first))
            else:
                self.layers.append(
                    EncoderLayer(dim, dim_head, num_heads, dropout, dim_out,
                                 dim_ff, activation, bias, post_norm_act,
                                 norm_first))
            if norm_after_layer:
                self.layers.append(nn.LayerNorm(dim_out))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 dim,
                 dim_head=64,
                 num_heads=8,
                 dropout=0.,
                 dim_out=None,
                 dim_ff=None,
                 activation=F.gelu,
                 bias=True,
                 post_norm_act=False,
                 norm_first=False,
                 num_layers=1,
                 norm_after_layer=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(
                    DecoderLayer(dim, dim_head, num_heads, dropout, dim,
                                 dim_ff, activation, bias, post_norm_act,
                                 norm_first))
            else:
                self.layers.append(
                    DecoderLayer(dim, dim_head, num_heads, dropout, dim_out,
                                 dim_ff, activation, bias, post_norm_act,
                                 norm_first))
            if norm_after_layer:
                self.layers.append(nn.LayerNorm(dim_out))

    def forward(self, x, mem=None):
        for layer in self.layers:
            if isinstance(layer, DecoderLayer):
                x = layer(x, mem, mem)
            else:
                x = layer(x)
        return x


class ViT(nn.Module):

    def __init__(self,
                 image_size,
                 patch_size,
                 atten_layers,
                 patch_dim=32,
                 num_classes=None,
                 post_embed_norm=False,
                 emb_dropout=0.) -> None:
        super().__init__()

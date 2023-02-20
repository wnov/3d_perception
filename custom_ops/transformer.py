import torch
import copy
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn.modules import MultiheadAttention, Transformer

from inspect import isfunction
from functools import partial
from collections import namedtuple
from typing import Sequence

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

Itermediate = namedtuple("Intermediates",
                         ["pre_softmax_attn", "post_softmax_attn"])


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if layer.bias:
        nn.init.constant_(layer.weight, 0.)


def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")


class Residual(nn.Module):

    def __init__(self,
                 dim,
                 scale_residual=False,
                 scale_residual_constant=1.) -> None:
        super().__init__()
        self.residual_scale = nn.Parameter(
            torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if self.residual_scale:
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant
        return x + residual


class GLU(nn.Module):

    def __init__(self, dim, dim_out, activation) -> None:
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim, 2 * dim_out)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class ReluSquared(nn.Module):

    def forward(self, x):
        return F.relu(x)**2


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 dim_head,
                 heads=8,
                 drop_out=0.0,
                 sparse_topk=None,
                 value_dim_head=None) -> None:
        super().__init__()

        self.heads = heads
        self.sparse_topk = sparse_topk
        self.q_dim = self.k_dim = dim_head * heads
        self.v_dim = out_dim = value_dim_head if value_dim_head else dim_head * heads

        self.to_q = nn.Linear(dim, self.q_dim, bias=False)
        self.to_k = nn.Linear(dim, self.k_dim, bias=False)
        self.to_v = nn.Linear(dim, self.v_dim, bias=False)

        self.atten_fn = partial(F.softmax, dtype=torch.float32)

        self.drop_out = nn.Dropout(drop_out)

        self.to_out = nn.Linear(out_dim, dim, bias=False)

    def forward(
        self,
        quiery,
        context=None,
        mask=None,
        context_mask=None,
        atten_mask=None,
        prev_attn=None,
        mem=None,
    ):
        b, n, _, h = *quiery.shape, self.heads

        kv_input = default(context, quiery)

        q_input = quiery
        k_input = kv_input
        v_input = kv_input

        if mem:
            k_input = torch.cat([mem, k_input], dim=-2)
            v_input = torch.cat([mem, k_input], dim=-2)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q = rearrange(q, "b i (h d) -> b h i d", h=h)
        k, v = map(lambda t: rearrange(t, 'b i (h d) -> b h i d', h=h), (k, v))

        dots = einsum(f'b h i d, b h j d -> b h i j', q, k)

        if prev_attn:
            dots = dots + prev_attn

        mask_value = max_neg_value(dots)

        pre_softmax_atten = dots.clone()

        context_mask = default(context_mask, mask)

        if context_mask:
            context_mask = rearrange(context_mask, "b j -> b 1 1 j")
            dots = dots.masked_fill(context_mask, mask_value)
            del context_mask

        if atten_mask:
            assert 2 <= atten_mask.dim <= 4, "attension mask must be greater than 2 and less than 4."
            if atten_mask.dim == 2:
                atten_mask = rearrange(atten_mask, "i j -> 1 1 i j")
            elif atten_mask.dim == 3:
                atten_mask = rearrange(atten_mask, "h i j -> 1 h i j")
            dots = dots.masked_fill(atten_mask, mask_value)

        if self.sparse_topk and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            thresh = rearrange(top[..., -1], "... -> ... 1")
            sparse_topk_mask = dots < thresh
            dots = dots.masked_fill(sparse_topk_mask, mask_value)

        dtype = dots.dtype
        atten = self.atten_fn(dots, dim=-1)
        atten = atten.type(dtype)

        post_softmax_atten = atten.clone()

        atten = self.drop_out(atten)

        out = einsum(f'b h i j, b h j d -> b h i d', atten, v)

        out = rearrange(out, "b h j d -> b j (h d)")

        itermediateRecursionError = Itermediate(
            pre_softmax_attn=pre_softmax_atten,
            post_softmax_attn=post_softmax_atten)

        out = self.to_out(out)

        if mask:
            mask = rearrange(mask, "b j -> b j 1")
            out = out.masked_fill(~mask, 0.)
        return out, itermediate


class FeedForward(nn.Module):

    def __init__(self,
                 dim,
                 dim_out=None,
                 mult=4,
                 glu=False,
                 swish=False,
                 relu_squared=False,
                 post_act_ln=False,
                 drop_out=0.,
                 no_bias=False,
                 zero_init_output=False) -> None:
        super().__init__()
        inner_dim = dim * mult
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        projection = nn.Sequential(nn.Linear(dim, inner_dim, bias=not no_bias),
                                   activation) if not glu else GLU(
                                       dim, inner_dim, activation)

        self.ff = nn.Sequential(
            projection,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(drop_out),
            nn.Linear(inner_dim, dim_out, bias=not no_bias))

        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)


class AttenLayer(nn.Module):
    """
    Currently pre_norm is applied by default.
    """

    def __init__(
        self,
        dim,
        dim_head,
        heads=8,
        drop_out=0.0,
        norm=None,
        sparse_topk=None,
        value_dim_head=None,
        dim_out=None,
        mult=4,
        norm_ff=None,
        glu=False,
        swish=False,
        relu_squared=False,
        post_act_ln=False,
        drop_out_ff=0.,
        no_bias=False,
        zero_init_output=False,
        residual=None,
        scale_residual=False,
        scale_residual_constant=1.,
        type='encoder',
    ) -> None:
        super().__init__()
        assert type in [
            'encoder', 'decoder'
        ], "layer attribute 'type' must be in ['encoder', 'decoder']."
        self.type = type
        self.dim = dim
        self.self_attention = Attention(dim, dim_head, heads, drop_out,
                                        sparse_topk, value_dim_head)
        self.ff = FeedForward(dim, dim_out, mult, glu, swish, relu_squared,
                              post_act_ln, drop_out_ff, no_bias,
                              zero_init_output)
        norm = nn.LayerNorm if not norm else norm
        norm_ff = nn.LayerNorm if not norm_ff else norm_ff
        self.norm = norm(dim)
        self.norm_ff = norm_ff(dim_out) if dim_out else norm_ff(dim)
        residual = Residual if not residual else residual
        self.residual = residual(dim, scale_residual, scale_residual_constant)
        self.residual_ff = residual(dim, scale_residual,
                                    scale_residual_constant)
        if self.type == 'decoder':
            self.cross_attention = Attention(dim, dim_head, heads, drop_out,
                                             sparse_topk, value_dim_head)
            self.norm_cross = norm(dim)
            self.residual_cross = residual(dim, scale_residual,
                                           scale_residual_constant)

    def forward(self,
                x,
                context=None,
                mask=None,
                context_mask=None,
                attn_mask=None,
                memory_mask=None,
                memory_context_mask=None,
                memory_attn_mask=None,
                prev_mask=None,
                mem=None):
        res = x
        x, itermediate = self.self_attention(self.norm(x), None, mask,
                                             context_mask, attn_mask,
                                             prev_mask, mem)
        x = self.residual(x, res)

        if self.type == 'decoder':
            res = x
            x = self.cross_attention(self.norm_cross(x), context, memory_mask,
                                     memory_context_mask, memory_attn_mask,
                                     None, None)
            x = self.residual_cross(x, res)

        res = x
        x = self.ff(self.norm_ff(x))
        x = self.residual_ff(x, res)
        return x


class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None) -> None:
        super().__init__()
        self.dim = encoder_layer.dim
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                x,
                mask=None,
                context_mask=None,
                atten_mask=None,
                prev_mask=None,
                mem=None):
        for layer in self.layers:
            x = layer(x,
                      x,
                      mask,
                      context_mask,
                      atten_mask,
                      prev_mask=prev_mask,
                      mem=mem)
        if self.norm:
            x = self.norm(x)
        return x


class Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None) -> None:
        super().__init__()
        self.dim = decoder_layer.dim
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                x,
                context,
                mask=None,
                context_mask=None,
                atten_mask=None,
                memory_mask=None,
                memory_context_mask=None,
                memory_attn_mask=None,
                prev_mask=None,
                mem=None):
        for layer in self.layers:
            x = layer(x, context, mask, context_mask, atten_mask, memory_mask,
                      memory_context_mask, memory_attn_mask, prev_mask, mem)

        if self.norm:
            x = self.norm(x)
        return x


class ViTransformer(nn.Module):

    def __init__(self,
                 image_size,
                 patch_size,
                 attn_layers,
                 channels=3,
                 num_classes=None,
                 post_emb_norm=False,
                 emb_dropout=0.) -> None:
        super().__init__()
        assert isinstance(attn_layers,
                          Encoder), "attn_layer must be a enbcoder."
        if isinstance(image_size, int):
            assert image_size % patch_size == 0, "image_szie must be divisible by path_size."
        else:
            assert isinstance(image_size, Sequence) and isinstance(
                patch_size, Sequence
            ) and len(image_size) == 2 and len(
                patch_size
            ) == 2, "If image_size and patch_size aren't intigers, they must be sequence and len is 2."
            for s1, s2 in zip(image_size, patch_size):
                assert s1 % s2 == 0, "image_szie must be divisible by path_size."
        dim = attn_layers.dim
        num_patch = (image_size // patch_size)**2 if isinstance(
            image_size,
            int) else (image_size[0] // patch_size[0]) * (image_size[1] //
                                                          patch_size[1])
        patch_dim = channels * patch_size**2 if isinstance(
            patch_size, int) else channels * patch_size[0] * patch_size[1]
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, dim))
        self.patch_to_embdding = nn.Linear(patch_dim, dim)
        self.post_emb_norm = nn.LayerNorm(
            dim) if post_emb_norm else nn.Identity()
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(
            dim, num_classes) if num_classes else nn.Identity()

    def forward(self, img, return_embddings=False):
        p = self.patch_size
        if isinstance(p, int):
            p1 = p2 = p
        else:
            p1, p2 = p
        x = rearrange(img,
                      "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                      p1=p1,
                      p2=p2)
        x = self.patch_to_embdding(x)
        num_patch = x.shape[1]

        x += self.pos_embedding[:, :num_patch]

        x = self.post_emb_norm(x)
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)

        if not isinstance(self.mlp_head, nn.Linear) or return_embddings:
            return x

        x = x.mean(dim=-2)
        return self.mlp_head(x)


if __name__ == "__main__":
    bs = 2
    dim = 64
    encoder_seq_len = 10
    decoder_seq_len = 12
    num_encoders = 2
    num_decoders = 2

    layer1 = AttenLayer(dim,
                        dim_head=32,
                        heads=2,
                        drop_out=0.1,
                        sparse_topk=5,
                        value_dim_head=32,
                        dim_out=64,
                        drop_out_ff=0.1,
                        no_bias=True,
                        zero_init_output=False,
                        type='encoder')
    norm1 = nn.LayerNorm(dim)

    layer2 = AttenLayer(dim,
                        dim_head=32,
                        heads=2,
                        drop_out=0.1,
                        sparse_topk=5,
                        value_dim_head=32,
                        dim_out=64,
                        drop_out_ff=0.1,
                        no_bias=True,
                        zero_init_output=False,
                        type='encoder')
    norm2 = nn.LayerNorm(dim)

    encoder = Encoder(layer1, num_encoders, norm1)
    decoder = Decoder(layer2, num_decoders, norm2)

    src = torch.randn(bs, encoder_seq_len, dim)
    tgt = torch.randn(bs, decoder_seq_len, dim)

    memory = encoder(src)
    res = decoder(tgt, memory)
    print(res.shape)
    del encoder, decoder

    bs = 16
    img_size = (64, 64)
    patch_size = (8, 8)
    imgs = torch.randn(bs, 3, *img_size)

    dim = 32
    dim_head = 8
    heads = 4
    num_attn_layers = 2

    encoder_layer = AttenLayer(dim,
                               dim_head,
                               heads,
                               drop_out=0.1,
                               post_act_ln=True,
                               drop_out_ff=0.1,
                               no_bias=True)
    attn_layers = Encoder(encoder_layer, num_attn_layers)

    vit = ViTransformer(img_size, patch_size, attn_layers, 3, None, False, 0.1)
    res = vit(imgs, True)
    print(res.shape)

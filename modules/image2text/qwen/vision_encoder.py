import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Optional

class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout_prob: float = 0,
    ):
        super().__init__()
        
        self.use_sdp = int(torch.__version__[0]) > 1

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

        self.dropout_prob = dropout_prob
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:

        query = self.reshape(self.query(x))
        key = self.reshape(self.key(x if context is None else context))
        value = self.reshape(self.value(x if context is None else context))

        if self.use_sdp:
            x = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask,
                dropout_p=self.dropout_prob if self.training else 0,
                is_causal=is_causal,
            )
        else:
            attn = query @ key.transpose(-2, -1) * self.scale
            if attn_mask is not None:
                attn += attn_mask

            attn = attn.softmax(dim=-1)
            x = attn @ value

        return self.out(x.transpose(2, 1).flatten(2))

    def reshape(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(2, 1)


class MLP(nn.Module):
   
    def __init__(
        self,
        dim: int,
        dim_expand_factor: int = 4,
    ):
        super().__init__()

        self.hidden_layer = nn.Linear(dim, dim * dim_expand_factor)
        self.output_layer = nn.Linear(dim * dim_expand_factor, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.hidden_layer(x))
        return self.output_layer(x)


class LayerScale(nn.Module):

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False
    ):
        super().__init__()
        self.weight = nn.Parameter(init_values * torch.ones(dim))
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.weight) if self.inplace else x * self.weight


class VisionEncoderBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class VisionEncoder(nn.Module):
    
    def __init__(
        self,
        dim: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()

        self.n_patch = 224 // patch_size
        self.seq_len = self.n_patch ** 2
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.interpolate_offset = 0.1
        self.interpolate_antialias = False

        self.blocks = nn.Sequential(
            *[
                VisionEncoderBlock(dim, num_heads)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype

        if x.shape[1] == self.seq_len and w == h:
            return self.pos_embed
        
        pos_embed = self.pos_embed.float()

        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset
        sx, sy = float(w0) / self.n_patch, float(h0) / self.n_patch

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, self.n_patch, self.n_patch, dim).permute(0, 3, 1, 2),
            scale_factor=(sy, sx),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        return pos_embed.to(previous_dtype).flatten(start_dim=2).transpose(2, 1)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[2:]
        x = self.patch_embed(x).flatten(start_dim=2).transpose(2, 1)
        x = x + self.interpolate_pos_encoding(x, h, w)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.blocks(x)
        return self.norm(x)
""" ************************************************
* fileName: local_transformer.py
* desc: 
* author: mingdeng_cao
* date: 2021/07/07 15:31
* last revised: None
************************************************ """


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange



class Downsample(nn.Module):
    def __init__(self, in_channels, inner_channels, times=2):

        super().__init__()
        self.downsample = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (p1 p2 c) h w",
                      p1=times, p2=times),
            nn.Conv2d(in_channels * times * times, inner_channels, 1, 1, 0),
        )

    def forward(self, x):
        assert x.dim() == 4, "The input tensor should be in 4 dims!"
        out = self.downsample(x)
        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, inner_channels, up_times):

        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels *
                      up_times * up_times, 1, 1, 0),
            Rearrange(
                "b (p1 p2 c) h w -> b c (h p1) (w p2)",
                c=inner_channels,
                p1=up_times,
                p2=up_times,
            ),
        )

    def forward(self, x):
        assert x.dim() == 4, "Input tensor should be in 4 dims!"
        out = self.upsample(x)
        return out

class FFN(nn.Module):
    def __init__(self, in_channels, inner_channels=None, dropout=0.0):
        super().__init__()
        inner_channels = in_channels if inner_channels is None else inner_channels
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, inner_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_channels, in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class LocalAttnLayer(nn.Module):
    def __init__(
        self, embedding_dim, patch_size=4, num_heads=3, dropout=0.0, shift_size=0
    ):
        super().__init__()
        self.embedding = embedding_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.rearrange = Rearrange(
            "(p1 p2) (b hp wp) c -> b c (hp p1) (wp p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            hp=64 // self.patch_size,
            wp=64 // self.patch_size
        )

        self.to_patches = nn.Sequential(
            Rearrange(
                "b c (hp p1) (wp p2) -> (p1 p2) (b hp wp) c",
                p1=patch_size,
                p2=patch_size,
            )
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = FFN(embedding_dim, embedding_dim * 4, dropout)

        self.shift_size = shift_size
        self.row_embed = nn.Parameter(
            torch.Tensor(self.patch_size, self.embedding // 2)
        )
        self.col_embed = nn.Parameter(
            torch.Tensor(self.patch_size, self.embedding // 2)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed)
        nn.init.uniform_(self.col_embed)

    def get_learnable_pos(self):
        pos = (
            torch.cat(
                [self.row_embed.unsqueeze(1).repeat(1, self.patch_size, 1),
                 self.col_embed.unsqueeze(0).repeat(self.patch_size, 1, 1),],
                dim=-1,
            ).flatten(0, 1).unsqueeze(1)
        )
        return pos

    def get_padding_mask(self, padding_h, padding_w, x):
        B, C, H, W = x.shape

        if padding_h == 0 and padding_w == 0:
            return None

        img_mask = torch.zeros(1, 1, H, W).to(x)
        img_mask[..., H - padding_h:, :] = 1
        img_mask[..., W - padding_w:] = 1

        if self.shift_size > 0:
            img_mask = torch.roll(img_mask, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        img_mask_patches = self.to_patches(img_mask)
        img_mask_patches = img_mask_patches.squeeze(-1).transpose(0, 1)
        attn_mask = img_mask_patches.unsqueeze(1) - img_mask_patches.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0).masked_fill(
            attn_mask != 0, -100
        )

        return attn_mask.unsqueeze(1).repeat(B, self.num_heads, 1,
                                             1).flatten(0, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: (b, c, h, w), the input_feature maps
        """
        B, C, H, W = x.shape
        if self.shift_size > 0:
            x = torch.roll(
                x, shifts=(self.shift_size, self.shift_size), dims=(2, 3)
            )
        padding_h = (self.patch_size - H % self.patch_size) % self.patch_size
        padding_w = (self.patch_size - W % self.patch_size) % self.patch_size

        x = F.pad(x, (0, padding_w, 0, padding_h))
        h_paded, w_paded = x.shape[-2:]
        x_patches = self.to_patches(x)
        x_patches += self.get_learnable_pos()
        residual = x_patches
        x_patches = self.norm1(x_patches)
        x_attn = self.attn(
            query=x_patches, key=x_patches, value=x_patches, attn_mask=attn_mask
        )[0]

        x = x_attn + residual
        x = x + self.ffn(self.norm2(x))
        x = rearrange(
            x,
            "(p1 p2) (b hp wp) c -> b c (hp p1) (wp p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            hp=h_paded // self.patch_size,
            wp=w_paded // self.patch_size,
        )

        if padding_h > 0 or padding_w > 0:
            x = x[..., :H, :W]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -
                           self.shift_size), dims=(2, 3))
        return x
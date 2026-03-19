
from .MOE import MoE
from .local_transformer import Downsample
from .temporal_transformer import TemporalFusion
from simdeblur.model.build import BACKBONE_REGISTRY
from torchvision.models.resnet import ResNet, BasicBlock
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .local_transformer import Upsample, Downsample, LocalAttnLayer

class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels,
        down_times=4,
        patch_size=4,
        num_layers=12,
        num_heads=3,
        dropout=0.0,
        ffn_dim=None,
        attn_type="LocalAttnLayer",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.down_times = down_times
        self.patch_size = patch_size
        if down_times > 1:
            self.downsample = Downsample(in_channels, inner_channels, down_times)

        self.attn_layers = nn.Sequential(
            *[
                LocalAttnLayer(
                    self.inner_channels,
                    self.patch_size,
                    num_heads,
                    dropout,
                    shift_size=0 if i % 2 == 0 else self.patch_size // 2,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        """
        feats = x
        if self.down_times > 1:
            feats = self.downsample(feats)
        return self.attn_layers(feats)



class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)


        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )


        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


@BACKBONE_REGISTRY.register()
class CATF(nn.Module):
    def __init__(
        self,
        in_channels=3,
        inner_channels=256,
        num_frames=5,
        num_frames1 = 5,
        patch_size=4,
        cnn_patch_embedding=False,
        patch_embedding_size=4,
        temporal_patchsize=4,
        temporal_two_layer=False,
        num_layer_rec=20,
        num_heads=8,
        dropout=0.0,
        ffn_dim=None,
        ms_fuse=False
    ):
        super().__init__()
        self.num_frames = num_frames
        self.down_times = patch_embedding_size

        if cnn_patch_embedding:
            self.img2feats = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels // 4, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(inner_channels // 4, inner_channels // 2, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(inner_channels // 2, inner_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1)
            )
        else:
            self.img2feats = Downsample(in_channels, inner_channels, patch_embedding_size)


        self.temporal_fusion = TemporalFusion(
            inner_channels,
            num_frames,
            temporal_patchsize,
            num_heads,
            dropout,
            temporal_two_layer
        )
        self.reconstructor = nn.Sequential(
            EncoderBlock(
                inner_channels,
                inner_channels,
                down_times=1,
                patch_size=patch_size,
                num_layers=num_layer_rec,
                num_heads=num_heads,
                dropout=dropout,
                ffn_dim=ffn_dim,
            )
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels*4, 1, 1, 0),
            nn.PixelShuffle(2),
            nn.Conv2d(inner_channels, inner_channels*4, 1, 1, 0),
            nn.PixelShuffle(2)
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(inner_channels, 3, 1, 1, 0))

        self.upsample1 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels * 4, 1, 1, 0),
            nn.PixelShuffle(2),
            nn.Conv2d(inner_channels, inner_channels * 4, 1, 1, 0),
            nn.PixelShuffle(2)
        )
        self.out_proj1 = nn.Sequential(
            nn.Conv2d(inner_channels, 3, 1, 1, 0))
        self.NAFBlock = nn.Sequential(
            *[NAFBlock(inner_channels) for _ in range(5)]
        )
        self.NAFBlock1 = nn.Sequential(
            *[NAFBlock(inner_channels) for _ in range(25)]
        )
        self.NAFBlock2 = nn.Sequential(
            *[NAFBlock(inner_channels) for _ in range(5)]
        )

        self.alpha = nn.Parameter(torch.zeros((1, inner_channels, 1, 1)), requires_grad=True)
        self.MoE = MoE(input_size=256, num_experts = 3, k=2)


    def forward(self, x):

        assert x.dim() == 5, "Input tensor should be in 5 dims!"

        B, N, C, H, W = x.shape
        feats = self.img2feats(x.flatten(0, 1))
        feats = feats.reshape(B, 4, -1, H // self.down_times, W // self.down_times)
        input_1 = feats[:,0].unsqueeze(1)
        input_2 = feats[:,1]
        input_21 = feats[:, 2]
        input_3 = feats[:,3]
        feats1 = torch.cat((input_2.unsqueeze(1), input_3.unsqueeze(1), input_21.unsqueeze(1)), dim=1)
        print("111", feats1.shape)
        out = self.temporal_fusion(feats1)
        out = self.reconstructor(out)
        out = self.NAFBlock2(out)
        out_1, loss1 = self.MoE(out)
        out = out_1 + out
        outt = self.NAFBlock(input_1.squeeze(1))
        outt_1, loss2 = self.MoE(outt)
        outt = outt_1 + outt
        out2 = out + self.alpha * outt
        out2 = self.NAFBlock1(out2)
        out2_1, loss3 = self.MoE(out2)
        out2 = out2 + out2_1
        out5 = self.upsample(out2)
        out5 = self.out_proj(out5)
        out5 = out5 + x[:, 0]
        loss = loss1+loss2+loss3
        return out5, loss


if __name__ == "__main__":
    pass

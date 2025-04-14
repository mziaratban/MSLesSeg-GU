from functools import partial
import math
import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from torch import nn
from utils import merge_pre_bn

NORM_EPS = 1e-5


class ConvBNReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x



class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))






class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class ConvSA(nn.Module):
    def __init__(
            self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
            mlp_ratio=2, head_dim=32, mix_block_ratio=0.8, attn_drop=0, drop=0,
    ):
        super(ConvSA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)


        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.norm1 = norm_layer(out_channels)
        self.e_mhsa = E_MHSA(out_channels, head_dim=head_dim, sr_ratio=sr_ratio, attn_drop=attn_drop, proj_drop=drop)

        self.mhsa_path_dropout = DropPath(path_dropout)
        

        
        # ************** start of lffn **************
        expand_ratio = sr_ratio
        in_dim = out_channels
        out_dim = out_channels
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3
        stride = 1

        self.conv1x1_LFFN_1 = nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False)
        self.BN_LFFN_1 = nn.BatchNorm2d(hidden_dim)
        self.Relu = nn.ReLU6(inplace=True)
        
        self.convGroup_LFFN = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False)
        self.BNGroup_LFFN = nn.BatchNorm2d(hidden_dim)
        self.ReluGroup = nn.ReLU6(inplace=True)
        
        self.conv1x1_LFFN_2 = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False)
        self.BN_LFFN_2 = nn.BatchNorm2d(out_dim)
        # ************** end of lffn **************

        self.projection_1 = nn.Conv2d(3*out_dim, out_dim, 1, 1, 1, groups=1, bias=False)
        self.projection_3 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=out_dim, bias=False)

    
    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x0 = x

        
        out = x0
        out = rearrange(out, "b c h w -> b (h w) c")  # b n c
        out = self.e_mhsa(out)
        out = rearrange(out, "b (h w) c -> b c h w", h=H)
        out0 = self.mhsa_path_dropout(out)

                
        x = x0
        # ************** start of lffn **************
        x = self.conv1x1_LFFN_1(x)
        x = self.BN_LFFN_1(x)
        x = self.Relu(x)
        x = self.convGroup_LFFN(x)
        x = self.BNGroup_LFFN(x)
        x = self.ReluGroup(x)
        x = self.conv1x1_LFFN_2(x)
        x = self.BN_LFFN_2(x)
        x = self.Relu(x)
        out1 = self.mhsa_path_dropout(x)

        x1 = self.mhsa_path_dropout(x0)
        
        x = torch.cat([x1, out0, out1], dim=1)
        x = self.projection_1(x)
        #x = self.projection_3(x)
        #x = self.BN_LFFN_2(x)
        #out = self.Relu(x)
        out = x
        
        return out


class my_Net(nn.Module):
    def __init__(self, p_drop, use_checkpoint=False):
        super(my_Net, self).__init__()
        self.use_checkpoint = use_checkpoint


        self.stem = nn.Sequential(
            ConvBNReLU(3, 64, kernel_size=3, stride=2, groups=1),
            #ConvBNReLU(64, 128, kernel_size=3, stride=2, groups=16),
            #ConvBNReLU(128, 128, kernel_size=3, stride=1, groups=16),
            #ConvBNReLU(128, 128, kernel_size=3, stride=1, groups=16),
            #ConvBNReLU(128, 128, kernel_size=3, stride=1, groups=16),
            #ConvBNReLU(128, 128, kernel_size=3, stride=1, groups=16),
            #ConvBNReLU(128, 128, kernel_size=3, stride=1, groups=16),
        )

        sr_ratio = 2
        head_dim = 16
        features = []

        layer = ConvSA(64, 128, path_dropout=p_drop, stride=2, sr_ratio=sr_ratio, head_dim=head_dim)
        features.append(layer)
        layer = ConvSA(128, 128, path_dropout=p_drop, stride=2, sr_ratio=sr_ratio, head_dim=head_dim)
        features.append(layer)
        layer = ConvSA(128, 256, path_dropout=p_drop, stride=2, sr_ratio=sr_ratio, head_dim=head_dim)
        features.append(layer)
        layer = ConvSA(256, 512, path_dropout=p_drop, stride=2, sr_ratio=sr_ratio, head_dim=head_dim)
        features.append(layer)
        layer = ConvSA(512, 1024, path_dropout=p_drop, stride=2, sr_ratio=sr_ratio, head_dim=head_dim)
        features.append(layer)

        
        self.features = nn.Sequential(*features)

        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.norm = norm_layer(1024)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head = nn.Linear(1024, 3)

        print('initialize_weights...')
        self._initialize_weights()



    
    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)
        return x


@register_model
def my_Network(pretrained=False, pretrained_cfg=None, **kwargs):
    model = my_Net(p_drop=0.05, **kwargs)
    return model


import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class FreBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        xori = x
        # print(f"x.shape: {x.shape}")
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out1 = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x


class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.chara_ch = dim
        self.fre_ch = 3
        self.out_ch = dim

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                               groups=dim)
        self.mid_gelu1 = nn.GELU()

        self.gate = nn.Sequential(
            nn.Conv2d(self.fre_ch * 3 + self.out_ch, dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv_end = nn.Conv2d(dim, dim, kernel_size=1)
        self.fre_redim = nn.AdaptiveAvgPool2d((None, None))

    def forward(self, x, hl, lh, hh):
        B, C, H, W = x.shape
        # hl = self.conv_up(hl)
        # lh = self.conv_up(lh)
        # hh = self.conv_up(hh)
        # print(f"hh:{hh.shape}, hl:{lh.shape}")
        high_freq = torch.cat((hl, lh, hh), dim=1)
        high_freq = torch.nn.functional.adaptive_avg_pool2d(high_freq, (H, W))

        # 原始 CNN 处理
        cx = self.conv1(x)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)
        # print(f"cx:{cx.shape}")
        # print(f"high_freq:{high_freq.shape}")

        high_freq_weight = self.gate(torch.cat([cx, high_freq], dim=1))

        # print(f"high_freq:{high_freq.shape}")
        # print(f"high_freq_weight:{high_freq_weight.shape}")
        cx = cx + high_freq_weight

        cx = self.conv_end(cx)

        return cx


class LowMixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.fre_dim = 3

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.pool = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim,
                              bias=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=2, padding=1, output_padding=1,
                                         bias=False) if pool_size > 1 else nn.Identity()

        self.low_freq_fuse = nn.Conv2d(self.fre_dim + dim, dim, kernel_size=1, stride=1, padding=0)
        self.gate = nn.Sequential(
            nn.Conv2d(self.fre_dim + dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.se_block = SEBlock(dim)
        self.fre_redim = nn.AdaptiveAvgPool2d((None,None))

    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn - attn.max(dim=-1, keepdim=True)[0]  # 减去最大值，防止数值不稳定
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x

    def forward(self, x, ll):
        B, _, H, W = x.shape
        ll = torch.nn.functional.adaptive_avg_pool2d(ll,(H, W))
        # print(f"x: {x.shape}, ll: {ll.shape}")
        x_low = torch.cat([x, ll], dim=1)
        gate_weight = self.gate(x_low)  # 计算低频信息的门控权重
        x_low = self.low_freq_fuse(x_low) * gate_weight  # 门控融合
        x_low = self.se_block(x_low)
        x = self.pool(x_low)
        xa = x.permute(0, 2, 3, 1).view(B, -1, self.dim)  # B, N, C
        B, N, C = xa.shape
        qkv = self.qkv(xa).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, int(N ** 0.5), int(N ** 0.5))
        xa = self.uppool(xa)
        return xa


class MFDM(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, pool_size=2):
        super().__init__()
        self.num_heads = num_heads
        self.low_dim = low_dim = dim // 2
        self.high_dim = high_dim = dim - low_dim
        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(low_dim, num_heads=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                  pool_size=pool_size)

        self.conv_fuse = nn.Conv2d(low_dim + high_dim, low_dim + high_dim, kernel_size=3, stride=1, padding=1,
                                   bias=False, groups=low_dim + high_dim)
        self.proj = nn.Conv2d(low_dim + high_dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

        self.freblock = FreBlock(dim, dim)
        self.finalproj = nn.Conv2d(2 * dim, dim, 1, 1, 0)

    def forward(self, x, ll, hl, lh, hh):
        x_ori = x

        hx = x[:, :self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx, hl, lh, hh)

        lx = x[:, self.high_dim:, :, :].contiguous()
        lx = self.low_mixer(lx, ll)
        x = torch.cat((hx, lx), dim=1)

        x = x + self.conv_fuse(x)
        x_sptial = self.proj(x)
        x_freq = self.freblock(x_ori)

        x_out = torch.cat((x_sptial, x_freq), 1)
        x_out = self.finalproj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out + x_ori

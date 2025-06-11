import math
import torch
import torch.nn as nn
from model.BCP import bright_channel_prior_dehaze
from model.DCP import dark_channel_prior_dehaze
from model.DWT import DWT


from model.WEM import MyNet

from model.MFDM import Mfdm
from model.DPRM import Dprm


# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2

    # emb = math.log(10000) / (half_dim - 1)
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))  # 在嵌入矩阵的最后一列进行零填充。
    return emb



def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, incep=True):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.incep = incep

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        if incep:
            self.conv2 = Mfdm(dim=out_channels)

        else:
            self.conv2 = torch.nn.Conv2d(out_channels,
                                         out_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb, ll, hl, lh, hh):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        if self.incep:
            h = self.conv2(h, ll, hl, lh, hh)
        else:
            h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.restore_size

        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()

        self.inceplayers = 2

        self.FA = MyNet(3, 128)
        self.FA_flag = True
        self.incep = True
        self.Dprm_flag = True
        self.outlist = []
        self.outlist_re = []

        if self.FA_flag:
            print("Ues FA module")
        else:
            print("Not ues FA module")

        if self.Dprm_flag:
            print("Ues Dprm block")
        else:
            print("Not ues Dprm block")

        if self.incep:
            print("Ues Mfdm module")
        else:
            print("Not ues Mfdm module")


        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        #############################################################################################################

        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + ch_mult

        self.down = nn.ModuleList()
        block_in = None

        ### down
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                if i_level + self.inceplayers < self.num_resolutions:
                    block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout, incep=self.incep))
                else:
                    block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout, incep=self.incep))
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if ch_mult[i_level] * 2 == ch_mult[i_level + 1]:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                    curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout, incep=self.incep)

        if self.Mid_flag:
            self.mid.mid_block = Dprm(block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout, incep=self.incep)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                if i_level + self.inceplayers >= self.num_resolutions:
                    block.append(ResnetBlock(in_channels=block_in + skip_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout, incep=self.incep))
                else:
                    block.append(ResnetBlock(in_channels=block_in + skip_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout, incep=self.incep))
                block_in = block_out

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        # self.mynet = MyNet(3, 128)
        self.dwt = DWT()

    def forward(self, x, t):
        # print(f"x shape: {x.shape}")
        assert x.shape[2] == x.shape[3] == self.resolution

        x_hazy = x[:, :3, :, :]
        # print(f"x_haze:{x_hazy.shape}")
        x_dcp = dark_channel_prior_dehaze(x_hazy)
        x_bcp = bright_channel_prior_dehaze(x_hazy)
        x_ll, x_hl, x_lh, x_hh = self.dwt(x_hazy)
        # print(f"x_ll:{x_ll.shape}")
        if self.FA_flag:
            x_out1, x_out2, x_out3, x_out4 = self.FA(x_hazy)
            self.out_list = [[x_out1, x_out1], [x_out2, x_out2], [x_out3, x_out3], [x_out4, x_out4]]

        # timestep embedding

        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        # print(f"x shape:{x.shape}")
        # downsampling
        hs = [self.conv_in(x)]
        if self.FA_flag:
            for i_level in range(self.num_resolutions):
                for i_block in range(self.num_res_blocks):
                    if i_level < 3:
                        h = self.down[i_level].block[i_block](hs[-1], temb, x_ll, x_hl, x_lh, x_hh)
                        # print(f"{i_level}:{i_block}--{h.shape}")
                        h = h * self.out_list[i_level][i_block]
                    else:
                        h = self.down[i_level].block[i_block](hs[-1], temb, x_ll, x_hl, x_lh, x_hh)
                        # print(f"{i_level}:{i_block}--{h.shape}")
                        h = h + self.out_list[i_level][i_block]

                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    hs.append(h)
                if i_level != self.num_resolutions - 1:
                    hs.append(self.down[i_level].downsample(hs[-1]))
        else:
            for i_level in range(self.num_resolutions):
                for i_block in range(self.num_res_blocks):
                    h = self.down[i_level].block[i_block](hs[-1], temb, x_ll, x_hl, x_lh, x_hh)
                    if len(self.down[i_level].attn) > 0:
                        h = self.down[i_level].attn[i_block](h)
                    hs.append(h)
                if i_level != self.num_resolutions - 1:
                    hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb, x_ll, x_hl, x_lh, x_hh)
        h = self.mid.attn_1(h)
        if self.Dprm_flag:
            h = self.mid.mid_block(h, x_dcp, x_bcp)
        h = self.mid.block_2(h, temb, x_ll, x_hl, x_lh, x_hh)


        if self.FA_flag:
            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    if i_level == 0:
                        h = self.up[i_level].block[i_block](
                            torch.cat([h, hs.pop()], dim=1), temb, x_ll, x_hl, x_lh, x_hh)
                    else:
                        h = self.up[i_level].block[i_block](
                            torch.cat([h, hs.pop()], dim=1), temb, x_ll, x_hl, x_lh, x_hh)

                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
                if i_level != 0:
                    h = self.up[i_level].upsample(h)

        else:
            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    h = self.up[i_level].block[i_block](
                        torch.cat([h, hs.pop()], dim=1), temb, x_ll, x_hl, x_lh, x_hh)
                    if len(self.up[i_level].attn) > 0:
                        h = self.up[i_level].attn[i_block](h)
                if i_level != 0:
                    h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

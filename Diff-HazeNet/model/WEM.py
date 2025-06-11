import torch.nn as nn
import torch

from model.DWT import DWT
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


class CLP(nn.Module):
    def __init__(self, channel, norm=False):
        super(CLP, self).__init__()

        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_out = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.PReLU(channel)
        self.norm = nn.GroupNorm(num_channels=channel, num_groups=1)  # nn.InstanceNorm2d(channel)#

    def forward(self, x):
        x_1 = self.act(self.norm(self.conv_1(x)))
        x_2 = self.act(self.norm(self.conv_2(x_1)))
        x_out = self.act(self.norm(self.conv_out(x_2)) + x)
        return x_out


class CAB(nn.Module):
    def __init__(self, k_size=3):
        super(CAB, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x) + self.max_pool(x)
        y_temp = self.conv(y.squeeze(-1).transpose(-1, -2))
        y_temp = y_temp.unsqueeze(-1)
        y = y.transpose(-2, -3)
        y_temp = F.interpolate(y_temp, y.size()[2:], mode='bilinear')
        y = y_temp.transpose(-2, -3)

        camap = self.sigmoid(y)

        return camap


class WEM(nn.Module):
    def __init__(self, in_channel, channel):
        super(WEM, self).__init__()
        self.dwt = DWT()
        self.bb_ll = CLP(channel)
        self.bb_hl = CLP(channel)
        self.bb_lh = CLP(channel)
        self.bb_hh = CLP(channel)

        self.conv1x1_ll = nn.Conv2d(in_channel, channel, kernel_size=1, padding=0)
        self.conv1x1_hl = nn.Conv2d(in_channel, channel, kernel_size=1, padding=0)
        self.conv1x1_lh = nn.Conv2d(in_channel, channel, kernel_size=1, padding=0)
        self.conv1x1_hh = nn.Conv2d(in_channel, channel, kernel_size=1, padding=0)

        self.conv_out1 = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0)
        self.conv_out2 = nn.Conv2d(channel * 6, channel, kernel_size=1, stride=1, padding=0)
        self.conv_out3 = nn.Conv2d(channel * 4, 6 * channel, kernel_size=1, stride=1, padding=0, bias=False)

        # WEM中的A
        self.cab = CAB(2 * channel)
        self.cab_dwt = CAB(4 * channel)

    def forward(self, x):
        dwt_ll, dwt_hl, dwt_lh, dwt_hh = self.dwt(x)


        dwt_ll_resize = self.conv1x1_ll(dwt_ll)
        # print(f"dwt_ll_resize:{dwt_ll_resize.shape}")
        dwt_hl_resize = self.conv1x1_hl(dwt_hl)
        dwt_lh_resize = self.conv1x1_lh(dwt_lh)
        dwt_hh_resize = self.conv1x1_hh(dwt_hh)


        x_ll = self.bb_ll(dwt_ll_resize)
        # print(f"x_ll:{x_ll.shape}")
        x_hl = self.bb_hl(dwt_hl_resize)
        x_lh = self.bb_lh(dwt_lh_resize)
        x_hh = self.bb_hh(dwt_hh_resize)

        x_ll_hl = self.conv_out1(self.cab(torch.cat((x_ll, x_hl), 1)))
        # print(f"x_ll_hl:{x_ll_hl.shape}")
        x_ll_lh = self.conv_out1(self.cab(torch.cat((x_ll, x_lh), 1)))
        x_ll_hh = self.conv_out1(self.cab(torch.cat((x_ll, x_hh), 1)))
        x_hl_lh = self.conv_out1(self.cab(torch.cat((x_hl, x_lh), 1)))
        x_hl_hh = self.conv_out1(self.cab(torch.cat((x_hl, x_hh), 1)))
        x_lh_hh = self.conv_out1(self.cab(torch.cat((x_lh, x_hh), 1)))
        x_idwt = self.cab_dwt(torch.cat((x_ll, x_hl, x_lh, x_hh), 1))
        x_idwt = self.conv_out3(x_idwt)

        x_out = self.conv_out2(
            torch.cat((x_ll_hl, x_ll_lh, x_ll_hh, x_hl_lh, x_hl_hh, x_lh_hh), 1) + x_idwt)
        return x_ll, x_hl, x_lh, x_hh, x_out


class MyNet(nn.Module):
    def __init__(self, inchannel, channel):
        super(MyNet, self).__init__()
        self.wem1 = WEM(inchannel, channel)
        self.wem2 = WEM(channel * 2, channel * 2)
        self.wem3 = WEM(channel * 4, channel * 4)

        self.conv_r = nn.Conv2d(channel, 2 * channel, kernel_size=1, stride=1, padding=0)
        self.conv_g = nn.Conv2d(channel, 2 * channel, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(channel, 2 * channel, kernel_size=1, stride=1, padding=0)
        self.conv_d = nn.Conv2d(channel, 2 * channel, kernel_size=1, stride=1, padding=0)
        # self.conv_dp = nn.Conv2d(4 * channel, 8 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_dp = nn.Conv2d(2 * channel, 4 * channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_in = nn.Conv2d(inchannel, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_eltem = nn.Conv2d(channel, 4 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_emtes = nn.Conv2d(4 * channel, 16 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_dstdm = nn.Conv2d(16 * channel, 4 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_dmtdl = nn.Conv2d(4 * channel, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out = nn.Conv2d(channel, 3, kernel_size=1, stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # def visualize_output(self, output, title, index):
    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear')

    def forward(self, x):
        # print(f"x:{x.shape}")
        x_ll, x_hl, x_lh, x_hh, x_out_1 = self.wem1(x)

        x_r, _, _, _, _ = self.wem2(self.conv_r(self.maxpool(x_ll)))
        _, x_g, _, _, _ = self.wem2(self.conv_g(self.maxpool(x_hl)))
        _, _, x_b, _, _ = self.wem2(self.conv_b(self.maxpool(x_lh)))
        _, _, _, x_d, x_out_2 = self.wem2(self.conv_d(self.maxpool(x_hh)))

        _, _, _, _, x_out_3 = self.wem3(self.conv_dp(self.maxpool(x_r)))

        x_out_4 = torch.cat((x_r, x_g, x_b, x_d), 1)

        return x_out_1, x_out_2, x_out_3, x_out_4


def main(img_path):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to('cuda')
    mynet = MyNet(3, 128).to('cuda')

    with torch.no_grad():
        out1, out2, out3 = mynet(image_tensor)

    # print("output Shapes:")
    # print(f"Output 1: {out1.shape}")
    # print(f"Output 2: {out2.shape}")
    # print(f"Output 3: {out3.shape}")


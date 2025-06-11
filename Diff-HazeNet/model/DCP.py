import torch
import torch.nn.functional as F

def dark_channel(img, kernel_size=15):
    dark = torch.min(img, dim=1)[0]  # (B, H, W)
    dark = -F.max_pool2d(-dark.unsqueeze(1), kernel_size, stride=1, padding=kernel_size // 2)

    return dark.squeeze(1)  # (B, H, W)

def estimate_airlight(img, dark_channel_map, top_k=0.001):
    B, C, H, W = img.shape
    num_pixels = int(H * W * top_k)

    flat_dark = dark_channel_map.view(B, -1)
    top_indices = torch.argsort(flat_dark, descending=True)[:, :num_pixels]

    flat_img = img.view(B, C, -1)  # (B, C, H*W)
    airlight = torch.stack([torch.mean(flat_img[b, :, top_indices[b]], dim=1) for b in range(B)])

    return airlight  # (B, C)

def transmission_map(img, airlight, omega=0.95, kernel_size=15):
    B, C, H, W = img.shape
    normalized_img = img / airlight.view(B, C, 1, 1)
    dark = dark_channel(normalized_img, kernel_size)
    transmission = 1 - omega * dark
    return transmission.clamp(0.1, 1)

def recover_image(img, transmission, airlight, t_min=0.1):
    B, C, H, W = img.shape
    t = transmission.clamp(min=t_min).unsqueeze(1)  # (B, 1, H, W)
    recovered = (img - airlight.view(B, C, 1, 1)) / t + airlight.view(B, C, 1, 1)
    return recovered.clamp(0, 1)

def dark_channel_prior_dehaze(img):
    dark_map = dark_channel(img)
    airlight = estimate_airlight(img, dark_map)
    transmission = transmission_map(img, airlight)
    recovered_img = recover_image(img, transmission, airlight)
    return recovered_img

import os.path
import time

import torch
from torch.backends import cudnn
import numpy as np
from model import DiffusionUNet
import utils
import torch.nn as nn
from model.WEM import MyNet
from datetime import datetime


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


def noise_estimation_loss(model, x0, t, e, b):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start


    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):

        if isinstance(module, nn.DataParallel):
            module = module.module

        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module

        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        ### 检测模型是否为DataParallel类型，如果是，则读出模型
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.ch = config.model.ch
        self.MyNet = MyNet(3, self.ch)

        self.model = DiffusionUNet(config)

        self.model.to(self.device)

        self.model = torch.nn.DataParallel(self.model)
        num_params = count_parameters(self.model)
        print("parameters: {:,}".format(num_params))

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.load_save.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))


    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps

        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)

        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)

        if last:
            xs = xs[0][-1]

        return xs

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder)

        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break

            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x)

            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                utils.load_save.save_image(x_cond[i], os.path.join(image_folder, str(step), f"{i}_cond.png"))
                utils.load_save.save_image(x[i], os.path.join(image_folder, str(step), f"{i}.png"))

    def safe_save_checkpoint(self, state, save_dir, step, snapshot_freq):
        os.makedirs(save_dir, exist_ok=True)
        model_idx = (step // snapshot_freq) % 2 + 1
        temp_filename = os.path.join(save_dir, f'model{model_idx}.pth.tar.tmp')
        final_filename = os.path.join(save_dir, f'model{model_idx}.pth.tar')


        torch.save(state, temp_filename)

        os.replace(temp_filename, final_filename)

        print(f"Step {step}: Safely saved checkpoint as {final_filename}")

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            print(formatted_time)

            for i, (x, y) in enumerate(train_loader):

                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x

                n = x.size(0)


                data_time += time.time() - data_start

                self.model.train()

                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                #############################################################


                loss_df = noise_estimation_loss(self.model, x, t, e, b)
                loss = loss_df
                #############################################################

                if self.step % 10 == 0:
                    print(f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i + 1)}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0 or self.step == 1 or self.step == 200:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                if self.step % self.config.training.snapshot_freq == 0:
                    print("step:{}, save checkpoint safely...".format(self.step))

                    state = {
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }

                    save_dir = os.path.join(self.config.data.data_dir, 'ckpts')
                    snapshot_freq = self.config.training.snapshot_freq
                    self.safe_save_checkpoint(state, save_dir, self.step, snapshot_freq)


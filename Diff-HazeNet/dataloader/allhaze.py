import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL.Image
import re
import random
import PIL


class AllWeather:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='haze'):
        if validation == 'haze':
            print("=> evaluating haze test set...")
            path = os.path.join(self.config.data.data_dir, 'haze')
            filename = 'haze_test.txt'

        elif validation == 'dusty':
            print("=> evaluating dusty test set...")
            path = os.path.join(self.config.data.data_dir, 'dusty')
            filename = 'dusty_test.txt'

        elif validation == 'colorhaze':
            path = os.path.join(self.config.data.data_dir, 'colorhaze')
            filename = 'colorhaze_test.txt'

        elif validation == 'smoke':
            path = os.path.join(self.config.data.data_dir, 'smoke')
            filename = 'smoke_test.txt'

        else:   #
            print("=> evaluating haze test set...")
            path = os.path.join(self.config.data.data_dir, 'haze')
            filename = 'haze_test.txt'

        train_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir),
                                          n=self.config.training.patch_n,
                                          patch_size=self.config.data.image_size,
                                          transforms=self.transforms,
                                          filelist='train_list2.txt',
                                          parse_patches=parse_patches)
        val_dataset = AllWeatherDataset(path, n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=filename,
                                        parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True):
        super().__init__()

        self.dir = dir
        train_list = os.path.join(dir, filelist)
        print(train_list)
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input', 'gt') for i in input_names]
        print("len:{}".format(len(input_names)))


        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        # print(self.dir+input_name)
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id


    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

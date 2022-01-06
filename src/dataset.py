import os
import skimage
import numpy as np
from skimage.io import imread
from skimage.color import rgb2lab

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):
    def __init__(self, dir_dataset, image_size=320, train_set=True):
        self.dir_dataset = dir_dataset
        self.image_size = image_size
        self.list_images = sorted(os.listdir(self.dir_dataset))

        if train_set:
            self.transform = transforms.Compose([
                transforms.RESIZE((self.image_size, self.image_size), Image.BILINEAR),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RESIZE((self.image_size, self.image_size), Image.BILINEAR)
            ])

    def __len__(self):
        # return number of images
        return len(self.list_images)

    def __getitem__(self, idx):
        img_rgb = imread(os.path.join(self.dir_dataset, self.list_images[idx]))
        img_rgb = self.transform(img_rgb)

        # convert to numpy array
        img_rgb = np.array(img_rgb)

        # convert rgb to lab image
        img_lab = rgb2lab(img_rgb).astype(np.float32)
        img_lab = transforms.ToTensor()(img_lab)

        img_l = img_lab[[0], ...] / 50. - 1.
        # img_l belongs to [-1, 1]
        img_ab = img_lab[[1, 2]], ...] / 110.
        # img_ab belongs to [-1, 1]

        # return a dict of domain_1 and domain_2 images
        # domain_1 is l channel and
        # domain_2 is ab channel
        return {"domain_1": img_l, "domain_2": img_ab}

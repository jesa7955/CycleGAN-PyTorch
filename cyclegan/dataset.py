import os
import sys
import glob
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

available_datasets = ['horse2zebra', 'facades']
base_url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'

class CycleGANDataset(Dataset):
    """
    Class used to read images in
    """
    def __init__(self, data_root=f"{__file__}/../data", dataset_name='facades', transform=None, unaligned=True, mode='train'):

        # Check whether the specified dataset is available
        if dataset_name not in available_datasets:
            print(f"{dataset_name}という名前のデータセットが存在しないようです")
            sys.exit(1)
        # Check whether the dataset is downloaded
        base_dir = os.path.abspath(data_root)
        dataset_dir = os.path.join(base_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, True)
            print(f"{base_url}からデータセットをダウンロードし{base_dir}に解凍してください")
            sys.exit(1)

        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}/A/*.*")))
        self.files_B = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}/B/*.*")))

    def __getitem__(self, index):

        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B)-1)]))
        else:
            item_B = self.transform(Image.open(self.files_A[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

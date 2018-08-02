import os
import sys

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

available_datasets = ['horse2zebra']
base_url = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'

def get_data_loader(dataset_name, type_name, opts):
    """Creates training and test data loaders.
    """

    if dataset_name not in available_datasets:
        print(f"{dataset_name}という名前のデータセットが存在しないようです")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(f"{__file__}/../data"))
    dataset_dir = os.path.join(base_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        print(f"{base_url}からデータセットをダウンロードし{base_dir}に解凍してください")
        sys.exit(1)

    transform = transforms.Compose([
                    transforms.Scale(opts.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_path = os.path.join(base_dir, f"{dataset_name}/train{type_name}")
    test_path = os.path.join(base_dir, f"{dataset_name}/test{type_name}")

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    return train_dloader, test_dloader

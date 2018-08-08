import argparse
import os
import sys
import itertools
import math
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
from PIL import Image

from models import CycleGenerator, Discriminator
from lr_helpers import get_lambda_rule
from dataset import CycleGANDataset

def train_loop(opts):

    if opts.image_height == 128:
        res_blocks = 6
    elif opts.image_height >= 256:
        res_blocks = 9

    # Create networks
    G_AB = CycleGenerator(opts.a_channels, opts.b_channels, res_blocks).to(device)
    G_BA = CycleGenerator(opts.b_channels, opts.a_channels, res_blocks).to(device)
    D_A = Discriminator(opts.a_channels, opts.d_conv_dim).to(device)
    D_B = Discriminator(opts.b_channels, opts.d_conv_dim).to(device)

    # Print network architecture
    print("                 G_AtoB                ")
    print("---------------------------------------")
    print(G_AB)
    print("---------------------------------------")

    print("                 G_BtoA                ")
    print("---------------------------------------")
    print(G_BA)
    print("---------------------------------------")

    print("                  D_A                  ")
    print("---------------------------------------")
    print(D_A)
    print("---------------------------------------")

    print("                  D_B                  ")
    print("---------------------------------------")
    print(D_B)
    print("---------------------------------------")


    # Create losses
    criterion_gan = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    if opts.load:
        #TODO
        pass

    # Weights cycle loss and identity loss
    lambda_cycle = 10
    lambda_id = 0.5 * lambda_cycle

    # Create optimizers
    g_optimizer = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                    lr=opts.lr, betas=(opts.beta1, opts.beta2))
    d_a_optimizer = torch.optim.Adam(D_A.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))
    d_b_optimizer = torch.optim.Adam(D_B.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

    # Create learning rate update schedulers
    LambdaLR = get_lambda_rule(opts)
    g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=LambdaLR)
    d_a_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(d_a_optimizer, lr_lambda=LambdaLR)
    d_b_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(d_b_optimizer, lr_lambda=LambdaLR)

    # Image transformations
    transform = transforms.Compose([transforms.Resize(int(opts.image_height*1.12), Image.BICUBIC),
                                   transforms.RandomCrop((opts.image_height, opts.image_width)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    train_dataloader = DataLoader(CycleGANDataset(opts.dataroot_dir, opts.dataset_name, transform), batch_size=opts.batch_size, shuffle=True, num_workers=opts.n_cpu)
    test_dataloader = DataLoader(CycleGANDataset(opts.dataroot_dir, opts.dataset_name, transform, mode='test'), batch_size=5, shuffle=False, num_workers=1)


    end_epoch = opts.epochs + opts.start_epoch
    total_batch = len(train_dataloader) * opts.epochs
    for epoch in range(opts.start_epoch, end_epoch):
        for index, batch in enumerate(train_dataloader):
            # Create adversarial target
            real_A = Variable(batch['A'].to(device))
            real_B = Variable(batch['B'].to(device))
            fake_A, fake_B = G_BA(real_B), G_AB(real_A)

            # Train discriminator A
            d_a_optimizer.zero_grad()

            patch_real = D_A(real_A)
            loss_a_real = criterion_gan(patch_real, torch.tensor(1.0).expand_as(patch_real).to(device))
            patch_fake = D_A(fake_A)
            loss_a_fake = criterion_gan(patch_fake, torch.tensor(0.0).expand_as(patch_fake).to(device))
            loss_d_a = (loss_a_real + loss_a_fake) / 2
            loss_d_a.backward()
            d_a_optimizer.step()

            # Train discriminator B
            d_b_optimizer.zero_grad()

            patch_real = D_B(real_B)
            loss_b_real = criterion_gan(patch_real, torch.tensor(1.0).expand_as(patch_real).to(device))
            patch_fake = D_B(fake_B)
            loss_b_fake = criterion_gan(patch_fake, torch.tensor(0.0).expand_as(patch_fake).to(device))
            loss_d_b = (loss_b_real + loss_b_fake) / 2
            loss_d_b.backward()
            d_b_optimizer.step()

            # Train generator

            g_optimizer.zero_grad()
            fake_A, fake_B = G_BA(real_B), G_AB(real_A)
            reconstructed_A, reconstructed_B = G_BA(fake_B), G_AB(fake_A)
            # GAN loss
            patch_a = D_A(fake_A)
            loss_gan_ba = criterion_gan(patch_a, torch.tensor(1.0).expand_as(patch_a).to(device))
            patch_b = D_B(fake_B)
            loss_gan_ab = criterion_gan(patch_b, torch.tensor(1.0).expand_as(patch_b).to(device))
            loss_gan = (loss_gan_ab + loss_gan_ba) / 2

            # Cycle loss
            loss_cycle_a = criterion_cycle(reconstructed_A, real_A)
            loss_cycle_b = criterion_cycle(reconstructed_B, real_B)
            loss_cycle = (loss_cycle_a + loss_cycle_b) / 2

            # Identity loss
            loss_id_a = criterion_identity(G_BA(real_A), real_A)
            loss_id_b = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_a + loss_id_b) / 2

            # Total loss
            loss_g = loss_gan + lambda_cycle * loss_cycle + lambda_id * loss_identity
            loss_g.backward()
            g_optimizer.step()

            current_batch = epoch * len(train_dataloader) + index
            sys.stdout.write(f"\r[Epoch {epoch+1}/{opts.epochs-opts.start_epoch}] [Index {index}/{len(train_dataloader)}] [D_A loss: {loss_d_a.item():.4f}] [D_B loss: {loss_d_b.item():.4f}] [G loss: adv: {loss_gan.item():.4f}, cycle: {loss_cycle.item():.4f}, identity: {loss_identity.item():.4f}]")

            if current_batch % opts.sample_every == 0:
                save_sample(G_AB, G_BA, current_batch, opts, test_dataloader)

        # Update learning reate
        g_lr_scheduler.step()
        d_a_lr_scheduler.step()
        d_b_lr_scheduler.step()
        if epoch % opts.checkpoint_every == 0:
            torch.save(G_AB.state_dict(), '{opts.checkpoint_dir}/{opts.dataset_name}/G_AB_{epoch}.pth')
            torch.save(G_BA.state_dict(), '{opts.checkpoint_dir}/{opts.dataset_name}/G_BA_{epoch}.pth')
            torch.save(D_A.state_dict(), '{opts.checkpoint_dir}/{opts.dataset_name}/D_A_{epoch}.pth')
            torch.save(D_B.state_dict(), '{opts.checkpoint_dir}/{opts.dataset_name}/D_B_{epoch}.pth')

def save_sample(G_AB, G_BA, batch, opts, test_dataloader):
    images = next(iter(test_dataloader))
    real_A = Variable(images['A'].to(device))
    real_B = Variable(images['B'].to(device))
    fake_A = G_BA(real_B)
    fake_B = G_AB(real_A)
    image_sample = torch.cat((real_A.data, fake_B.data,
                              real_B.data, fake_A.data), 0)
    save_image(image_sample, f"{opts.sample_dir}/{opts.dataset_name}/{batch}.png", nrow=5, normalize=True)

def create_parser():

    parser = argparse.ArgumentParser()

    # モデル用ハイパーパラメータ
    parser.add_argument('--image_height', type=int, default=256, help='画像の高さ.')
    parser.add_argument('--image_width', type=int, default=256, help='画像の広さ.')
    parser.add_argument('--a_channels', type=int, default=3, help='A類画像のChannels数.')
    parser.add_argument('--b_channels', type=int, default=3, help='B類画像のChannels数.')
    parser.add_argument('--d_conv_dim', type=int, default=64)

    # トレーニング用ハイパーパラメータ
    parser.add_argument('--dataset_name', type=str, default='facades', help='使用するデータセット.')
    parser.add_argument('--epochs', type=int, default=200, help='Epochの数.')
    parser.add_argument('--start_epoch', type=int, default=0, help='実行開始のEpoch数.')
    parser.add_argument('--decay_epoch', type=int, default=100, help='lr decayを実行し始めるEpoch数.')
    parser.add_argument('--batch_size', type=int, default=1, help='一つのBatchに含まれる画像の数.')
    parser.add_argument('--num_workers', type=int, default=0, help='Dataloaderに使われるスレッド数.')
    parser.add_argument('--lr', type=float, default=0.0002, help='学習率(defaultは0.0002).')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adamオプチマイザーに使われるハイパーパラメータ.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adamオプチマイザーに使われるハイパーパラメータ.')
    parser.add_argument('--n_cpu', type=int, default=8, help='batchを生成するときに使用するスレッド数.')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用するGPUのID.')

    # サンプルやチェックポイントをとる頻度と場所
    parser.add_argument('--dataroot_dir', type=str, default='../data/')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='samples_cyclegan')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--log_step', type=int , default=20)
    parser.add_argument('--sample_every', type=int , default=100, help='サンプルをとる頻度、batch単位.')
    parser.add_argument('--checkpoint_every', type=int , default=1, help='Check pointをとる頻度、epoch単位.')
    return parser


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    device = torch.device(f'cuda:{opts.gpu_id}' if torch.cuda.is_available() else 'cpu')

    os.makedirs(f"{opts.sample_dir}/{opts.dataset_name}", exist_ok=True)
    os.makedirs(f"{opts.checkpoint_dir}/{opts.dataset_name}", exist_ok=True)

    if opts.load:
        opts.sample_dir = '{}_pretrained'.format(opts.sample_dir)
        opts.sample_every = 20

    print_opts(opts)
    train_loop(opts)

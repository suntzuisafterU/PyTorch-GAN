import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--photos-path", type=str, help="path of input photos to generate in style of Monet")
parser.add_argument("--models-path", type=str, help="path of models to load")
# parser.add_argument("--monet-path", type=str, help="path of input Monet images to ") # TODO: Also test the Monet to photo loop
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
opt = parser.parse_args()
print(opt)

import signal
import sys
stop_next_epoch_flag = False
def signal_handler(signum, frame):
    print('Caught ctrl-c')
    if not stop_next_epoch_flag:
        print('First time ctrl-c was pressed, setting flag')
        stop_next_epoch_flag = True
    else:
        print('Second time ctrl-c was pressed, exiting')
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)


def get_output_dir():
    in_steps = opt.photos_path.split('/')
    assert in_steps[0] == 'data'
    assert in_steps[-1] == 'B', 'For now we are only parsing the PyTorch-GAN-provided data, so want to make sure the photos are photos'
    in_steps[0] = 'out'
    return os.path.join(in_steps)









G_AB.load_state_dict(torch.load("%s/G_AB_%d.pth" % (opt.model_path, opt.epoch)))
G_BA.load_state_dict(torch.load("%s/G_BA_%d.pth" % (opt.model_path, opt.epoch)))
D_A.load_state_dict(torch.load("%s/D_A_%d.pth" % (opt.model_path, opt.epoch)))
D_B.load_state_dict(torch.load("%s/D_B_%d.pth" % (opt.model_path, opt.epoch)))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset(opt.photos_path, transforms_=transforms_, unaligned=True),
    batch_size=1,
    shuffle=True,
    num_workers=opt.n_cpu,
)


def sample_images():
    """Saves a generated sample from the test set"""
    for imgs in dataloader:
        # TODO: Need the image name
        # G_AB.eval()
        G_BA.eval()
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = G_BA(real_B)
        save_image(fake_A, "out/%s/%s.png" % (opt.dataset_name, ), normalize=False)
        # # Arange images along x-axis
        # real_A = make_grid(real_A, nrow=5, normalize=True)
        # real_B = make_grid(real_B, nrow=5, normalize=True)
        # fake_A = make_grid(fake_A, nrow=5, normalize=True)
        # fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        # image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        # save_image(image_grid, "out/%s/%s.png" % (opt.dataset_name, ), normalize=False)





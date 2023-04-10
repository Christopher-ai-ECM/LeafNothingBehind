import os
import numpy as np
from tqdm import tqdm

import torch

import parameter as PARAM

from loss import MSE
from model import UNet
from dataloader import create_generators


def test():

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = UNet(input_channels=3, output_classes=1, hidden_channels=PARAM.HIDDEN_CHANNELS, dropout_probability=PARAM.DROPOUT)
    model.to(device)

    # Loss
    criterion = MSE([8, 9, 11])

    model.eval()
    
import os
import numpy as np
from tqdm import tqdm

import torch

import src.parameter as PARAM

from src.loss import MSE
from src.model import UNet
from src.dataloader import create_generators

# Fix seed for reproducibility
seed = 123
torch.manual_seed(seed)


def test():
    
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    _, _, test_generator = create_generators()
    model = UNet(input_channels=3, output_classes=1, hidden_channels=PARAM.HIDDEN_CHANNELS, dropout_probability=PARAM.DROPOUT)

    model.to(device)
    checkpoint_path = os.path.join('checkpoint', 'poids_unet_bis.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Loss
    criterion = MSE([1, 2, 3, 6, 7, 8, 9, 11])

    ###############################################################
    # Start Evaluation                                            #
    ###############################################################

    model.eval()
    test_loss = []
    with torch.no_grad():
        for (image, target) in tqdm(test_generator, desc='test'):
            image = image.to(device).to(torch.float)
            y_true = target.to(device).to(torch.float)
            y_true, y_bis = y_true[:,0,:,:].to(torch.float).to(device), y_true[:,1,:,:].to(torch.float).to(device)
            y_pred = model(image)

            loss = criterion(y_pred, y_true, y_bis)
            test_loss.append(loss.item())

    print('final loss:', np.mean(test_loss))
    
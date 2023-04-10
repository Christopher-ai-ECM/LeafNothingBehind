import os
import numpy as np
import pickle

import torch

import parameter as PARAM

from model import UNet
from dataloader import create_predict_generateur


def test(csv_path, save_infers_under):

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = UNet(input_channels=3, output_classes=1, hidden_channels=PARAM.HIDDEN_CHANNELS, dropout_probability=PARAM.DROPOUT)
    model.to(device)
    checkpoint_path = os.path.join('checkpoint', 'UNET_50.pth')
    print('checkpoint_path:', checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    loader = create_predict_generateur(csv_path)
    number_of_batches = PARAM.BATCH_SIZE if len(loader) > PARAM.BATCH_SIZE > -1 else len(loader)
    results = {"outputs": [], "paths": []}

    for i, data in enumerate(loader):
        if i == number_of_batches:
            break
        results["paths"] += list(data["paths"][-1])
        results["outputs"].append(model(data).detach().cpu().numpy())
        if i % 10 == 9 or i+1 == number_of_batches:
            print(f"Performed batch {i+1}/{number_of_batches}")

    results["outputs"] = np.concatenate(results["outputs"], axis=0)
    with open(os.path.join(save_infers_under, "results.pickle"), 'wb') as file:
        pickle.dump(results, file)
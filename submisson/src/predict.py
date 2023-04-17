import os
import numpy as np
import pickle

import torch

import src.parameter as PARAM

from src.model import UNet
from src.dataloader import create_predict_generateur


def predict(csv_path, save_infers_under):
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = UNet(input_channels=3, output_classes=1, hidden_channels=PARAM.HIDDEN_CHANNELS, dropout_probability=PARAM.DROPOUT)
    model.to(device)
    checkpoint_path = os.path.join('checkpoint', 'UNET_50.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    loader = create_predict_generateur(csv_path)
    number_of_batches = len(loader)
    results = {"outputs": [], "paths": []}
    
    for i, data in enumerate(loader):
        if i == number_of_batches:
            break
        
        X, moy, image_name = data
        X = X.to(torch.float).to(device)
        moy = moy.to(torch.float).to(device)
        # results["paths"] += list(data["paths"][-1])
        results["paths"] += list(image_name)    # add path to the s2 image name
        predict = model(X)
        result = (predict + moy)[0]
        results["outputs"].append(result.detach().cpu().numpy())
        # if i % 10 == 9 or i + 1 == number_of_batches:
        #     print(f"Performed batch {i+1}/{number_of_batches}")

    results["outputs"] = np.expand_dims(np.concatenate(results["outputs"], axis=0), axis=-1)
    csv_name = os.path.basename(os.path.normpath(csv_path)).split('.')[0]
    save_path = os.path.join(save_infers_under, csv_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "results.pickle"), 'wb') as file:
        pickle.dump(results, file)
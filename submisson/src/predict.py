import os
import numpy as np
import pickle

import torch

import src.parameter as PARAM
from src.model import UNet
from src.dataloader import create_predict_generateur
from src.utils import de_normalize_s2, affiche_image


def predict(csv_path, save_infers_under):
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = UNet(input_channels=3, output_classes=1, hidden_channels=PARAM.HIDDEN_CHANNELS, dropout_probability=PARAM.DROPOUT)
    model.to(device)
    checkpoint_path = os.path.join('checkpoint', 'UNET_13_normalisation.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    loader = create_predict_generateur(csv_path)
    number_of_batches = len(loader)
    results = {"outputs": [], "paths": []}
    
    for i, data in enumerate(loader):
        if i == number_of_batches:
            break
        
        X, image_name = data
        X = X.to(torch.float).to(device)
        # print(image_name)
        # moy = moy.to(torch.float).to(device)
        # results["paths"] += list(data["paths"][-1])
        results["paths"] += list(image_name)    # add path to the s2 image name
        predict = model(X)
        # affiche_image(predict[0, 0, :, :].detach().cpu().numpy())
        # result = (predict + moy)[0]
        result = de_normalize_s2(predict[0].detach().cpu().numpy())
        # affiche_image(result[0])
        results["outputs"].append(result)

    results["outputs"] = np.expand_dims(np.concatenate(results["outputs"], axis=0), axis=-1)
    csv_name = os.path.basename(os.path.normpath(csv_path)).split('.')[0]
    save_path = os.path.join(save_infers_under, csv_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "results.pickle"), 'wb') as file:
        pickle.dump(results, file)
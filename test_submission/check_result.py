import pickle
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from skimage import io
from torch import nn


class MSE(nn.Module):
    def __init__(self, cloud_index):
        super(MSE, self).__init__()
        self.cloud_index = cloud_index

    def forward(self, S2_pred, S2_true, S2_mask):
        mask = torch.ones_like(S2_mask, dtype=torch.bool)
        for index in self.cloud_index:
            mask &= (S2_mask != index)
        
        S2_true_masked = S2_true[mask]
        S2_pred_masked = S2_pred.squeeze(1)[mask]
        
        loss = torch.mean(torch.pow((S2_true_masked - S2_pred_masked), 2))
        
        return loss


def read_pickle(result_path):
    result = []
    with (open(result_path, "rb")) as openfile:
        while True:
            try:
                result.append(pickle.load(openfile))
            except EOFError:
                break
    return result[0]


def check_attributes(result):
    if 'paths' in result and 'outputs' in result:
        print('il y a bien les bons attribues')
    else:
        print('erreur, il ny a pas les attribues paths ou outputs')
        exit()
    return 'paths' in result and 'outputs' in result


def check_shapes(result):
    outputs = result['outputs']
    paths = result['paths']
    bool = True
    try:

        if type(outputs) != np.ndarray:
            print('erreur, ce n est pas un numpy array')
            bool = False

        if outputs.dtype != 'float32':
            print('erreur, le dtype n est pas float32')
            bool = False

        shape = outputs.shape
        print('outputs shape:', shape)
        print('path length:', len(paths))
        if shape[0] != len(paths):
            print("erreur, nombre d'image dans le chemain != de celui de l'ouput")
            bool = False

        if shape[1] != 256 or shape[2] != 256:
            print("erreur, nombre d'image dans le chemain != de celui de l'ouput")
            bool = False
        
        if len(shape) != 4:
            print("erreur dans la shape de l'ouputs, format attendu: (nb_images, 256, 256, 1)")
            bool = False
        
        if shape[3] != 1:
            print("erreur, il faut que shape[3] = 1")
            bool = False
        
    except:
        print('une erreur est apparue, fin du programme')
        bool = False

    return bool


def load_image(path):
    image = io.imread(path)
    if len(image.shape) == 3:
        image = np.moveaxis(image, -1, 0)
    return torch.from_numpy(image)


def load_tiff(names, path):
    n = len(names)
    data = torch.empty((n, 256, 256))
    for i, name in enumerate(names):
        data[i] = load_image(os.path.join(path, name))
    return data


def test_result(result_path, data_path):
    result = read_pickle(result_path)
    if not(check_attributes(result)) or not(check_shapes(result)) :
        exit('erreur, le test ne va pas se lancer')
    else:
        print('resultat dans le bon format \n')
    
    outputs = result['outputs']
    names = result['paths']
    
    outputs = torch.from_numpy(outputs.squeeze(axis=-1))
    print(outputs.shape)

    s2_path = os.path.join(data_path, 's2')
    mask_path = os.path.join(data_path, 's2-mask')
    s2 = load_tiff(names, s2_path)
    mask = load_tiff(names, mask_path)

    loss = MSE([1, 2, 3, 6, 7, 8, 9, 11])
    loss_value = loss(outputs, s2, mask)
    print('test loss:', loss_value)
    

if __name__ == "__main__":
    result_path = 'result\\test_csv\\results.pickle'
    data_path = '..\\data'
    test_result(result_path, data_path)
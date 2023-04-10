import torch
import numpy as np
import torch.nn.functional as F


def find_previous(names, index):
    """
    prends un nom d'image avec un Measurement_id = 2
    revoie les noms des images associé avec un Measurement_id = 0 et 1
    """
    name_2 = names[index]
    name_0 = ''
    name_1 = ''
    for i, x in enumerate(name_2.split('-')):
        if i != 5:
            name_0 += x + '-'
            name_1 += x + '-'
        else:
            name_0 += '0-'
            name_1 += '1-'
    return name_0[:-1], name_1[:-1]


def normalize_s1(tensor):
    """
    Normalize les images de s1 (plage (-30,10)) ya yavais de 20 quasiment
    """
    return torch.clamp(tensor, -30, 0) / (-30)


def normalize_s2(tensor):
    """
    on supprime les valeurs abberantes de LAI pour qu'il soit compris entre 0 et 10
    puis on comprime sa distribution avec le log puis on normalize la distribution comprimée entre 0 et 1
    """
    return np.log(torch.clamp(tensor, 0, 10) + 1) / np.log(11) 


def moyenne(s2_0, s2_1, mask_0, mask_1):
    """
    fait une "moyenne" des images s2_0 (image de S2 au temps t-2), s2_1 (image de S2 au temps t-1)
    tel que pour chaque pixel, on fait la moyenne si y n'y a pas de nuage sur les 2 photos,
                               on prend seulement la valeur d'un pixel s'il dans il y un nuage dans l'autre
                               on fait la moyenne s'il des nuages dans les 2 photos
    renvoie le résultat dans une matrice numpy de taille (256, 256)
    """
    nuage_indice = [8, 9]
    
    # Check if any pixel in both masks is a cloud
    both_clouds = (np.isin(mask_0, nuage_indice)) & (np.isin(mask_1, nuage_indice))
    
    # Check if any pixel in one mask is a cloud
    one_cloud = np.isin(mask_0, nuage_indice) ^ np.isin(mask_1, nuage_indice)
    
    # Compute output array using NumPy array operations
    output = np.where(both_clouds, (s2_0 + s2_1) / 2, s2_1)
    output = np.where(one_cloud & ~both_clouds, s2_1, output)
    output = np.where(one_cloud & ~both_clouds, s2_0, output)
    output = np.where(~one_cloud & ~both_clouds, (s2_0 + s2_1) / 2, output)
    
    return output


def zscore_normalize(tensor):
    # calculer la moyenne et l'écart-type du tenseur
    mean = tensor.mean()
    std = tensor.std()

    # normaliser le tenseur
    tensor_normalized = (tensor - mean) / std

    return tensor_normalized


def image_similarity(image1, image2):
    # Calculer la différence entre les deux images
    diff = image1 - image2

    # Calculer le carré de chaque élément de la différence
    squared_diff = torch.pow(diff, 2)

    # Calculer la moyenne des carrés
    mse = torch.mean(squared_diff)

    # Convertir la MSE en une valeur scalaire
    mse = mse.item()

    # Calculer la similarité entre les deux images
    similarity = 1.0 / (1.0 + mse)

    return similarity
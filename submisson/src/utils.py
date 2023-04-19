import torch
import numpy as np
import matplotlib.pyplot as plt


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
    # return np.log(torch.clamp(tensor, 0, 10) + 1) / np.log(11) 
    return np.log(torch.clamp(tensor, -0.5, 12) + 1.5) / np.log(13.5)


def moyenne(s2_0, s2_1, mask_0, mask_1):
    """
    fait une "moyenne" des images s2_0 (image de S2 au temps t-2), s2_1 (image de S2 au temps t-1)
    tel que pour chaque pixel, on fait la moyenne si y n'y a pas de nuage sur les 2 photos,
                               on prend seulement la valeur d'un pixel s'il dans il y un nuage dans l'autre
                               on fait la moyenne s'il des nuages dans les 2 photos
    renvoie le résultat dans une matrice numpy de taille (256, 256)
    """
    nuage_indice = [1, 3, 4, 8, 9, 10, 11]
    
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


def de_normalize_s2(tensor):
    """
    fonction inverse de normalize_s2
    """
    # return np.exp(tensor * np.log(11)) - 1
    return np.exp(tensor * np.log(13.5)) - 1.5


def affiche_image(X):
    plt.imshow(X)
    plt.show()


def compte_clamp(tensor):
    compte = 0
    l = tensor.flatten()
    for x in l:
        if x < -1 or x > 10:
            print(x)
            compte += 1
    return compte
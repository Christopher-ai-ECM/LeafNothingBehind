import os
import tqdm
import torch
from osgeo import gdal
import numpy as np


np.random.seed(0)

S1_PATH = os.path.join('assignment-2023', 's1')
S2_PATH = os.path.join('assignment-2023', 's2')
MASK_PATH = os.path.join('assignment-2023', 's2-mask')


def get_name(txt_path):
    f = open(txt_path, "r")
    names = []
    for line in f:
        names.append(line[:-1])
    return names


ALREADY_SAVE = get_name('names_on_drive.txt')


def load(path):
    return torch.tensor(gdal.Open(path).ReadAsArray())


def check_data(name):
    # False -> ne pas rajouter
    path = os.path.join(MASK_PATH, name)
    mask = load(path)
    return not(0 in mask)


def check_name(name):
    # False -> ne pas rajouter
    name_0 = ''
    name_1 = ''
    for i, x in enumerate(name.split('-')):
        if i != 5:
            name_0 += x + '-'
            name_1 += x + '-'
        else:
            name_0 += '0-'
            name_1 += '1-'
    name_0, name_1 =  name_0[:-1], name_1[:-1]

    in_s1 = name in os.listdir(S1_PATH)
    in_s2_0 = name_0 in os.listdir(S2_PATH)
    in_s2_1 = name_1 in os.listdir(S2_PATH)
    in_s2_2 = name in os.listdir(S2_PATH)
    in_mask_0 = name_0 in os.listdir(MASK_PATH)
    in_mask_1 = name_1 in os.listdir(MASK_PATH)
    in_mask_2 = name in os.listdir(MASK_PATH)
    bool = in_s1 and in_s2_0 and in_s2_1 and in_s2_2 and in_mask_0 and in_mask_1 and in_mask_2
    return bool


def check_all(name):
    already = name in ALREADY_SAVE
    return check_name(name) and check_data(name) and not(already)


def main(save):
    names = os.listdir(S1_PATH)
    names_2 = list(filter(lambda x: x.split('-')[5] == '2', names))
    np.random.shuffle(names_2)
    f = open(save, "w") 
    for id in tqdm.tqdm(range(len(names_2))):
        name = names_2[id]
        add = check_all(name)
        if add:
            f.write(name + '\n')


if __name__ == '__main__':
    main('have_to_add.txt')

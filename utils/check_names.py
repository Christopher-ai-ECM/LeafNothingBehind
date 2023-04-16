"""
compare le nom des images de new_assigment-2023 avec le dossier image_series.csv
"""

import os
import numpy as np


def get_name_from_csv(csv_path):
    """
    récupère la liste des noms des images au temps t du csv
    """
    f = open(csv_path, "r")
    names = []
    for line in f:
        names.append(line.split(",")[-1][:-1])
    return names


def get_name_from_s2data(s2_path):
    """
    récupère la liste des noms des images s2 qui est dans le drive
    """
    names = os.listdir(s2_path)
    return names


def differance(csv_names, s2_names):
    """
    récupère les noms des images qui sont dans s2_names et qui ne sont pas dans csv_names
    """
    names = list(filter(lambda x: x not in csv_names, s2_names))
    return names


def intersection(csv_names, s2_names):
    """
    renvoie la liste des noms qui se trouve à la fois dans csv_name et dans s2_names
    """
    names = list(filter(lambda x: x in csv_names, s2_names))
    return names


def write(names, file):
    """
    écrit les noms des images (names) dans file
    """
    f = open(file, "w")
    for name in names:
        f.write(name + "\n")
    f.close()


def split_data(good_names):
    np.random.shuffle(good_names)
    n = len(good_names)
    split_1 = int(0.8 * n)
    split_2 = split_1 + int(0.1 * n)
    train, val, test = good_names[:split_1], good_names[split_1:split_2], good_names[split_2:]
    print(len(train), len(val), len(test))
    write(train, 'train.txt')
    write(val, 'val.txt')
    write(test, 'test.txt')


if __name__ == "__main__":
    csv_path = '..\\data\\image_series.csv'
    s2_path = '..\\new_assignment-2023\\s2'
    csv_names = get_name_from_csv(csv_path)
    s2_names = get_name_from_s2data(s2_path)
    good_names = intersection(csv_names, s2_names)
    split_data(good_names)
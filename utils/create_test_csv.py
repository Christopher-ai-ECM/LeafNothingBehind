"""
crée le csv_test contenant l'ensemble des images de tests
"""

import os


def read_test_file(file='test.txt'):
    f = open(file, 'r')
    names = [line[:-1] for line in f]
    return names


def find_previous(name_2):
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


def write_csv(csv_path, names):
    f = open(csv_path, "w")
    for name_2 in names:
        name_0, name_1 = find_previous(name_2)
        f.write(name_0 + ',')
        f.write(name_1 + ',')
        f.write(name_2 + '\n')


test_names = read_test_file()
csv_path = '..\\data\\test_csv.csv'
write_csv(csv_path, test_names)
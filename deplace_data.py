import shutil
import os
import tqdm


def get_name(txt_path):
    print(txt_path)
    f = open(txt_path, "r")
    names = []
    for line in f:
        names.append(line[:-1])
    return names


def find_previous(names, index):
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


def deplace(path_1, path_2, names):
    print('transfère des images de ', path_1, 'à', path_2)
    for name in tqdm.tqdm(names, desc='deplace data'):
        src = os.path.join(path_1, name)
        dst = os.path.join(path_2, name)
        shutil.copyfile(src, dst)
 
def other_names(names):
    names_0, names_1 = [], []
    for i in range(len(names)):
        name_0, name_1 = find_previous(names, i)
        names_0.append(name_0)
        names_1.append(name_1)
    return names_0, names_1


names_2 = get_name('names_to_save.txt')
names_0, names_1 = other_names(names_2)
names = names_0 + names_1
folder = 's2-mask'
path_1 = os.path.join('assignment-2023', folder)
path_2 = os.path.join('add', folder)
deplace(path_1, path_2, names)
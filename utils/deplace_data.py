import shutil
import os
import tqdm


def get_name(txt_path):
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
    """
    déplace la liste des fichies names de path_1 à path_2
    """
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


def main(src, dst, txt_file, nb=-1):
    """
    copie tous les fichiers de txt_file se trouvant dans src pour les collers dans le dossier dst
    nb == 2: copie seulement les images au temps t
    nb == 1: copie seulement les images au temps t-1 et t-2
    nb == 0: copie toutes les images
    """
    names_2 = get_name(txt_file)

    if nb == 2:
        print('copie des images au temps t')
        names = names_2

    elif nb == 1:
        print('copie des images au temps t-1 et t-2')
        names_0, names_1 = other_names(names_2)
        names = names_0 + names_1

    else:
        print('copie de toutes les images')
        names_0, names_1 = other_names(names_2)
        names = names_0 + names_1 + names_2

    print('copie de', src, 'et collage sur', dst)
    deplace(src, dst, names)


if __name__ == '__main__':
    folders = [('s1', 2), ('s2', 2), ('s2_01', 1), ('s2-mask', 2), ('s2-mask_01', 1)]
    txt_file = 'names_on_drive.txt'
    for folder, nb in folders:
        if '01' in folder:
            src = os.path.join('assignment-2023', folder[:-3])
        else:
            src = os.path.join('assignment-2023', folder)  
        dst = os.path.join('new_assignment-2023', folder)
        main(src, dst, txt_file, nb=nb)
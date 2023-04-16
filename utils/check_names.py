"""
compare le nom des images de new_assigment-2023 avec le dossier image_series.csv
"""

import os


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


def write_diff_names(diff_names, file_name):
    f = open(file_name, "w")
    for name in diff_names:
        f.write(name + "\n")
    f.close()


if __name__ == "__main__":
    csv_path = '..\\data\\image_series.csv'
    s2_path = '..\\new_assignment-2023\\s2'
    csv_names = get_name_from_csv(csv_path)
    s2_names = get_name_from_s2data(s2_path)
    diff_names = differance(csv_names, s2_names)
    print(diff_names)
    print(len(diff_names))
    write_diff_names(diff_names, "names_not_on_csv.txt")


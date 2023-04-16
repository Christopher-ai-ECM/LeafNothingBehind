import os
import shutil


def get_csv_names(csv_file):
    f = open(csv_file, 'r')
    names = []
    for line in f:
        if line[0] != '0':
            if line[-1] == '\n':
                name = line[:-1].split(',')
                names.append(name)
            else:
                name = line.split(',')
                names.append(name)
    return names


def copy(src, dst, names):
    print('copy:', src, dst)
    for name in names:
        if name not in os.listdir(dst):
            shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))


def main(src, dst, names):
    name_0 = list(map(lambda x: x[:][0], names))
    name_1 = list(map(lambda x: x[:][1], names))
    name_2 = list(map(lambda x: x[:][2], names))

    # copy s1 file
    print('main:', src, dst)
    copy(os.path.join(src, 's1'), os.path.join(dst, 's1'), name_0 + name_1 + name_2)

    # copy s2 file
    copy(os.path.join(src, 's2'), os.path.join(dst, 's2'), name_0 + name_1)

    # copy mask file
    copy(os.path.join(src, 's2-mask'), os.path.join(dst, 's2-mask'), name_0 + name_1)


if __name__ == "__main__":
    csv_file = '..\\test_submission\\test_data.csv'
    src = '..\\data'
    dst = '..\\test_submission'

    names = get_csv_names(csv_file)
    # print(names)
    main(src, dst, names)
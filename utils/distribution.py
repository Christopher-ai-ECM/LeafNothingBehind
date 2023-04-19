from skimage import io
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm



class Distribution():
    def __init__(self, _min, _max, pas) -> None:
        self.min = _min
        self.max = _max
        self.pas = pas
        self.n = int((_max - _min) / pas)
        self.data = [0 for _ in range(self.n)]
        self.nb_max = 0
        self.nb_min = 0
        self.indice = [i * self.pas + self.min for i in range(self.n)]
    
    def add_elt(self, x):
        if x > self.max:
            self.nb_max += 1
        elif self.min > x:
            self.min += 1
        else:
            index = int((x - self.min) / self.pas)
            self.data[index] += 1
    
    def add_vector(self, l):
        for x in l:
            self.add_elt(x)
    
    def __str__(self):
        return str(self.data)
    
    def plot(self):
        print('dépacement au min:', self.nb_min)
        print('dépacement au max:', self.nb_max)
        plt.hist(self.indice, weights=self.data, bins=self.n)
        plt.show()
    
    def get_data(self):
        return self.data

    def normalise(self):
        self.data = np.array(self.data)
        self.data = self.data / np.sum(self.data)
         

def get_data_names(txt_path):
    """
    prend un chemin vers un fichier txt et renvoie la liste des noms des images qui sont dedans
    """
    f = open(txt_path, "r")
    names = [line[:-1] for line in f]
    return names


def check_image(name, distribution):
    image_path = '..\\data\\s2\\' + name
    image = io.imread(image_path).flatten()
    distribution.add_vector(image)
    return distribution


if __name__ == '__main__':
    txt_path = 'train.txt'
    names = get_data_names(txt_path)
    distribution = Distribution(-1, 12, 0.1)
    for name in tqdm.tqdm(names):
        check_image(name, distribution)
    distribution.normalise()
    distribution.plot()
    print(distribution)
    

   
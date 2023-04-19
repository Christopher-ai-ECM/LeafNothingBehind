import os
import csv
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils import find_previous, normalize_s1, normalize_s2, moyenne
import src.parameter as PARAM

np.random.seed(0)


class DataGenerator(Dataset):
    def __init__(self, names):
        """
        names: liste des noms d'image qui ont un Measurement_id = 2
        """
        self.names = names
    
    def __len__(self):
        return len(self.names)
      
    def __getitem__(self, index):
        name_0, name_1 = find_previous(self.names, index)

        s2_0_path = os.path.join(PARAM.S2_01_PATH, name_0)
        s2_1_path = os.path.join(PARAM.S2_01_PATH, name_1)
        s2_2_path = os.path.join(PARAM.S2_PATH, self.names[index])

        s1_path = os.path.join(PARAM.S1_PATH, self.names[index])

        mask_0_path = os.path.join(PARAM.MASK_01_PATH, name_0)
        mask_1_path = os.path.join(PARAM.MASK_01_PATH, name_1)
        mask_2_path = os.path.join(PARAM.MASK_PATH, self.names[index])
        

        try:
          # mask_0 = torch.tensor(gdal.Open(mask_0_path).ReadAsArray())
          mask_0 = load_image(mask_0_path)
        except:
          mask_0 =  torch.zeros(256, 256)
        try:
          # mask_1 = torch.tensor(gdal.Open(mask_1_path).ReadAsArray())
          mask_1 = load_image(mask_1_path)
        except:
          mask_1 =  torch.zeros(256, 256)
        try:
          # mask_2 = torch.tensor(gdal.Open(mask_2_path).ReadAsArray())
          mask_2 = load_image(mask_2_path)
        except:
          mask_2 =  torch.zeros(256, 256)

        s1 = load_image(s1_path)
        s2_0 = load_image(s2_0_path)
        s2_1 = load_image(s2_1_path)
        s2_2 = load_image(s2_2_path)

        moy = normalize_s2(torch.tensor(moyenne(s2_0, s2_1, mask_0, mask_1))).unsqueeze(0)
        s1 = normalize_s1(s1.clone().detach()) 
        s2_2 = normalize_s2(s2_2.clone().detach().unsqueeze(0)) 
        s2_1 = normalize_s2(s2_1.clone().detach().unsqueeze(0)) 
        # difference = s2_2 - moy

        merged_tensor = torch.cat((moy, s1), dim=0)
        
        X = merged_tensor #shape(3,256,256)
        # Y = torch.cat((difference, mask_2.unsqueeze(0)),dim=0) #s2_2 avant 
        Y = torch.cat((s2_2, mask_2.unsqueeze(0)),dim=0) #s2_2 avant 
        return X, Y


class Predict_DataGenerator(Dataset):
    def __init__(self, names, path):
        """
        names[i] = [nom de l'image i au temps 0, nom de l'image i au temps 1, nom de l'image i au temps 2]
        """
        self.names = names
        self.path = path
        self.s1 = os.path.join(path, 's1')
        self.s2 = os.path.join(path, 's2')
        self.mask = os.path.join(path, 's2-mask')
    
    def __len__(self):
        return len(self.names)
      
    def __getitem__(self, index):
        name_0, name_1, name_2 = self.names[index]

        s2_0_path = os.path.join(self.s2, name_0)
        s2_1_path = os.path.join(self.s2, name_1)

        s1_path = os.path.join(self.s1, name_2)

        mask_0_path = os.path.join(self.mask, name_0)
        mask_1_path = os.path.join(self.mask, name_1)
        
        try:
          mask_0 = load_image(mask_0_path)
        except:
          mask_0 =  torch.zeros(256, 256)
        try:
          mask_1 = load_image(mask_1_path)
        except:
          mask_1 =  torch.zeros(256, 256)
       
        # s1 = torch.tensor(gdal.Open(s1_path).ReadAsArray())
        # s2_0 = torch.tensor(gdal.Open(s2_0_path).ReadAsArray())
        # s2_1 = torch.tensor(gdal.Open(s2_1_path).ReadAsArray())
        s1 = load_image(s1_path)
        s2_0 = load_image(s2_0_path)
        s2_1 = load_image(s2_1_path)

        moy = normalize_s2(torch.tensor(moyenne(s2_0, s2_1, mask_0, mask_1))).unsqueeze(0)
        s1=normalize_s1(s1.clone().detach()) 
        # s2_2=normalize_s2(s2_2.clone().detach().unsqueeze(0)) 
        s2_1=normalize_s2(s2_1.clone().detach().unsqueeze(0)) 
        # difference = s2_2 - moy

        merged_tensor = torch.cat((moy, s1), dim=0)
        #print("shape_mask: ", np.shape(mask_1))
        
        X = merged_tensor #shape(3,256,256)
        # Y = torch.cat((difference, mask_2.unsqueeze(0)),dim=0) #s2_2 avant 

        s2_name = self.names[index][-1]
        return X, s2_name


# def data_split(names):
#     n = len(names)
#     split_1 = int(PARAM.TRAIN_SPLIT * n)
#     split_2 = split_1 + int(PARAM.VAL_SPLIT * n)
#     return names[:split_1], names[split_1: split_2], names[split_2:]


def get_data_names(txt_path):
  """
  prend un chemin vers un fichier txt et renvoie la liste des noms des images qui sont dedans
  """
  f = open(txt_path, "r")
  names = [line[:-1] for line in f]
  return names


def create_generators():
    # names = os.listdir(PARAM.S2_PATH)
    # train_data, val_data, test_data = data_split(names)
    train_data = get_data_names(os.path.join(PARAM.TXT_PATH, 'train.txt'))
    val_data = get_data_names(os.path.join(PARAM.TXT_PATH, 'val.txt'))
    test_data = get_data_names(os.path.join(PARAM.TXT_PATH, 'test.txt'))
    train_loader = DataLoader(DataGenerator(train_data), batch_size=PARAM.BATCH_SIZE, shuffle=PARAM.SHUFFLE_DATA, drop_last=True)
    val_loader = DataLoader(DataGenerator(val_data), batch_size=PARAM.BATCH_SIZE, shuffle=PARAM.SHUFFLE_DATA, drop_last=True)
    test_loader = DataLoader(DataGenerator(test_data), batch_size=PARAM.BATCH_SIZE, shuffle=PARAM.SHUFFLE_DATA, drop_last=True)
    return train_loader, val_loader, test_loader


def read_csv(csv_path):
    with open(csv_path) as path_list:
      data_paths = [[os.path.basename(path) for path in row] for row in csv.reader(path_list, delimiter=",")]
    return data_paths


def create_predict_generateur(csv_path):
    names = read_csv(csv_path)
    data_path = os.path.dirname(csv_path)
    predict = DataLoader(Predict_DataGenerator(names, data_path), batch_size=1, shuffle=False, drop_last=False)
    return predict


def load_image(path):
    image = io.imread(path)
    if len(image.shape) == 3:
        image = np.moveaxis(image, -1, 0)
    return torch.from_numpy(image)


def check():
  _, _, test = create_generators()
  X1, _ = next(iter(test))
    
  pred = create_predict_generateur('..\\data\\test_csv.csv')
  X2 = next(iter(pred))

  print(X1[0])
  print(X2)

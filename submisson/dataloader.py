import os
import numpy as np
from osgeo import gdal
import torch
from torch.utils.data import Dataset, DataLoader

from utils import find_previous, normalize_s1, normalize_s2, moyenne
import parameter as PARAM

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
          mask_0 = torch.tensor(gdal.Open(mask_0_path).ReadAsArray())
        except:
          mask_0 =  torch.zeros(256, 256)
        try:
          mask_1 = torch.tensor(gdal.Open(mask_1_path).ReadAsArray())
        except:
          mask_1 =  torch.zeros(256, 256)
        try:
          mask_2 = torch.tensor(gdal.Open(mask_2_path).ReadAsArray())
        except:
          mask_2 =  torch.zeros(256, 256)

       
        s1 = torch.tensor(gdal.Open(s1_path).ReadAsArray())
        s2_0 = torch.tensor(gdal.Open(s2_0_path).ReadAsArray())
        s2_1 = torch.tensor(gdal.Open(s2_1_path).ReadAsArray())
        s2_2 = torch.tensor(gdal.Open(s2_2_path).ReadAsArray())


        moy = normalize_s2(torch.tensor(moyenne(s2_0, s2_1, mask_0, mask_1))).unsqueeze(0)
        s1=normalize_s1(s1.clone().detach()) 
        s2_2=normalize_s2(s2_2.clone().detach().unsqueeze(0)) 
        s2_1=normalize_s2(s2_1.clone().detach().unsqueeze(0)) 
        difference=s2_2-moy

        merged_tensor = torch.cat((moy, s1), dim=0)
        #print("shape_mask: ", np.shape(mask_1))
        
        X = merged_tensor #shape(3,256,256)
        Y = torch.cat((difference,mask_2.unsqueeze(0)),dim=0) #s2_2 avant 
        return X, Y


class Predict_DataGenerator(Dataset):
    def __init__(self, names):
        """
        names[i] = [nom de l'image i au temps 0, nom de l'image i au temps 1, nom de l'image i au temps 2]
        """
        self.names = names
    
    def __len__(self):
        return len(self.names)
      
    def __getitem__(self, index):
        S1_PATH = os.path.join('data', 's1')
        S2_PATH = os.path.join('data', 's2')
        MASK_PATH = os.path.join('data', 's2-mask')

        name_0, name_1, name_2 = self.names[index]

        s2_0_path = os.path.join(S2_PATH, name_0)
        s2_1_path = os.path.join(S2_PATH, name_1)

        s1_path = os.path.join(S1_PATH, name_2[:-1])

        mask_0_path = os.path.join(MASK_PATH, name_0)
        mask_1_path = os.path.join(MASK_PATH, name_1)
        
        try:
          mask_0 = torch.tensor(gdal.Open(mask_0_path).ReadAsArray())
        except:
          mask_0 =  torch.zeros(256, 256)
        try:
          mask_1 = torch.tensor(gdal.Open(mask_1_path).ReadAsArray())
        except:
          mask_1 =  torch.zeros(256, 256)
       
        s1 = torch.tensor(gdal.Open(s1_path).ReadAsArray())
        s2_0 = torch.tensor(gdal.Open(s2_0_path).ReadAsArray())
        s2_1 = torch.tensor(gdal.Open(s2_1_path).ReadAsArray())

        moy = normalize_s2(torch.tensor(moyenne(s2_0, s2_1, mask_0, mask_1))).unsqueeze(0)
        s1=normalize_s1(s1.clone().detach()) 
        # s2_2=normalize_s2(s2_2.clone().detach().unsqueeze(0)) 
        s2_1=normalize_s2(s2_1.clone().detach().unsqueeze(0)) 
        # difference=s2_2-moy

        merged_tensor = torch.cat((moy, s1), dim=0)
        #print("shape_mask: ", np.shape(mask_1))
        
        X = merged_tensor #shape(3,256,256)
        # Y = torch.cat((difference, mask_2.unsqueeze(0)),dim=0) #s2_2 avant 
        Y = moy
        return X, Y


def data_split(names):
    n = len(names)
    split_1 = int(PARAM.TRAIN_SPLIT * n)
    split_2 = split_1 + int(PARAM.VAL_SPLIT * n)
    return names[:split_1], names[split_1: split_2], names[split_2:]


def create_generators():
    names = os.listdir(PARAM.S2_PATH)
    train_data, val_data, test_data = data_split(names)
    train_loader = DataLoader(DataGenerator(train_data), batch_size=PARAM.BATCH_SIZE, shuffle=PARAM.SHUFFLE_DATA, drop_last=True)
    val_loader = DataLoader(DataGenerator(val_data), batch_size=PARAM.BATCH_SIZE, shuffle=PARAM.SHUFFLE_DATA, drop_last=True)
    test_loader = DataLoader(DataGenerator(test_data), batch_size=PARAM.BATCH_SIZE, shuffle=PARAM.SHUFFLE_DATA, drop_last=True)
    return train_loader, val_loader, test_loader


def read_csv(csv_path):
    names = []
    f = open(csv_path, "r")
    for line in f:
        names.append(line.split(','))
    return names[1:]


def create_predict_generateur(csv_path):
    names = read_csv(csv_path)
    print(names)
    predict = DataLoader(Predict_DataGenerator(names), batch_size=1, shuffle=False, drop_last=False)
    return predict


if __name__ == "__main__":
    # train, _, _ = create_generators()
    # X, Y = next(iter(train))
    # print(X)

    pred = create_predict_generateur('..\\data\\image_series.csv')
    X, Y = next(iter(pred))
    print(X)
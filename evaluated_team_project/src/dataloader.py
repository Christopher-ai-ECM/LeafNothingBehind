import os
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

       
        # s1 = torch.tensor(gdal.Open(s1_path).ReadAsArray())
        # s2_0 = torch.tensor(gdal.Open(s2_0_path).ReadAsArray())
        # s2_1 = torch.tensor(gdal.Open(s2_1_path).ReadAsArray())
        # s2_2 = torch.tensor(gdal.Open(s2_2_path).ReadAsArray())
        # print('gdal')
        # print(f'{s1.shape = }')
        # print(f'{s2_0.shape = }')
        # print(f'{s2_1.shape = }')
        # print(f'{s2_2.shape = }')
        # print(s1[0,1])

        s1 = load_image(s1_path)
        s2_0 = load_image(s2_0_path)
        s2_1 = load_image(s2_1_path)
        s2_2 = load_image(s2_2_path)
        # print('io')
        # print(f'{s1.shape = }')
        # print(f'{s2_0.shape = }')
        # print(f'{s2_1.shape = }')
        # print(f'{s2_2.shape = }')

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
        # difference=s2_2-moy

        merged_tensor = torch.cat((moy, s1), dim=0)
        #print("shape_mask: ", np.shape(mask_1))
        
        X = merged_tensor #shape(3,256,256)
        # Y = torch.cat((difference, mask_2.unsqueeze(0)),dim=0) #s2_2 avant 
        Y = moy
        s2_name = self.names[index][-1]
        return X, Y, s2_name


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
    data_path = os.path.dirname(csv_path)
    predict = DataLoader(Predict_DataGenerator(names, data_path), batch_size=1, shuffle=False, drop_last=False)
    return predict


def load_image(path):
    image = io.imread(path)
    if len(image.shape) == 3:
        image = np.moveaxis(image, -1, 0)
    return torch.from_numpy(image)


if __name__ == "__main__":
    train, _, _ = create_generators()
    X, Y = next(iter(train))
    print(X.shape)

    # pred = create_predict_generateur('..\\test_submission\\data\\test_data.csv')
    # X, Y = next(iter(pred))
    # print(X)
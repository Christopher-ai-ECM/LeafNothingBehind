import os

PATH = '..'         # 'gdrive/MyDrive/Compétition_IA'
S1_PATH = os.path.join(PATH, 'assignment-2023/s1')
S2_PATH = os.path.join(PATH, 'assignment-2023/s2')
S2_01_PATH = os.path.join(PATH, 'assignment-2023/s2_01')
MASK_PATH = os.path.join(PATH, 'assignment-2023/s2-mask')
MASK_01_PATH = os.path.join(PATH, 'assignment-2023/s2-mask_01')
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SHUFFLE_DATA = True

NB_DATA = 900        # nombre d'image utilisé (nombre image de S2 avec un Measurement_id = 2)

BATCH_SIZE = 16
VIRTUAL_BATCH_SIZE = 16  
SHUFFLE_DATA = True
HIDDEN_CHANNELS = 16 #16 marche bien
DROPOUT = 0.0
LEARNING_RATE = 0.0001
NB_EPOCHS = 50
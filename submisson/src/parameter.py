import os

PATH = os.path.join('..', 'new_assignment-2023')
S1_PATH = os.path.join(PATH, 's1')
S2_PATH = os.path.join(PATH, 's2')
S2_01_PATH = os.path.join(PATH, 's2_01')
MASK_PATH = os.path.join(PATH, 's2-mask')
MASK_01_PATH = os.path.join(PATH, 's2-mask_01')

TXT_PATH = os.path.join('..', 'utils')

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

BATCH_SIZE = 16
VIRTUAL_BATCH_SIZE = 16  
SHUFFLE_DATA = True

HIDDEN_CHANNELS = 16
DROPOUT = 0.0
LEARNING_RATE = 0.0001
NB_EPOCHS = 50
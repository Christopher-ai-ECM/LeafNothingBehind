import tensorflow as tf
#import tensorlayer as tl
#from tensorlayer.layers import *
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Input, Dropout, concatenate, UpSampling3D, Softmax 
import numpy as np
from variable import *

CONV=8


# le github: https://github.com/zhixuhao/unet/blob/master/model.py

def u_net(input_size):
    inputs = Input(input_size,batch_size=1)

    print('inputs:',np.shape(inputs))
    
    conv1 = Conv3D(CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv3D(CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)
    print('--- block 1 ---')
    print('CONV:', CONV)
    print('conv1:', np.shape(conv1))
    print('pool1:',np.shape(pool1))

    conv2 = Conv3D(2*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv3D(2*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)
    print('--- block 2 ---')
    print('CONV:', 2 * CONV)
    print('conv2:', np.shape(conv2))
    print('pool2:',np.shape(pool2))

    conv3 = Conv3D(4*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv3D(4*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)
    print('--- block 3 ---')
    print('CONV:', 4 * CONV)
    print('conv3:', np.shape(conv3))
    print('pool3:',np.shape(pool3))

    conv4 = Conv3D(8*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv3D(8*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(drop4)
    print('--- block 4 ---')
    print('CONV:', 8 * CONV)
    print('conv4:', np.shape(conv4))
    print('pool4:',np.shape(pool4))

    conv5 = Conv3D(16*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv3D(16*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)
    print('--- block 5 ---')
    print('CONV:', 16 * CONV)
    print('conv4:', np.shape(conv5))


    up6 = Conv3D(8*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2, 2, 1))(conv5)) #dans le code d'origine on prends drop5
    merge6 = concatenate([conv4,up6], axis = 4)
    conv6 = Conv3D(8*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(8*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    print('--- block 6 ---')
    print('CONV:', 8 * CONV)
    print('up+conv6:',np.shape(up6))
    print('merge6(conv4, up6):', np.shape(merge6))
    print('conv4:', np.shape(conv4))
    

    up7 = Conv3D(4*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2, 2, 1))(conv6))
    merge7 = concatenate([conv3,up7], axis = 4)
    conv7 = Conv3D(4*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(4*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    print('--- block 7 ---')
    print('CONV:', 4 * CONV)
    print('up+conv7:', np.shape(up7))
    print('merge7(conv3, up7):', np.shape(merge7))
    print('conv7:', np.shape(conv7))

    up8 = Conv3D(2*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2, 2, 1))(conv7))
    merge8 = concatenate([conv2,up8], axis = 4)
    conv8 = Conv3D(2*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv3D(2*CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    print('--- block 8 ---')
    print('CONV:', 2 * CONV)
    print('up+conv8:',np.shape(up8))
    print('merge8(conv2, up8):', np.shape(merge8))
    print('conv8:', np.shape(conv8))

    up9 = Conv3D(CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling3D(size = (2, 2, 1))(conv8))
    merge9 = concatenate([conv1,up9], axis = 4)
    conv9 = Conv3D(CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv3D(CONV, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = Conv3D(2, (3,3,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv3D(4, 1, activation = 'softmax')(conv9)
    print('--- block 9 ---')
    print('CONV:', CONV)
    print('up+conv9:',np.shape(up9))
    print('merge9(conv1, up9):', np.shape(merge9))
    print('conv9:', np.shape(conv9))
    print('conv10:', np.shape(conv10))

    print('--- resume ---')
    print('entree:', np.shape(inputs))
    print('sortie:', np.shape(conv10))

    model = tf.keras.Model(inputs = inputs, outputs = conv10)

    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    # if(pretrained_weights):
    # 	model.load_weights(pretrained_weights)

    return model


if __name__ == "__main__":
    u_net((208,192,8,1))

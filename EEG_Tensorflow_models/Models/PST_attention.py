from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow import Variable
import tensorflow as tf
from .TF_blocks import Channel_attention,Time_attention,Attention,square,log


def PST_attention(nb_classes=4, Chans = 22, Samples = 128, dropoutRate = 0.5,last_layer = 'Conv'):

    bias_spatial = False
    pool         = (1,75)
    strid        = (1,15)
    filters      = (1,25)


    input_main   = Input((Chans, Samples, 1))
    block1       = Attention(num_ch=Chans,num_t=Samples)(input_main)
    block1       = Conv2D(40, filters, 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = Conv2D(40, (Chans, 1), use_bias=bias_spatial, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=pool, strides=strid)(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)

    if last_layer=='Conv':
        ConvC    = Conv2D(nb_classes, (1, block1.shape[2]),kernel_constraint = max_norm(0.5, axis=(0,1,2)),name='ouput')(block1)
        flat     = Flatten(name='F_1')(ConvC)
        softmax  = Activation('softmax',name='A_out')(flat)

    elif last_layer=='Dense':
        flatten  = Flatten(name='F_1')(block1)
        dense    = Dense(nb_classes, kernel_constraint = max_norm(0.5),name='output')(flatten)
        softmax  = Activation('softmax',name='A_out')(dense)

    
    return Model(inputs=input_main, outputs=softmax)

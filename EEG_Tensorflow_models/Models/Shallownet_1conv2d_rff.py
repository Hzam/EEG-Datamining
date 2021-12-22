import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l1_l2



def Shallownet_1conv2d_rff(nb_classes, Chans = 22, Samples = 250, dropoutRate = 0.2, l1 = 0, l2 = 0):

    filters      = (1,40)
    kernel_time  = 40
    strid        = (1,5)
    pool         = (1,75)
    bias_spatial = False

    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(20, (Chans, kernel_time),strides=(1,2), use_bias=bias_spatial, kernel_regularizer=l1_l2(l1=l1,l2=l2),
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1       = tf.math.cos(block1)
    #Act1         = Activation('elu')(block1)
    block1       = AveragePooling2D(pool_size=pool, strides=strid)(block1)
    block1       = Dropout(dropoutRate,name='bottleneck')(block1)
    ConvC        = Conv2D(nb_classes, (1, block1.shape[2]),kernel_regularizer=l1_l2(l1=l1,l2=l2),kernel_constraint = max_norm(0.5, axis=(0,1,2)),name='ouput')(block1)
    flat         = Flatten(name='F_1')(ConvC)
    softmax      = Activation('softmax',name='Classif')(flat)
    return Model(inputs=input_main, outputs=softmax)

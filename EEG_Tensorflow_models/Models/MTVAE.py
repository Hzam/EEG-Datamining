from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten, Reshape
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf

class reparametrize(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        eta = tf.random.normal(tf.shape(log_var))
        sigma = tf.math.exp(log_var / 2)
        return  mean + sigma * eta

def MTVAE(nb_classes, Chans = 22, Samples = 250, dropoutRate = 0.5, l1 = 0, l2 = 0):

    filters      = (1,40)
    strid        = (1,15)
    pool         = (1,75)
    bias_spatial = True

    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(40, filters, strides=(1,2),
                                 input_shape=(Chans, Samples, 1),kernel_regularizer=l1_l2(l1=l1,l2=l2),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=bias_spatial, kernel_regularizer=l1_l2(l1=l1,l2=l2),
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    Act1         = Activation('elu')(block1)
    block1       = AveragePooling2D(pool_size=pool, strides=strid)(Act1)
    block1       = Dropout(dropoutRate,name='bottleneck')(block1)

    mu           = Dense(40,name='mu')(block1)
    log_var      = Dense(40,name='log_var')(block1)
    codings      = reparametrize(name='Code')([mu, log_var])

    ConvC        = Conv2D(nb_classes, (1, block1.shape[2]),kernel_regularizer=l1_l2(l1=l1,l2=l2),kernel_constraint = max_norm(0.5, axis=(0,1,2)),name='ouput')(block1)
    flat          = Flatten(name='F_1')(ConvC)
    softmax      = Activation('softmax',name='Classif')(flat)

    block2       = Conv2DTranspose(40, pool,strides=strid,activation='tanh', kernel_regularizer=l1_l2(l1=l1,l2=l2),
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(codings)
    block2       = Resizing(block2.shape[1], Act1.shape[2])(block2)
    block2       = Conv2DTranspose(40, (Chans, 1), use_bias=bias_spatial, kernel_regularizer=l1_l2(l1=l1,l2=l2),
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block2       = Conv2DTranspose(1, filters,strides=(1,2),
                                 input_shape=(Chans, Samples, 1),kernel_regularizer=l1_l2(l1=l1,l2=l2),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    
    model = Model(inputs=input_main, outputs=[block2,softmax])

    var_flat      = Flatten()(log_var)
    mu_flat       = Flatten()(mu)
    
    KL = -0.5 * tf.keras.backend.sum( 1 + var_flat - tf.keras.backend.exp(var_flat) - tf.keras.backend.square(mu_flat),axis=-1)
    model.add_loss(tf.keras.backend.mean(KL)/var_flat.shape[-1])#Chans*Samples)
    return model

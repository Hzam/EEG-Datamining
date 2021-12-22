from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Conv1D,Conv2D, AveragePooling2D,SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout, Add, Lambda,DepthwiseConv2D,Input, Permute, Reshape
from tensorflow.keras.constraints import max_norm
from tensorflow import Variable
from tensorflow.keras import backend as K
import tensorflow as tf


def TCN_block(input_layer,input_dimension,depth,kernel_size,filters,dropout,activation='elu'):
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=1,activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if(input_dimension != filters):
        conv = Conv1D(filters,kernel_size=1,padding='same')(input_layer)
        added = Add()([block,conv])
    else:
        added = Add()([block,input_layer])
    out = Activation(activation)(added)
    
    for i in range(depth-1):
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(filters,kernel_size=kernel_size,dilation_rate=2**(i+1),activation='linear',
                   padding = 'causal',kernel_initializer='he_uniform')(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)
        
    return out

class Channel_attention(Model):
    def __init__(self,num_ch):
        super(Channel_attention,self).__init__(name='')
        self.conv1      = Conv2D(filters=8,kernel_size=(1,1),kernel_constraint = max_norm(2., axis=(0,1,2)))
        self.conv2      = Conv2D(filters=8,kernel_size=(1,1),kernel_constraint = max_norm(2., axis=(0,1,2)))
        self.reshape    = Reshape((num_ch,-1))
        self.gamma      = Variable(initial_value=0,dtype='float32',trainable=True)

    def call(self,x):
        proj_query      = self.reshape(self.conv1(x))
        proj_key        = self.reshape(self.conv2(x))

        energy          = tf.matmul(proj_query,proj_key,transpose_b=True)

        max_H           = tf.math.reduce_max(energy,axis=2,keepdims=True)
        min_H           = tf.math.reduce_min(energy,axis=2,keepdims=True)

        tmp_b           = (energy-min_H)
        tmp_c           = (max_H-min_H)+1e-8
        energy          = tmp_b/tmp_c
        attention       = tf.nn.softmax(energy)

        out = self.gamma * tf.expand_dims(tf.matmul(attention,tf.squeeze(x,axis=-1),transpose_a=True),axis=3) + x

        return out

class Time_attention(Model):
    def __init__(self,num_t):
        super(Time_attention,self).__init__(name='')
        self.conv1      = Conv2D(filters=8,kernel_size=(1,1),kernel_constraint = max_norm(2., axis=(0,1,2)))
        self.conv2      = Conv2D(filters=8,kernel_size=(1,1),kernel_constraint = max_norm(2., axis=(0,1,2)))
        self.reshape    = Reshape((num_t,-1))
        self.permute    = Permute((2,1,3))
        self.gamma      = Variable(initial_value=0,dtype='float32',trainable=True)

    def call(self,x):
        b,ch,t,d        = x.shape
        proj_query      = self.reshape(self.permute(self.conv1(x)))
        proj_key        = self.reshape(self.permute(self.conv2(x)))

        energy          = tf.matmul(proj_query,proj_key,transpose_b=True)

        max_H           = tf.math.reduce_max(energy,axis=2,keepdims=True)
        min_H           = tf.math.reduce_min(energy,axis=2,keepdims=True)

        tmp_b           = (energy-min_H)
        tmp_c           = (max_H-min_H)+1e-8
        energy          = tmp_b/tmp_c
        attention       = tf.nn.softmax(energy)

        out = self.gamma * tf.expand_dims(tf.matmul(tf.squeeze(x,axis=-1),attention,transpose_b=True),axis=3) + x

        return out

class Attention(Model):
    def __init__(self,num_ch,num_t):
        super(Attention,self).__init__()
        self.CA = Channel_attention(num_ch=num_ch)
        self.TA = Time_attention(num_t=num_t)

    def call(self,x,training=None):
        out1 = self.CA(x)
        out2 = self.TA(x)

        out = tf.concat([x,out1,out2],axis=-1)
        
        return out



def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 

from .TF_blocks import TCN_block
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Conv1D,Conv2D, AveragePooling2D,SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, concatenate, Flatten
from tensorflow.keras.layers import Dropout, Add, Lambda,DepthwiseConv2D,Input, Permute
from tensorflow.keras.constraints import max_norm


def TCNet_fusion(nb_classes, Chans=64, Samples=128, layers=3, kernel_s=10,filt=10, dropout=0, activation='relu', F1=4, D=2, kernLength=64, dropout_eeg=0.1):
    
    input1 = Input((Chans, Samples, 1),name='Input')
    input2 = Permute((2,1,3))(input1)
    regRate=.25
    numFilters = F1
    F2= numFilters*D
    
    block1 = Conv2D(F1, (kernLength, 1), padding = 'same',data_format='channels_last',use_bias = False)(input2)
    block1 = BatchNormalization(axis = -1)(block1)
    block2 = DepthwiseConv2D((1, Chans), use_bias = False, 
                                    depth_multiplier = D,
                                    data_format='channels_last',
                                    depthwise_constraint = max_norm(1.))(block1)
    block2 = BatchNormalization(axis = -1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((8,1),data_format='channels_last')(block2)
    block2 = Dropout(dropout)(block2)
    block3 = SeparableConv2D(F2, (16, 1),
                            data_format='channels_last',
                            use_bias = False, padding = 'same')(block2)
    block3 = BatchNormalization(axis = -1)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D((8,1),data_format='channels_last')(block3)
    block3 = Dropout(dropout)(block3)    
    block2 = Lambda(lambda x: x[:,:,-1,:])(block3)
    outs = TCN_block(input_layer=block2,input_dimension=F2,depth=layers,kernel_size=kernel_s,filters=filt,dropout=dropout,activation=activation)
    CON1 = concatenate([outs,block2])
    FC1=Flatten()(block2)
    FC2=Flatten()(CON1)
    CON1 = concatenate([FC1,FC2])
    dense        = Dense(nb_classes, name = 'dense',kernel_constraint = max_norm(regRate), activation='softmax')(CON1)
    # softmax      = Activation('softmax', name = 'softmax')(dense)
    return Model(inputs=input1, outputs=dense)

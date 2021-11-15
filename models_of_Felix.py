from tensorflow import keras
from tensorflow.keras import layers, utils, backend


class ACRNN:

    def __init__(self, input_shape, class_num):
        self.input_shape = input_shape
        self.class_num = class_num

    def build_model(self):
        inputs = keras.Input(self.input_shape, name='InputLayer')

        # channel attention model
        channel_attention_in = layers.Permute((1, 4, 3, 2))(inputs) # (batchsize, slices, 1, 400, 22)
        channel_attention_out = self.channel_attention(channel_attention_in)(channel_attention_in)

        # # CNN model
        cnn_model_in = layers.Permute((1, 4, 3, 2))(channel_attention_out)
        cnn_model_out = self.CNN_model(cnn_model_in)(cnn_model_in)

        # RNN model
        rnn_model_out = self.RNN_model(cnn_model_out)(cnn_model_out)
        print(rnn_model_out.shape)

        # self-attention model
        attention_out = self.customed_self_attention(rnn_model_out)(rnn_model_out)


        # softmax layer
        # classification_in = Flatten()(self_attention_model_out)
        classification_in = layers.Dense(16, activation='relu')(attention_out)
        outputs = layers.Dense(self.class_num, activation='softmax', name='classification_Layer')(classification_in)

        # build
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        utils.plot_model(model=model, to_file='./model.png', show_shapes=True)
        return model

    def CNN_model(self, input_feature):
        # input_shape should be liked np.array([fs*window, lead_channel, 1])
        print(input_feature.shape)
        lead_channel = input_feature.shape[2]
        inputs = keras.Input(shape=input_feature.shape[1:], name='CNN_InputLayer')
        x = layers.Conv2D(filters=40, kernel_size=(lead_channel, 45), strides=1, name='CNN_Conv2DLayer_1')(inputs)
        print(x.shape)
        x = layers. MaxPool3D(pool_size=(1, 75, 1), strides=(1, 10, 1), data_format='channels_first')(x)
        print(x.shape)
        print(x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4])
        x = layers.Reshape((x.shape[1], x.shape[2] * x.shape[3] * x.shape[4]))(x)
        x = layers.Dropout(0.5, noise_shape=(x.shape[0], 1, x.shape[2]))(x)
        print(x.shape)
        #x = layers.BatchNormalization(axis=-1)(x)

        model = keras.Model(inputs=inputs, outputs=x, name='CNN_model')
        return model

    def channel_attention(self, input_feature, ratio=2):
        # 注意修改pooling和multiply，这边用BCIⅣ的数据来跑，输入为（None, 10, 1, 400, 22）
        # 对channe维度做attention
        channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]

        shared_layer_one = layers.Dense(channel // ratio,
                                        kernel_initializer='he_normal',
                                        activation='relu',
                                        use_bias=True,
                                        bias_initializer='zeros')

        shared_layer_two = layers.Dense(channel,
                                        kernel_initializer='he_normal',
                                        use_bias=True,
                                        bias_initializer='zeros')

        inputs = keras.Input(shape=input_feature.shape[1:], name='channelAttention_InputLayer')
        avg_pool = layers.GlobalAveragePooling3D()(inputs)#layers.GlobalAveragePooling2D()(inputs)
        # avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
        #assert avg_pool.shape[1:] == (1, 1, channel)
        avg_pool = shared_layer_one(avg_pool)
        #assert avg_pool.shape[1:] == (1, 1, channel // ratio)
        avg_pool = shared_layer_two(avg_pool)
        # assert avg_pool.shape[1:] == (1, 1, channel)

        max_pool = layers.GlobalMaxPooling3D()(inputs)# layers.GlobalMaxPooling2D()(inputs)
        # max_pool = layers.Reshape((1, 1, channel))(max_pool)
        #assert max_pool.shape[1:] == (1, 1, channel)
        max_pool = shared_layer_one(max_pool)
        #assert max_pool.shape[1:] == (1, 1, channel // ratio)
        max_pool = shared_layer_two(max_pool)
        #assert max_pool.shape[1:] == (1, 1, channel)

        cbam_feature = layers.Add()([avg_pool, max_pool])
        cbam_feature = layers.Activation('hard_sigmoid')(cbam_feature)

        if backend.image_data_format() == "channels_first":
            cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)

        multiply = layers.multiply([inputs, cbam_feature])
        model = keras.Model(inputs=inputs, outputs=multiply, name='channelAttention_model')
        utils.plot_model(model=model, to_file='./channel_attention_model.png', show_shapes=True)
        return model

    def spatial_attention(self, input_feature):
        kernel_size = 7
        if backend.image_data_format() == "channels_first":
            channel = input_feature.shape[1]
            cbam_feature = layers.Permute((2, 3, 1))(input_feature)
        else:
            channel = input_feature.shape[-1]
            cbam_feature = input_feature

        avg_pool = layers.Lambda(lambda x: backend.mean(x, axis=3, keepdims=True))(cbam_feature)
        assert avg_pool.shape[-1] == 1
        max_pool = layers.Lambda(lambda x: backend.max(x, axis=3, keepdims=True))(cbam_feature)
        assert max_pool.shape[-1] == 1
        concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
        assert concat.shape[-1] == 2
        cbam_feature = layers.Conv2D(filters=1,
                                     kernel_size=kernel_size,
                                     activation='hard_sigmoid',
                                     strides=1,
                                     padding='same',
                                     kernel_initializer='he_normal',
                                     use_bias=False)(concat)
        assert cbam_feature.shape[-1] == 1

        if backend.image_data_format() == "channels_first":
            cbam_feature = layers.Permute((3, 1, 2))(cbam_feature)

        return layers.multiply([input_feature, cbam_feature])

    def RNN_model(self, input_feature):
        # input_shape should be liked np.array([time_step, word_vec])
        inputs = layers.Input(shape=input_feature.shape[1:], name='RNN_InputLayer')
        x = layers.LSTM(64, return_sequences=True)(inputs)
        #x = layers.BatchNormalization(axis=-1)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        #x = layers.BatchNormalization(axis=-1)(x)

        model = keras.Model(inputs=inputs, outputs=x, name='RNN_model')
        # model.summary()
        # plot_model(model=model, to_file='./RNN_model.png', show_shapes=True)

        return model

    def customed_self_attention(self, input_feature):
        '''
        :desc:
            你是否还在将LSTM最后一层的return_sequences=False才获取时序特征？你是否还苦恼于效果不太好！！？
            原因在于，对于复杂的时序特征，LSTM很难注意到最重要的点key-point！比如对于sample1，key_point在1s，对于sample2，key_point在10s，这种现象EEG数据中非常常见
            欢迎使用我的customed_self_attention模块对你的LSTM结果进行权重标识
            写在前面：我们可以认为query是环境，key是数据特征，v是待分配权重的数据。通常k=v。
            我们将LSTM的最后一层输出(batch_size, hidden_size)作为query，代表这段数据的整体特征，然后key是(batch_size, time_steps, hidden_size)
        ,获取RNN、LSTM的many-many
        :param input_feature: (batch_size, time_steps, hidden_size)
        :return:
        '''

        hidden_states = layers.Input(shape=input_feature.shape[1:], name='self_attention_layer')
        hidden_size = input_feature.shape[2]
        # Inside dense layer
        #              hidden_states            dot               W            =>           key
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        key = layers.Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        query = layers.Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = layers.Dot(axes=[1, 2], name='attention_score')([query, key])
        attention_weights = layers.Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        attention_out = layers.Dot(axes=[1, 1], name='context_vector')([hidden_states, attention_weights])
        attention_out = layers.Concatenate(name='attention_output')([attention_out, query])
        model = keras.Model(inputs=hidden_states, outputs=attention_out, name='self_attention_model')
        utils.plot_model(model=model, to_file='./LSTM_self_attention_model.png', show_shapes=True)
        return model
        # attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)

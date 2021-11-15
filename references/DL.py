#! /usr/bin/python3
import numpy as np
import pandas as pd
import tensorflow as tf
from cnn_class import cnn
import time
import os
import scipy.io as sio
from sklearn.metrics import classification_report, roc_auc_score, auc, roc_curve, f1_score
from RnnAttention.attention import attention
from scipy import interp
from sne import t_sne2

# 无改进的四类散点图


# ROC曲线越靠近左上角,试验的准确性就越高。最靠近左上角的ROC曲线的点是错误最少的最好阈值，其假阳性和假阴性的总数最少
# 靠近左上角的ROC曲线所代表的受试者工作最准确！！！
# 通过分别计算各个试验的ROC曲线下的面积(AUC)进行比较，哪一种试验的 AUC最大，则哪一种试验的价值最佳
def multiclass_roc_auc_score(y_true, y_score):
    assert y_true.shape == y_score.shape  # y_score:模型预测值；
    fpr = dict()  # false positive rate:假正例率为横轴.FPR = FP /（FP + TN） （被预测为正的负样本结果数 /负样本实际数）
    tpr = dict()  # true positive rate: 真正例率为纵轴.TPR = TP /（TP + FN） （正样本预测结果数 / 正样本实际数）
    roc_auc = dict()  # ROC曲线下的面积AUC（Area Under Roc_curve）
    n_classes = y_true.shape[1]  # （n,4）一共四类



    # 对每一类计算ROC曲线和ROC曲线面积
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])  # 计算ROC曲线
        roc_auc[i] = auc(fpr[i], tpr[i])  # 计算ROC曲线面积
    # compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())  # ravel将数组维度拉成一维
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))  # concatenate()进行数组拼接，默认第axis=0维
    # 然后在这些点上插入ROC曲线 Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)  # 生成一个和all_fpr.shape同样形状的0矩阵
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return roc_auc


###########################################################################
# prepare raw data
###########################################################################
subject_id = 1  # 标识哪一个被试作为测试集
data_folder = os.path.abspath("..")
data = sio.loadmat(data_folder + "\BCI_data\cross_sub2a\cross_subject_data_" + str(subject_id) + ".mat")
print("subject id ", subject_id)

test_X = data["test_x"]  # [trials, channels, time length]: (576, 22, 1125)
train_X = data["train_x"]  # 4608 × 22 × 1125：切片数 * 通道数 * 时间点

test_y = data["test_y"].ravel()
train_y1 = data["train_y"].ravel()

train_y = np.asarray(pd.get_dummies(train_y1), dtype=np.int8)  # 进行one-hot编码 （4608,4）
test_y = np.asarray(pd.get_dummies(test_y), dtype=np.int8)  # （576,4）

print("one-hot编码后train_y的形状:", train_y.shape)
print("one-hot编码后test_y的形状:", test_y.shape)
print("--------------------------------------------")

###########################################################################
# crop data
###########################################################################

window_size = 400
step = 50
n_channel = 22


# 滑动窗口【0,400】，【50,450】，【100,500】,etc
def windows(data, size, step):
    start = 0
    while (start + size) < data.shape[0]:
        yield int(start), int(start + size)
        start += step


# 在一个trial中进行滑动窗口切片，并连接
def segment_signal_without_transition(data, window_size, step):
    segments = []
    for (start, end) in windows(data, window_size, step):
        if len(data[start:end]) == window_size:
            segments = segments + [data[start:end]]  # 将一个trial的切片进行连接
    return np.array(segments)


# 将数据集X中的各个trial进行滑动窗口切片，并连接
def segment_dataset(X, window_size, step):
    win_x = []
    for i in range(X.shape[0]):
        win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]  # 将X[i]trial的切片进行连接
    win_x = np.array(win_x)
    return win_x


# 各个trials的通道和时间序列，随着trials数的增加，堆叠
# train_X:[trials,channels, time length] 将1轴和2轴调换[trials, time length, channel]
train_raw_x = np.transpose(train_X, [0, 2, 1])  # （4608，1125,22）
test_raw_x = np.transpose(test_X, [0, 2, 1])  # （576,1125,22）


# np.transpose处理之后的数据，train_raw_x：（4608,1125,22）共4608个trial
train_win_x = segment_dataset(train_raw_x, window_size, step)  # 原始数据集转化成”切片“的数据集
print("train_win_x shape: ", train_win_x.shape)  # (4608,15,400,22) ：4608个trial，每个trial 15个时间片，窗口大小400，22个通道
test_win_x = segment_dataset(test_raw_x, window_size, step)
print("test_win_x shape: ", test_win_x.shape)  # （576,15,400,22）：576个trial，每个trail 15个时间片，窗口大小400，22个通道

# [trial, slices, channel, window_size]
train_win_x = np.transpose(train_win_x, [0, 1, 3, 2])  # （4608,15,22,400）
print("train_win_x shape: ", train_win_x.shape)

test_win_x = np.transpose(test_win_x, [0, 1, 3, 2])  # （576,15,22,400）
print("test_win_x shape: ", test_win_x.shape)

# [trial, slices, channel, window_size, 1]
train_x = np.expand_dims(train_win_x, axis=4)  # (4608,15,22,400,1)
test_x = np.expand_dims(test_win_x, axis=4)  # (576, 15,22,400,1)

num_time_step = train_x.shape[1]  # 每个trial共15个切片
print("num_time_step：", num_time_step)  # 15

###########################################################################
# set model parameters
###########################################################################
# kernel parameter
kernel_height_1st = 22
kernel_width_1st = 45

kernel_stride = 1

conv_channel_num = 40

# pooling parameter
pooling_height_1st = 1
pooling_width_1st = 70

pooling_stride_1st = 10

# full connected parameter
attention_size = 512
n_hidden_state = 64

###########################################################################
# set dataset parameters
###########################################################################
# input channel
input_channel_num = 1

# # # # # # # # # ## # # # # # # # # #
#    train_x: （4608,15,22,400,1）
# # # # # # # # # ## # # # # # # # # #
# input height 
input_height = train_x.shape[2]  # 22

# input width
input_width = train_x.shape[3]  # 400

# prediction class
num_labels = 4
###########################################################################
# set training parameters
###########################################################################
# set learning rate
learning_rate = 1e-4

# set maximum training epochs
training_epochs = 401

# set batch size
batch_size = 32

# set dropout probability
dropout_prob = 0.5

# set train batch number per epoch  train_x:（4608，15，22，400，1）
batch_num_per_epoch = train_x.shape[0] // batch_size  # 4608 // 10 = 460，一共46个batch
print("每个epoch的batch数量：", batch_num_per_epoch)

# instance cnn class
padding = 'VALID'

cnn_2d = cnn(padding=padding)  # 初始化一个CNN类

# input placeholder # NHWC：输入[None, 22, 400, 1] None：表示第0维不定
X = tf.compat.v1.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_labels], name='Y')  # 输入类别：[None, 4] None：表示第0维不定
train_phase = tf.compat.v1.placeholder(tf.bool, name='train_phase')
keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')  # keep_prob：保留的概率

# first CNN layer
# Conv2D(Input[N,H,W,Channel], filters[k_h,k_w,in_channel,out_channel],strides,padding)
conv_1 = cnn_2d.apply_conv2d(X, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride,
                             train_phase)  # kernel: [22,45,1]，输出40个卷积通道, 步进为：1
print("conv 1 shape: ", conv_1.get_shape().as_list())  # [None, 1, 356, 40] 输入层卷积之后：【356 = (400-45)/1) + 1】
pool_1 = cnn_2d.apply_max_pooling(conv_1, pooling_height_1st, pooling_width_1st,
                                  pooling_stride_1st)  # pooling: [1, 75, 10]
print("pool 1 shape: ", pool_1.get_shape().as_list())  # [None, 1, 29,40]  卷积层池化之后：【29 = (356-75)/10 + 1】

pool1_shape = pool_1.get_shape().as_list()  # [None, 1, 29, 40]
pool1_flat = tf.reshape(pool_1, [-1, pool1_shape[1] * pool1_shape[2] * pool1_shape[3]])
print("pool1_flat shape：", pool1_flat.get_shape().as_list())  # [None, 1160] # 1160是一个slice提取出的特征数

fc_drop = tf.nn.dropout(pool1_flat, keep_prob)
print("fc_drop shape: ", fc_drop.get_shape().as_list())  # [None, 1160]

lstm_in = tf.reshape(fc_drop, [-1, num_time_step, pool1_shape[1] * pool1_shape[2] * pool1_shape[3]])
print("lstm_in shape: ", lstm_in.get_shape().as_list())  # [None, 15, 1160]	# LSTM输入的是，一个trial 15个slice × 1160特征

########################## RNN ########################
cells = []
for _ in range(2):
    # n_hidden_state: 神经元个数；forget_bias：LSTM门的忘记系数，为1表示不忘记任何信息；state_is_tuple：表示返回的状态用一个元组表示
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_state, forget_bias=1.0, state_is_tuple=True)  # n_hidden_state = 64
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)  # 每个神经单元在每次有数据流入时，以一定的概率keep_prob正常工作，否则输出0值
    cells.append(cell)
lstm_cell = tf.contrib.rnn.MultiRNNCell(cells)

init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # state_is_tuple返回状态的初始化函数

# --------------------------------------------------------------------------------------------------------------#
# time_major如果是True，就表示RNN的steps用第一个维度表示，output ==> [steps, batch_size, depth]
# 如果是False, 第二个维度表示steps, output ==> [batch_size, steps, depth]

# 这里time_major = False, 故 output ==> [batch, step, n_hidden_state]
# rnn_op：就是output, states：为lstm输出的最终状态，包含c和h，都是[batch_size, n_hidden];
rnn_op, states = tf.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state=init_state,
                                   time_major=False)  # lstm_in: [None,15,1160]s
print("rnn_op形状：", rnn_op)  # [batch_size, step, n_hidden] 即[10, 15, 64]
print("-----------------")
print("states中c和h的形状：", states)  # c和h：[batch_size, n_hidden] 即[10, 64]

# --------------------------------------------------------------------------------------------------------------#

########################## attention ########################
with tf.name_scope('Attention_layer'):  # rnn_op: [batch, step, n_hidden_state]：[10,15,64]
    attention_op, alphas = attention(rnn_op, attention_size, time_major=False, return_alphas=True)
# attention_op: (10,64), alphas: (10,15), attention_size=512

attention_drop = tf.nn.dropout(attention_op, keep_prob)  # (10,64)

########################## readout ######################## rnn_op: [batch, step, n_hidden_state]: [10, 15, 64]
y_ = cnn_2d.apply_readout(attention_drop, rnn_op.shape[2].value, num_labels)  # [10, 4]
print("y_ shape：", y_.get_shape().as_list())

# probability prediction 概率预测
y_prob = tf.nn.softmax(y_, name="y_prob")  # [10, 4]
print("y_prob shape：", y_prob.get_shape().as_list())

# class prediction 根据概率判断类
y_pred = tf.argmax(y_prob, 1, name="y_pred")  # [10]：10个batch_size，10个类
print("y_pred shape：", y_pred.get_shape().as_list())

########################## loss and optimizer ########################
# cross entropy cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name='loss')  # Y:[None, 4]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作
with tf.control_dependencies(update_ops):  # 定义函数依赖，执行完update_ops之后，才执行optimizer
    # set training SGD optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))  # 得到True和False的向量

########################## define accuracy ########################
# calculate prediction accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')  # 类型转换，将True和False转化为1.0和0.0，并求平均值

###########################################################################
# train test and save result
###########################################################################

# run with gpu memory growth
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

train_acc = []
test_acc = []
train_loss = []
with tf.compat.v1.Session(config=config) as session:
    session.run(tf.compat.v1.global_variables_initializer())
    best_acc = 0
    for epoch in range(training_epochs):  # 共 200 epoch
        pred_test = np.array([])  # 预测的类标签列表
        true_test = []  # 测试集中真实的标签
        prob_test = []  # 测试的softmax概率
        ########################## training process ########################
        for b in range(batch_num_per_epoch):  # 共460
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)  # train_y:[4608,4]

            batch_x = train_x[offset:(offset + batch_size), :, :, :,
                      :]  # train_x:（4608，15，22，400，1）;batch_x:(10,15,22,400,1)
            batch_x = batch_x.reshape(
                [len(batch_x) * num_time_step, n_channel, window_size, 1])  # batch_x: [10*15, 22, 400,1]

            batch_y = train_y[offset:(offset + batch_size), :]  # train_y: [4608, 4]

            _, c = session.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1 - dropout_prob,
                                                             train_phase: True})  # batch_x:(10*15,22,400,1)
        # ------------------------------------------------------------
        # calculate train and test accuracy after each training epoch
        # ------------------------------------------------------------
        if epoch % 1 == 0:
            train_accuracy = np.zeros(shape=[0], dtype=float)
            test_accuracy = np.zeros(shape=[0], dtype=float)
            train_l = np.zeros(shape=[0], dtype=float)
            test_l = np.zeros(shape=[0], dtype=float)
            # ------------------------------------------------------------
            # calculate 【 train accuracy 】 after each training epoch
            # ------------------------------------------------------------
            for i in range(batch_num_per_epoch):  # 共460
                ########################## prepare training data ########################
                offset = (i * batch_size) % (train_y.shape[0] - batch_size)
                train_batch_x = train_x[offset:(offset + batch_size), :, :, :]
                train_batch_x = train_batch_x.reshape([len(train_batch_x) * num_time_step, n_channel, window_size, 1])
                train_batch_y = train_y[offset:(offset + batch_size), :]

                ########################## calculate training results ########################
                train_a, train_c, fi= session.run([accuracy, cost, attention_op],
                                               feed_dict={X: train_batch_x, Y: train_batch_y, keep_prob: 1.0,
                                                          train_phase: False})

                train_l = np.append(train_l, train_c)  # 损失值
                train_accuracy = np.append(train_accuracy, train_a)  # 准确率

                ########################## 画图 ########################
                fi = np.array(fi).squeeze()
                if i == 0:
                    feature1 = fi
                else:
                    feature1 = np.concatenate((feature1, fi), axis=0)
            if (epoch >=220) and (epoch % 10 == 0):
                # 画出T-SNE降维图
                t_sne2(feature1, train_y1, feature1.shape[0], feature1.shape[1])
                # ------------------------------------END---------------------------------



            if epoch % 20 == 0:
                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Training Cost: ",
                    np.mean(train_l), "Training Accuracy: ", np.mean(train_accuracy))

            # ------------------------------------------------------------
            for j in range(batch_num_per_epoch):  # 共460
                ########################## prepare test data ########################
                offset = (j * batch_size) % (test_y.shape[0] - batch_size)
                test_batch_x = test_x[offset:(offset + batch_size), :, :, :]
                test_batch_x = test_batch_x.reshape([len(test_batch_x) * num_time_step, n_channel, window_size, 1])
                test_batch_y = test_y[offset:(offset + batch_size), :]

                ########################## 计算测试结果 ########################
                test_a, test_c, prob_v, pred_v = session.run([accuracy, cost, y_prob, y_pred],
                                                             feed_dict={X: test_batch_x, Y: test_batch_y,
                                                                        keep_prob: 1.0, train_phase: False})

                test_accuracy = np.append(test_accuracy, test_a)  # 测试集的准确率
                test_l = np.append(test_l, test_c)  # 测试集的loss值
                pred_test = np.append(pred_test, pred_v)  # 预测的类标签列表
                true_test.append(test_batch_y)  # 测试集中真实的标签
                prob_test.append(prob_v)  # 测试的softmax概率

            true_test = np.array(true_test).reshape([-1, num_labels])
            prob_test = np.array(prob_test).reshape([-1, num_labels])
            auc_roc_test = multiclass_roc_auc_score(y_true=true_test, y_score=prob_test)
            f1 = f1_score(y_true=np.argmax(true_test, axis=1), y_pred=pred_test, average='macro')
            if epoch % 20 == 0:
                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, "Test Cost: ",
                      np.mean(test_l),
                      "Test Accuracy: ", np.mean(test_accuracy),
                      "Test f1: ", f1,
                      "Test AUC: ", auc_roc_test['macro'],
                      "Test best_acc:", best_acc, "\n")

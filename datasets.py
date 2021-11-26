# -*- coding:utf-8 -*-
# @Time : 2021/11/13 13:34
# @Author: ShuhuiLin
# @File : datasets.py

import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from scipy.io import loadmat
from preprocessedPipeline import preprocessPipeline


class KaggleEDFMASD():

    def __init__(self):
        None

    def loadData(self, path, format='3D'):
        '''
        :param path: local path of dataset
        :param format:
            '3D' -- X with shape (n, H, W, C) -- y in one-hot format --  for CNN input;
            '2D' -- X with shape  (n, time_step, vector_num) -- y in one-hot format -- for RNN input;
            '1D' -- X with shape (n, m) -- y -- for SVM input;
        :return: Xtrain, Xtest, ytrain, ytest
        '''
        with open(path, 'rb') as f:
            data = pickle.load(f)
        x1 = data['trial_1']['focussed']
        x2 = data['trial_1']['unfocussed']
        x3 = data['trial_1']['drowsed']
        for i in range(2, 6):
            x1 = np.concatenate((x1, data[f'trial_{i}']['focussed']), axis=0)
            x2 = np.concatenate((x2, data[f'trial_{i}']['unfocussed']), axis=0)
            x3 = np.concatenate((x3, data[f'trial_{i}']['drowsed']), axis=0)
        size = len(x1)
        X = np.concatenate((x1, x2, x3), axis=0)
        y = np.concatenate((np.array([2 for _ in range(size)], dtype=int),
                            np.array([1 for _ in range(size)], dtype=int),
                            np.array([0 for _ in range(size)], dtype=int))
                           )
        per = np.random.permutation(X.shape[0])  # 打乱后的行号
        X = X[per, :, :]  # 获取打乱后的训练数据
        y = y[per]

        if format == '1D':
            X = X.reshape(np.shape(X)[0], np.shape(X)[1] * np.shape(X)[2])
            y = y.reshape(len(y))
        elif format == '2D':
            y = y.reshape(len(y), 1)
            y = tf.keras.utils.to_categorical(y, 3).astype('int8')
        elif format == '3D':
            y = y.reshape(len(y), 1)
            y = tf.keras.utils.to_categorical(y, 3).astype('int8')
            X = X.reshape(np.shape(X)[0], np.shape(X)[1], np.shape(X)[2], 1)
        else:
            print('Invalid Param: format, which shoule be choosed in ["1D","2D","3D"]')
            return None

        # 分割训练集，测试集
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15, random_state=33)
        return Xtrain, Xtest, ytrain, ytest

    def prepossessing(self, n_subjects, inp_dir='../../EEG Data/',
                      outp_dir='../../EEG Data/Disposed/withoutFeatureExtration', windows=2):
        '''
        :param n_subjects:
        :param inp_dir:
        :param outp_dir:
        :param windows:
        :return:
        '''
        fs = 128
        mkpt1 = int(fs * 10 * 60)
        mkpt2 = int(fs * 20 * 60)

        subject_map = {}
        for s in range(1, n_subjects + 1):
            a = int(7 * (s - 1)) + 3
            if s != 5:
                b = a + 5
            else:
                b = a + 4
            subject_map[s] = [i for i in range(a, b)]

        channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        useful_channels = ['F7', 'F3', 'P7', 'O1', 'O2', 'P8', 'AF4']
        # useful_channels = ['F7','F3','T7','P7','O1','O2','P8','T8','AF4']
        use_channel_inds = []
        for c in useful_channels:
            if c in channels:
                use_channel_inds.append(channels.index(c))

        for s in range(1, n_subjects + 1):
            data = {}
            data['channels'] = useful_channels
            data['fs'] = fs
            for i, t in enumerate(subject_map[s]):
                trial = {}
                trial_data = loadmat(inp_dir + f'eeg_record{t}.mat')
                eeg = np.transpose(trial_data['o']['data'][0][0][:, 3:17])
                eeg = eeg[use_channel_inds, :fs * 60 * 30]
                print(eeg.shape)
                eeg = preprocessPipeline(data=eeg, fs=fs).bandPassFilter(fs_highpass=0.5, fs_lowpass=40,
                                                                         module='scipy.signal')
                print(eeg.shape)

                # (7, timeSeries) -> (n, 7, 128*windows)
                eeg = eeg.reshape((len(useful_channels), -1, fs * windows), order='C').transpose(1, 0, 2)

                # 取中间的9分钟
                chunk_num = int(10 * 60 / windows)
                trash_num = int(30 / windows)
                trial['focussed'] = eeg[trash_num:chunk_num - trash_num]
                trial['unfocussed'] = eeg[chunk_num + trash_num:chunk_num * 2 - trash_num]
                trial['drowsed'] = eeg[chunk_num * 2 + trash_num:chunk_num * 3 - trash_num]

                data[f'trial_{i + 1}'] = trial
            with open(outp_dir + f'/subject_{s}.pkl', 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


class BCICompitionIV:
    def __init__(self):
        None

    def downloadAndSaveLocal(self):
        None


    def loadData(self, path='../../data/data.mat', format='4D'):
        '''
        :param format:
            '4D' -- X with shape (n, slices, leads, window_size, 1) -- y in one-hot format --  for CNN input;
            '3D' -- X with shape (n, C, time_series, 1) -- y in one-hot format --  for CNN input;
        :return: Xtrain, Xtest, ytrain, ytest
        '''
        data = loadmat(path)

        Xtrain, Xtest, ytrain, ytest = data['train_x'], data['test_x'], data['train_y'], data['test_y']
        print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
        if format == '3D':
            Xtrain = np.expand_dims(Xtrain, axis=3)  # (4608,22, 1125, 1)
            Xtest = np.expand_dims(Xtest, axis=3)  # (576, 22, 1125, 1)

        if format == '4D':
            # 对数据切片
            Xtrain = Xtrain.transpose(0, 2, 1)  # （4608，1125,22）
            Xtest = Xtest.transpose(0, 2, 1)  # （576,1125,22）

            # [trial, slices, channel, window_size]
            Xtrain = preprocessPipeline(data=Xtrain, fs=None).segment_dataset(400, 50)  # (4608,15,400,22)
            Xtest = preprocessPipeline(data=Xtest, fs=None).segment_dataset(400, 50)  # （576,15,400,22）

            # [trial, slices, channel, window_size]
            Xtrain = Xtrain.transpose(0, 1, 3, 2)  # （4608,15,22,400）
            Xtest = Xtest.transpose(0, 1, 3, 2)  # （576,15,22,400）

            # 输出为(n, slices, 导联, vec, 1)
            Xtrain = np.expand_dims(Xtrain, axis=4)  # (4608,15,22,400,1)
            Xtest = np.expand_dims(Xtest, axis=4)  # (576, 15,22,400,1)

        ytrain = ytrain.transpose(1, 0)
        ytest = ytest.transpose(1, 0)
        ytrain = tf.keras.utils.to_categorical(ytrain, 4).astype('int8')
        ytest = tf.keras.utils.to_categorical(ytest, 4).astype('int8')

        return Xtrain, Xtest, ytrain, ytest


class MNIST():
    def __init__(self):
        None

    def loadData(self):
        '''
        :return: train_images, train_labels, test_images, test_labels
        '''
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        img_row, img_col, channel = 28, 28, 1
        mnist_input_shape = (img_row, img_col, 1)
        # 将数据维度进行处理
        train_images = train_images.reshape(train_images.shape[0], img_row, img_col, channel)
        test_images = test_images.reshape(test_images.shape[0], img_row, img_col, channel)

        train_images = train_images.astype("float32")
        test_images = test_images.astype("float32")

        # 进行归一化处理
        train_images /= 255
        test_images /= 255

        # 将类向量，转化为类矩阵
        # 从 5 转换为 0 0 0 0 1 0 0 0 0 0 矩阵
        train_labels = tf.keras.utils.to_categorical(train_labels, 10)
        test_labels = tf.keras.utils.to_categorical(test_labels, 10)

        return train_images, train_labels, test_images, test_labels


def randomSort_and_reformat(Xtrain, Xtest, ytrain, ytest, format='3D'):
    '''
    :param Xtrain: np.array with shape (n, channel, timeSeries)
    :param Xtest: np.array with shape (n, channel, timeSeries)
    :param ytrain: np.array with shape (n, 1)
    :param ytest: np.array with shape (n, 1)
    :param format:
            '3D' -- return X with shape (n, H, W, C) -- y in one-hot format --  for CNN input;
            '2D' -- return X with shape  (n, time_step, vector_num) -- y in one-hot format -- for RNN input;
            '1D' -- return X with shape (n, m) -- y -- for SVM input;
    :return: Xtrain, Xtest, ytrain, ytest
    '''

    return None


if __name__ == '__main__':
    # K aggleEDFMASD().prepossessing(n_subjects=1, inp_dir='../../EEG Data/', windows=2)
    # Xtrain, Xtest, ytrain, ytest = KaggleEDFMASD().loadData(path='../../EEG Data/Disposed/withoutFeatureExtration/subject_1.pkl', format='3D')
    # Xtrain, Xtest, ytrain, ytest = BCICompitionIV().loadData(path='../../data/data.mat')
    Xtrain, Xtest, ytrain, ytest = BCICompitionIV().loadData(path='../../data/subject_1.mat')
    print(Xtrain.shape)
    print(ytrain.shape)
    print(ytest.shape)
    print(Xtest.shape)
    print(Xtrain)
    # print(ytest)
    # print(Xtrain)

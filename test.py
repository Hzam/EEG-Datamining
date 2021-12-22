# -*- coding:utf-8 -*-
# @Time : 2021/11/19 14:53
# @Author: ShuhuiLin
# @File : test.py

import numpy as np
import matplotlib.pyplot as plt
from preprocessedPipeline import preprocessPipeline
import pywt

def createData():
    # t=120，fs=40  ->  n=120*40
    duration = 120
    fs = 40
    timeSeries = np.linspace(start=0, stop=duration, num=fs*duration+1)
    timeSeries = timeSeries[:-1]
    print(len(timeSeries)) # 4800
    print(timeSeries[:41]) # [0.    0.025 0.05  0.075 0.1   0.125 0.15 ...

    # create x: fs=[2, 4, 4] Amplitude=[3, 2, 2]
    x1 = 3 * np.cos(4*np.pi*timeSeries)
    x2 = 2 * np.cos(8*np.pi*timeSeries)
    x = np.vstack((x1, x2, x2))
    print(x.shape)  # [3, 4800]

    return x, timeSeries, duration, fs

def testStft():
    x, timeSeries, duration, fs = createData()
    preprocessPipeline(data=x, fs=fs).getTimeFreqSpectrum(method='stft', window='dpss', nperseg=fs*2, noverlap=fs, nfft=fs*2, visualize=True)

def testCwt():
    x, timeSeries, duration, fs = createData()
    preprocessPipeline(data=x, fs=fs).getTimeFreqSpectrum(method='cwt', wavelet='morl', scales_num=128, visualize=True)

def testFft():
    x, timeSeries, duration, fs = createData()
    x = x[2]
    preprocessPipeline(data=x, fs=fs).getFreqSpectrum(visualize=True)

if __name__ == '__main__':
    testFft()





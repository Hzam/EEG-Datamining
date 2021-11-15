# -*- coding:utf-8 -*-
# @Time : 2021/11/13 14:19
# @Author: ShuhuiLin
# @File : prepossessPipeline.py


import mne
from scipy import signal
import numpy as np

class prepossessPipeline():
    def __init__(self, data, fs):
        '''
        :param data:
        :param fs: sampling frequency
        '''
        self.data = data
        self.fs = fs

    def bandPassFilter(self, fs_highpass, fs_lowpass, module='scipy.signal'):
        '''
        :param data: np.array with 1or2 dimension, like np.shape = (7,39120); Attention!! time series at the second dimension.
        :param fs_lowpass: lowpass frequency
        :param fs_highpass: highpass frequency
        :param module:  use 'mne' module, or 'scipy.signal' module
            mne 的过渡带较宽，scipy.signal的过渡带较窄，或许可以认为后者效果更好？
        :return: bandPassedData
        '''
        if module == 'scipy.signal':
            wn1 = 2 * fs_highpass / self.fs
            wn2 = 2 * fs_lowpass / self.fs
            b, a = signal.butter(8, [wn1, wn2], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
            bandPassedData = signal.filtfilt(b, a, self.data)
        elif module == 'mne':
            info = mne.create_info(
                ch_names=[str(i) for i in range(len(self.data))],
                ch_types=['eeg' for _ in range(len(self.data))],
                sfreq=self.fs
            )
            raw = mne.io.RawArray(self.data, info)
            raw.filter(fs_highpass, fs_lowpass, fir_design='firwin')
            # fig = raw.plot_psd(average=True)
            bandPassedData = raw.get_data()
        else:
            print('Invalid Param: "module", which shoule be choosed in ["mne", "scipy.signal"]')
            bandPassedData = None
        return bandPassedData

    def downSampling(self):
        return None

    ## (4608,15,400,22) ：4608个trial，每个trial 15个时间片，窗口大小400，22个通道
    def segment_dataset(self, window_size, step):
        win_x = []
        for i in range(self.data.shape[0]):
            win_x = win_x + [self.segment_signal_without_transition(self.data[i], window_size, step)]  # 将X[i]trial的切片进行连接
        win_x = np.array(win_x)
        return win_x







    ################################## 辅助函数 #############################
    # 滑动窗口【0,400】，【50,450】，【100,500】,etc
    def windows(self, dt, size, step):
        start = 0
        while (start + size) < dt.shape[0]:
            yield int(start), int(start + size)
            start += step

    # 在一个trial中进行滑动窗口切片，并连接
    def segment_signal_without_transition(self, dt, window_size, step):
        segments = []
        for (start, end) in self.windows(dt, window_size, step):
            if len(dt[start:end]) == window_size:
                segments = segments + [dt[start:end]]  # 将一个trial的切片进行连接
        return np.array(segments)







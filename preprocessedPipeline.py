# -*- coding:utf-8 -*-
# @Time : 2021/11/13 14:19
# @Author: ShuhuiLin
# @File : preprocessedPipeline.py


import mne
from scipy import signal, fftpack
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pywt


class preprocessPipeline:
    def __init__(self, data, fs):
        '''
        :param data: np.array with 1or2 dimension, like np.shape = (7,39120); Attention!! time series at the second dimension.
        :param fs: sampling frequency
        '''
        self.data = data
        self.fs = fs

    def bandPassFilter(self, fs_highpass, fs_lowpass, module='scipy.signal'):
        '''
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

    def getTimeFreqSpectrum(self, method='stft',
                            wavelet='morl', scales_num=100,  # 用于cwt的参数
                            window='dpss', nperseg=None, noverlap=None, nfft=None,  # 用于stft的参数
                            visualize=True):
        '''
        # param
            - method: 用于提取时频特征的方法
                当 method == 'cwt' 时的输入参数：
                    - wavelet: 参考 pywt.wavelist(kind='continuous')，默认值'morl'
                    - scales_num: 尺度函数的个数，详细说明见"desc"部分
                当 method == 'stft' 时的输入参数：
                    - window:
                    - nperseg: number per segment
                    - noverlap: number overlap between segments
                    - nfft: number of points used for fft, usually == nperseg
            - visualize: 是否画时频图

        # desc:
            小波变换中最重要的是选取小波的尺度函数scales，其关系到最终频带的位置，其确定较困难
            这里我们给出一个方便实用的接口，只需要设置两个参数：
                - wavelet： str类型，参考 pywt.wavelist(kind='continuous')
                - scales_num最终返回的频带的数目，从[0, fs/2]均分
                内部算法会根据小波的中心频率和尺度函数的个数计算出合适的尺度函数数组
            i.e.
                输入数据 x=[3,4800] 的三通道数据，采样频率fs=40。
                调用 coefs, freqs = preprocessPipeline(x, fs).getTimeFreqSpectrum(method='cwt', wavelet='morl', scales_num=100)
                print(coefs.shape)
                >>> (10, 3, 4800)

                print(freqs)
                >>>[1.00000000e+01 8.88912037e+00 7.77824074e+00 6.66736111e+00 5.55648148e+00 4.44560185e+00 3.33472222e+00 2.22384259e+00 1.11296296e+00 2.08333333e-03]
        '''

        if method == 'cwt':
            # 准备参数
            totalscal = np.shape(self.data)[-1]
            sampling_period = 1 / self.fs
            timeSeries = np.linspace(start=0, stop=totalscal-1, num=totalscal) / self.fs
            # 计算scales
            fc = pywt.central_frequency(wavelet)
            cparam = 4 * fc * totalscal   #  这里取4是对应最终freqs_max取到fs/2
            scales = cparam / np.linspace(start=totalscal, stop=1, num=scales_num)
            coefs, freqs = pywt.cwt(data=self.data, scales=scales, wavelet=wavelet, sampling_period=sampling_period, axis=-1)
            if visualize:
                for i in range(coefs.shape[1]):
                    plt.contourf(timeSeries, freqs, abs(coefs[:, i, :]))
                    plt.title('Magnitude of channel {0} --by CWT'.format(i))
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.show()
            return coefs, freqs
        elif method == 'stft':
            window = scipy.signal.windows.dpss(nperseg, nperseg//8) if window == 'dpss' else window
            f, t, Zxx = scipy.signal.stft(x=self.data, fs=self.fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=-1)
            if visualize:
                for i in range(Zxx.shape[0]):
                    #plt.pcolormesh(t, f, np.abs(Zxx[i]), vmin=0, vmax=3.5, shading='gouraud')
                    plt.contourf(t, f, abs(Zxx[i]))
                    plt.title('Magnitude of channel {0} --by STFT '.format(i))
                    plt.ylabel('Frequency [Hz]')
                    plt.xlabel('Time [sec]')
                    plt.show()
            return f, t, Zxx
        else:
            return None

    def getFreqSpectrum(self, visualize=True):
        # 参数
        L = int(np.shape(self.data)[-1])  # 信号长度
        N = int(np.power(2, np.ceil(np.log2(L))))  # 下一个最近二次幂
        if self.data.ndim == 1:
            FFT_y1 = np.abs(fftpack.fft(self.data, N)) / L * 2  # N点FFT 变化,但处于信号长度
            Fre = np.arange(int(N / 2)) * self.fs / N  # 频率坐标
            FFT_y1 = FFT_y1[range(int(N / 2))]  # 取一半
            if visualize:
                plt.plot(Fre, FFT_y1)
                plt.title('Magnitude of channel {0} --by FFT '.format(1))
                plt.ylabel('Magnitude')
                plt.xlabel('Frequency')
                plt.grid()
                plt.show()
            return Fre, FFT_y1
        else:
            print("目前仅支持1D数据")

    def downSampling(self):
        return None

    ## (n, vec, C) -> (N, slices, window_size, C)
    def segment_dataset(self, window_size, step):
        win_x = []
        for i in range(self.data.shape[0]):
            win_x = win_x + [
                self.segment_signal_without_transition(self.data[i], window_size, step)]  # 将X[i]trial的切片进行连接
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

    # 可视化time-frequency图
    def visualize_time_freq_graph(self, t, f, coefs,  method=None):
        return None

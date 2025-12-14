import pywt
import numpy as np 
import matplotlib.pyplot as plt

def wavelet_domain_denoise(x, data_signal):
    data_signal = list(data_signal)
    index = []
    data = []
    for i in range(len(x)):
        X = float(i)
        Y = float(data_signal[i])
        index.append(X)
        data.append(Y)
    w = pywt.Wavelet('db8')
    max_lev = pywt.dwt_max_level(len(data), w.dec_len)
    threadshold = 0.5
    wave_dec = pywt.wavedec(data, 'db8', level=max_lev)
    for i in range(1, len(wave_dec)):
        wave_dec[i] = pywt.threshold(wave_dec[i], threadshold*max(wave_dec[i]))
    data_rec = pywt.waverec(wave_dec, 'db8')
    return index, data, data_rec
        
        
if __name__ == '__main__':

   
    x = np.linspace(0, 10, 100)  # 时间序列
    data_signal = np.sin(x) + np.random.normal(0, 0.3, len(x))  # 添加噪声的正弦波
    index, data, data_rec = wavelet_domain_denoise(x, data_signal)
    
    # 绘制原始数据和去噪后的数据
    plt.figure()
    plt.plot(index, data, label='Original Data')
    plt.plot(index, data_rec, label='Denoised Data')
    plt.title('Denoising Test: Sine Wave with Noise')
    plt.legend()
    plt.savefig('.test.png')


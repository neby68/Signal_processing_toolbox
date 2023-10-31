import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import sys
sys.path.append(os.path.dirname(__file__))
from Running_mean_filter import running_mean_filter
from Gaussian_smoothing_time_series_filter import gaussian_smoothing_filter

def remove_outlier(signal, k=30, std_coef=2, debug=False):
    filtered_signal = signal.copy()
    n = len(filtered_signal)
    
    for i in range(k, n-k+1, k*2):
        print(i)

        mean = np.mean(signal[i-k:i+k])
        std = np.std(signal[i-k:i+k])
        outlier_arr = np.where( (signal[i-k:i+k]>mean+(std_coef*std)) | (signal[i-k:i+k]<mean-(std_coef*std)) )[0]
        if i-k >= 3700:
            print('ja')
            if debug:
                plt.figure()
                plt.plot(signal[i-k:i+k])
                plt.hlines(mean+(std_coef*std), 0, 2*k )
                plt.hlines(mean-(std_coef*std), 0, 2*k)
                plt.scatter(outlier_arr, signal[i-k:i+k][outlier_arr])
                plt.text(0,-5, f"start = {i-k} stop = {i+k}")
                plt.show(block=False)
        if (len(outlier_arr)>0):
            outlier_arr += (i-k)
            if 3184 in outlier_arr:
                print('ja')
            for outlier_idx,_ in enumerate(outlier_arr):
                lower_lim = max(1, outlier_arr[outlier_idx]-k)
                upper_lim = min(outlier_arr[outlier_idx]+k, n)
                filtered_signal[outlier_arr[outlier_idx]] = np.median(signal[lower_lim : upper_lim])

    return filtered_signal



data_path = os.path.join(os.path.dirname(__file__), "data")
data = sio.loadmat(os.path.join(data_path, 'denoising_codeChallenge.mat'))

signal = data["origSignal"][0]
prof_cleaned_signal = data["cleanedSignal"][0]

plt.figure()
plt.plot(signal)
plt.show(block=False)

sig_without_outlier = remove_outlier(signal, k=100, std_coef=2)
sig_without_outlier = remove_outlier(sig_without_outlier, k=100, std_coef=4)

plt.figure()
plt.plot(signal)

plt.figure()
plt.plot(sig_without_outlier)

filtered_signal = running_mean_filter(sig_without_outlier, k=40)
filtered_signal2 = running_mean_filter(filtered_signal, k=20)

filtered_gaussian_sig,_,_ = gaussian_smoothing_filter(signal, k = 40, fwhm = 25)

plt.figure()
plt.plot(sig_without_outlier, label="signal")
plt.plot(filtered_signal, label="mean_filtering")
plt.plot(filtered_signal2, label="mean_filtering2")
plt.plot(filtered_gaussian_sig, label="gaussian_filtering")
plt.legend()
plt.show()

plt.figure()
plt.plot(sig_without_outlier, alpha=0.2, label="sig without oulier")
plt.plot(filtered_signal2[40:-40], label="filtered signal")
plt.plot(filtered_signal[40:-40], label="filtered signal2")
plt.plot(prof_cleaned_signal, label="prof filtered signal")
plt.legend()
plt.show()
print('ja')
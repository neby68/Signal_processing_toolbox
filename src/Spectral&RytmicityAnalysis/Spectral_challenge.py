import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import os


## load in birdcall (source: https://www.xeno-canto.org/403881)
directory = os.path.dirname(__file__)
data = sio.loadmat(os.path.join(directory, 'data/spectral_codeChallenge.mat'))

srate = data["srate"][0][0]
signal = data["signal"][0]
time = data["time"][0]

plt.figure()
plt.plot(time, signal)

win_size = int(0.25*srate)
nb_win = int(len(signal)/win_size)
time_win = np.linspace(time[0], time[-1], nb_win)
freq_arr = np.linspace(0, srate/2, int(win_size/2))

fft_arr = np.zeros((int(win_size/2), nb_win))
for i in range(0, nb_win):
    idx = int(i*win_size)
    fft_arr[:,i] = abs(scipy.fftpack.fft(signal[idx:idx+win_size])[:int(win_size/2)]/win_size)**2

plt.figure()
plt.pcolormesh(time_win,freq_arr,fft_arr)
plt.show()
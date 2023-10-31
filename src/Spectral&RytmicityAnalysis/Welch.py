import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import os

directory = os.path.dirname(__file__)
matdat  = sio.loadmat(os.path.join(directory, 'data/EEGrestingState.mat'))
eegdata = matdat['eegdata'][0]
srate   = matdat['srate'][0]


# time vector
N = len(eegdata)
timevec = np.arange(0,N)/srate


# plot the data
plt.title("EEG signal")
plt.plot(timevec,eegdata,'k')
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage (\muV)')
plt.show()

## one big FFT (not Welch's method)

# "static" FFT over entire period, for comparison with Welch
eegpow = np.abs( scipy.fftpack.fft(eegdata)/N )**2
hz = np.linspace(0,srate/2,int(np.floor(N/2)+1))


# create Hann window
winsize = int( 2*srate ) # 2-second window
hannw = .5 - np.cos(2*np.pi*np.linspace(0,1,winsize))/2

# number of FFT points (frequency resolution)
nfft = srate*100

f, welchpow = scipy.signal.welch(eegdata,fs=srate,window=hannw,
                                 nperseg=winsize,noverlap=winsize/4,
                                 nfft=nfft)


plt.figure()
plt.plot(f,welchpow, label="welch")
plt.plot(hz,np.log(eegpow[0:len(hz)]), label="fft")
plt.xlim([0,40])
plt.xlabel('frequency [Hz]')
plt.ylabel('Power')
plt.legend()
plt.show()

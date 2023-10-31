import matplotlib.pyplot as plt
# from scipy import signal
import scipy
import copy
import os
import scipy.io as sio
import numpy as np
from WindowedSinc_filter import sinc_passband, sinc_window


data_path = os.path.join(os.path.dirname(__file__), "data")
data = sio.loadmat(os.path.join(data_path, 'filtering_codeChallenge.mat'))

signal = data["x"]
signal = np.squeeze(signal)
prof_cleaned_signal = data["y"]
prof_cleaned_signal = np.squeeze(prof_cleaned_signal)
fs = data["fs"][0][0]
f_niquyst = int(fs/2)
N = len(signal)
time = np.linspace(0,N,N)
freq_arr = np.linspace(0, fs, len(signal))


plt.figure()
plt.plot(signal)
plt.plot(prof_cleaned_signal)


#%% investigate frequency response

signal_f = scipy.fft.fft(signal)
signal_f_pow = abs(signal_f)**2

plt.figure()
plt.plot(freq_arr[:int(N/2)], signal_f_pow[:int(N/2)]) 


#%% Chose the best kernel

fc_low = 5
fc_high = 18
frange = [fc_low, fc_high]
FIR_order   = int( 15*fs/frange[0] )
FIR_order += 1 if FIR_order%2==0 else FIR_order
freq_kernel_arr = np.linspace(0,f_niquyst,f_niquyst)

###########  firwin  ###########
filtkern_firlwin = scipy.signal.firwin(FIR_order,frange,fs=fs,pass_zero=False)

###########  fir  ###########
transw = 0.1
shape = [ 0, 0, 1, 1, 0, 0 ]
frex  = [ 0, frange[0]-frange[0]*transw, frange[0], frange[1], frange[1]+frange[1]*transw, f_niquyst ]
filtkern_firls = scipy.signal.firls(FIR_order,frex,shape,fs=fs)

###########   wsync   ###########
# time  = np.arange(-4,4,1/fs)
N_wsinc = int( 15*fs/frange[0] )
time_wsync = np.linspace(-N_wsinc/2,N_wsinc/2,N_wsinc)/ fs

# create sinc low pass filter function
sincfilt_low_pass = (np.sin(2*np.pi*fc_high*time_wsync) / time_wsync)
# adjust NaN and normalize filter to unit-gain
sincfilt_low_pass[~np.isfinite(sincfilt_low_pass)] = np.max(sincfilt_low_pass)
sincfilt_low_pass = sincfilt_low_pass/np.sum(sincfilt_low_pass)

# create sinc high pass filter function
sincfilt_high_pass = (np.sin(2*np.pi*fc_low*time_wsync) / time_wsync)
# adjust NaN and normalize filter to unit-gain
sincfilt_high_pass[~np.isfinite(sincfilt_high_pass)] = np.max(sincfilt_high_pass)
sincfilt_high_pass = sincfilt_high_pass/np.sum(sincfilt_high_pass)

# create sinc pass band filter function
sincfilt = sincfilt_low_pass - sincfilt_high_pass

# windowed sinc filter
sincfiltW = sincfilt * np.hanning(N_wsinc)


###########   butter   ###########
butter_order = 5
b, a = scipy.signal.butter(butter_order, [fc_low, fc_high], btype='band', fs=fs)
# Calcul de la réponse en fréquence
freq, f_response_butter = scipy.signal.freqz(b, a, worN=1000, fs=fs)



#plot FIR function


kernel_name_arr = ["filtkern_firlwin", "filtkern_firls", "filtkern_wsync", "fkern_butter"]
plt.figure()
for i, filtkern in enumerate([filtkern_firlwin, filtkern_firls, sincfiltW]):
    kernel_name = kernel_name_arr[i]
    # kernel_name = filtkern_wsync
    hz      = np.linspace(0,fs/2,int(np.floor(len(filtkern)/2)+1))
    
    # plot amplitude spectrum of the filter kernel
    # compute the power spectrum of the filter kernel
    filtpow = np.abs(scipy.fftpack.fft(filtkern))**2
    plt.plot(hz,filtpow[0:len(hz)],label=kernel_name)

#plot IIR
hz = np.linspace(0,fs/2, int(len(f_response_butter)))
plt.plot(hz, np.abs(f_response_butter), label="butter")

plt.plot(frex,shape,'ro-',label='Ideal')
plt.xlim([0,frange[0]*4])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.legend()
plt.title('Frequency response of filter (firls)')
plt.show()

#%% filter

filt_sig_firlwin = scipy.signal.filtfilt(filtkern_firlwin,1, signal)
filt_sig_firls = scipy.signal.filtfilt(filtkern_firls, 1, signal)
filt_sig_wsinc = scipy.signal.filtfilt(sincfiltW, 1, signal)
filt_sig_butter = scipy.signal.filtfilt(b, a, signal)


plt.figure()
plt.plot(freq_arr[:int(N/2)], signal_f_pow[:int(N/2)]) 

for i, filt_sig in enumerate([filt_sig_firlwin, filt_sig_firls, filt_sig_wsinc, filt_sig_butter]):
    kernel_name = kernel_name_arr[i]
    plt.plot(freq_arr[:int(N/2)], (abs(scipy.fft.fft(filt_sig))**2)[:int(N/2)], label=kernel_name )
    
plt.legend()




#
# W-sinc has been choosen
#now apply the second pass band filter

fc_low = 25
fc_high = 32
frange = [fc_low, fc_high]

N_wsinc = int( 30*fs/frange[0] )
time_wsync = np.linspace(-N_wsinc/2,N_wsinc/2,N_wsinc)/ fs

# create sinc low pass filter function
sincfilt_low_pass = (np.sin(2*np.pi*fc_high*time_wsync) / time_wsync)
# adjust NaN and normalize filter to unit-gain
sincfilt_low_pass[~np.isfinite(sincfilt_low_pass)] = np.max(sincfilt_low_pass)
sincfilt_low_pass = sincfilt_low_pass/np.sum(sincfilt_low_pass)

# create sinc high pass filter function
sincfilt_high_pass = (np.sin(2*np.pi*fc_low*time_wsync) / time_wsync)
# adjust NaN and normalize filter to unit-gain
sincfilt_high_pass[~np.isfinite(sincfilt_high_pass)] = np.max(sincfilt_high_pass)
sincfilt_high_pass = sincfilt_high_pass/np.sum(sincfilt_high_pass)

# create sinc pass band filter function
sincfilt = sincfilt_low_pass - sincfilt_high_pass

# windowed sinc filter
sincfiltW = sincfilt * np.hanning(N_wsinc)

filt_sig_wsinc2 = scipy.signal.filtfilt(sincfiltW, 1, signal)

plt.figure()
plt.plot(freq_arr[:int(N/2)], signal_f_pow[:int(N/2)]) 
plt.plot(freq_arr[:int(N/2)], (abs(scipy.fft.fft(filt_sig_wsinc2))**2)[:int(N/2)], label="wsinc2" )



sig_freq = scipy.fft.fft(filt_sig_wsinc) + scipy.fft.fft(filt_sig_wsinc2) 

plt.figure()
plt.plot(freq_arr[:int(N/2)], signal_f_pow[:int(N/2)]) 
plt.plot(freq_arr[:int(N/2)], (abs(sig_freq)**2)[:int(N/2)], label="combined" )


cleaned_signal = scipy.fft.ifft(sig_freq)

plt.figure()
plt.plot(signal, label='signal', alpha=0.2)
plt.plot(cleaned_signal, label='cleaned_signal')
plt.plot(prof_cleaned_signal, label='prof cleaned signal')
plt.legend()


print('end')
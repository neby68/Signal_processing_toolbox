import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import os
import sys

root_path = os.path.join(os.path.dirname(__file__), "../.." )
sys.path.append(root_path)


def welch_example():
    """
    **Welch method:**

        Cut the signal in n window and compute fft on each of them.

        
        Imagine a time-domain signal represented as a sinusoidal wave. 
        At the core of the Welch method, you divide this signal into partially overlapping segments. 
        For each segment, you apply a window to mitigate edge effects. T
        hen, instead of taking the FFT of the entire signal, you take the FFT of each modified segment and average them.

        The idea is that this overlapping approach reduces the variance of the estimation,
        which can be particularly useful when the signal changes over time.

    **Advantage over fft:**

        More adapt if signal characteristics change over time.
        E.g Noise increase/ decrease/ change of shape over time.
    
    **Example:**

    .. image:: _static/images/Spectral_RytmicityAnalysis/EEG_signal.png

    .. image:: _static/images/Spectral_RytmicityAnalysis/welch_versus_fft.png

    """
    file_data_path = os.path.join(root_path, r"data\spetracl&rythmicity_analysis\EEGrestingState.mat")
    if not os.path.isfile(file_data_path):
        print("\n\tFile data could not be found. Please check that you have access to it\n")
        return

    matdat  = sio.loadmat(file_data_path)
    eegdata = matdat['eegdata'][0]
    srate   = matdat['srate'][0]

    # time vector
    N = len(eegdata)
    timevec = np.arange(0,N)/srate

    # plot the data
    plt.figure()
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
    plt.title("Welch versus fft")
    plt.plot(f,welchpow, label="welch")
    plt.plot(hz,np.log(eegpow[0:len(hz)]), label="fft")
    plt.xlim([0,40])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Power')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    welch_example()
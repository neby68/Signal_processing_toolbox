import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import os
import sys

root_path = os.path.join(os.path.dirname(__file__), "../.." )
sys.path.append(root_path)



def plot_spectrogramm_example():
    """_summary_
    Spectrogramm is usefull to see the evolution of the frequency along time.
    e.g in an audio recording different birds are singing, at different frequency.
    This is how the spectrogramm look like

    .. image:: _static/images/Spectral_RytmicityAnalysis/birdcall_spectrogramm.png

    """
    file_data_path = os.path.join(root_path, r"data\spetracl&rythmicity_analysis\XC403881.wav")
    if not os.path.isfile(file_data_path):
        print("\n\tFile data could not be found. Please check that you have access to it\n")
        return
    ## load in birdcall (source: https://www.xeno-canto.org/403881)
    fs,bc = scipy.io.wavfile.read(file_data_path)

    # create a time vector based on the data sampling rate
    n = len(bc)
    timevec = np.arange(0,n)/fs

    # plot the data from the two channels
    plt.figure()
    plt.plot(timevec,bc)
    plt.xlabel('Time (sec.)')
    plt.title('Time domain')
    plt.show()

    # compute the power spectrum
    hz = np.linspace(0,fs/2,int(np.floor(n/2)+1))
    bcpow = np.abs(scipy.fftpack.fft( scipy.signal.detrend(bc[:,0]) )/n)**2

    # now plot it
    plt.figure()
    plt.plot(hz,bcpow[0:len(hz)])
    plt.xlabel('Frequency (Hz)')
    plt.title('Frequency domain')
    plt.xlim([0,8000])
    plt.show()


    ## time-frequency analysis via spectrogram
    frex,time,pwr = scipy.signal.spectrogram(bc[:,0],fs)
    
    plt.figure()
    plt.title("spectrogramme of bird calls")
    plt.pcolormesh(time,frex,pwr,vmin=0,vmax=9)
    plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
    plt.show()
    # pwprint('rnd')


if __name__ == "__main__":
    plot_spectrogramm_example()
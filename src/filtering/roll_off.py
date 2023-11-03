import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy import signal
import scipy


if __name__ == "__main__":
    ## create a windowed sinc filter

    # simulation parameters
    srate = 1000
    time  = np.arange(-4,4,1/srate)
    pnts  = len(time)

    # FFT parameters
    nfft = 10000
    hz   = np.linspace(0,srate/2,int(np.floor(nfft/2)+1))

    filtcut  = 15
    sincfilt = np.sin(2*np.pi*filtcut*time) / time

    # adjust NaN and normalize filter to unit-gain
    sincfilt[~np.isfinite(sincfilt)] = np.max(sincfilt)
    sincfilt = sincfilt/np.sum(sincfilt)

    # windowed sinc filter
    sincfiltW = sincfilt * signal.windows.hann(pnts)

    # spectrum of filter
    sincX = 10*np.log10(np.abs(scipy.fftpack.fft(sincfiltW,n=nfft))**2)
    sincX = sincX[:len(hz)]


    ## create a Butterworth high-pass filter

    # generate filter coefficients (Butterworth)
    filtb,filta = signal.butter(5,filtcut/(srate/2),btype='lowpass')

    # test impulse response function (IRF)
    impulse  = np.zeros(1001)
    impulse[500] = 1
    fimpulse = signal.filtfilt(filtb,filta,impulse)

    # spectrum of filter response
    butterX = 10*np.log10(np.abs(scipy.fftpack.fft(fimpulse,nfft))**2)
    butterX = butterX[:len(hz)]


    ## plot frequency responses

    plt.plot(hz,sincX, label="w-sync")
    plt.plot(hz,butterX, label="butter")

    plotedge = int(np.argmin( (hz-filtcut*3)**2 ))
    plt.xlim([0,filtcut*3])
    plt.ylim([np.min((butterX[plotedge], sincX[plotedge])), 5])
    plt.plot([filtcut,filtcut],[-190, 5],'k--')



    # find -3 dB after filter edge
    filtcut_idx = np.min( (hz-filtcut)**2 )

    sincX3db   = np.argmin( (sincX--3)**2 )
    butterX3db = np.argmin( (butterX--3)**2 )

    # add to the plot
    plt.plot([hz[sincX3db],hz[sincX3db]],[-180,5],'b--')
    plt.plot([hz[butterX3db],hz[butterX3db]],[-180,5],'r--')



    # find double the frequency
    sincXoct   = np.argmin( (hz-hz[sincX3db]*2)**2 )
    butterXoct = np.argmin( (hz-hz[butterX3db]*2)**2 )

    # add to the plot
    plt.plot([hz[sincXoct],hz[sincXoct]],[-180,5],'b--')
    plt.plot([hz[butterXoct],hz[butterXoct]],[-180,5],'r--')



    # find attenuation from that point to double its frequency
    sincXatten   = sincX[sincX3db*2]
    butterXatten = butterX[butterX3db*2]

    sincXrolloff   = (sincX[sincX3db]-sincX[sincXoct]) / (hz[sincXoct]-hz[sincX3db])
    butterXrolloff = (butterX[butterX3db]-butterX[butterXoct]) / (hz[butterXoct]-hz[butterX3db])


    plt.plot([hz[sincX3db], hz[sincXoct]], [sincX[sincX3db], sincX[sincXoct]], color="green",label="rool_off w-sync")
    plt.plot([hz[butterX3db], hz[butterXoct]], [butterX[butterX3db], butterX[butterXoct]], color="black", label="rool_off butter")

    # report!
    plt.title('Sinc: %.3f, Butterworth: %.3f' %(sincXrolloff,butterXrolloff) )
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.show()
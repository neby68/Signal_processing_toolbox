import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import scipy
import scipy.io as sio
import copy




def morlet_wavelet_example():
    """
    .. image:: _static/images/wavelet/morlet_wavelet.png

    """
    # general simulation parameters
    fs = 1024
    npnts = fs*5 # 5 seconds

    # centered time vector
    timevec = np.arange(0,npnts)/fs
    timevec = timevec - np.mean(timevec)

    # for power spectrum
    hz = np.linspace(0,fs/2,int(np.floor(npnts/2)+1))
    freq = 4 # peak frequency

    ## Morlet wavelet

    # parameters
    csw  = np.cos(2*np.pi*freq*timevec) # cosine wave
    fwhm = .5 # full-width at half-maximum in seconds
    gaussian = np.exp( -(4*np.log(2)*timevec**2) / fwhm**2 ) # Gaussian

    # Morlet wavelet
    MorletWavelet = csw * gaussian

    # amplitude spectrum
    MorletWaveletPow = np.abs(scipy.fftpack.fft(MorletWavelet)/npnts)


    # time-domain plotting
    plt.subplot(211)
    plt.plot(timevec,MorletWavelet,'k')
    plt.xlabel('Time (sec.)')
    plt.ylabel('Amplitude')
    plt.title('Morlet wavelet in time domain')

    # frequency-domain plotting
    plt.subplot(212)
    plt.plot(hz,MorletWaveletPow[:len(hz)],'k')
    plt.xlim([0,freq*3])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Morlet wavelet in frequency domain')
    plt.show()



def haar_wavelet_example():
    """
    .. image:: _static/images/wavelet/haar_wavelet.png

    """

    ## general simulation parameters
    fs = 1024
    npnts = fs*5 # 5 seconds

    # centered time vector
    timevec = np.arange(0,npnts)/fs
    timevec = timevec - np.mean(timevec)

    # for power spectrum
    hz = np.linspace(0,fs/2,int(np.floor(npnts/2)+1))
    freq = 4 # peak frequency

    ## Haar wavelet

    # create Haar wavelet
    HaarWavelet = np.zeros(npnts)
    HaarWavelet[np.argmin(timevec**2) : np.argmin((timevec-.5)**2) ] = 1
    HaarWavelet[np.argmin((timevec-.5)**2) : np.argmin((timevec-1-1/fs)**2)] = -1

    # amplitude spectrum
    HaarWaveletPow = np.abs(scipy.fftpack.fft(HaarWavelet)/npnts)


    # time-domain plotting
    plt.subplot(211)
    plt.plot(timevec,HaarWavelet,'k')
    plt.xlabel('Time (sec.)')
    plt.ylabel('Amplitude')
    plt.title('Haar wavelet in time domain')

    # frequency-domain plotting
    plt.subplot(212)
    plt.plot(hz,HaarWaveletPow[:len(hz)],'k')
    plt.xlim([0,freq*3])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Haar wavelet in frequency domain')
    plt.show()



def mexican_wavelet_example():
    """
    .. image:: _static/images/wavelet/mexican_wavelet.png

    """
    ## general simulation parameters
    fs = 1024
    npnts = fs*5 # 5 seconds

    # centered time vector
    timevec = np.arange(0,npnts)/fs
    timevec = timevec - np.mean(timevec)

    # for power spectrum
    hz = np.linspace(0,fs/2,int(np.floor(npnts/2)+1))
    freq = 4 # peak frequency


    ## Mexican hat wavelet

    # the wavelet
    s = .4
    MexicanWavelet = (2/(np.sqrt(3*s)*np.pi**.25)) * (1- (timevec**2)/(s**2) ) * np.exp( (-timevec**2)/(2*s**2) )

    # amplitude spectrum
    MexicanPow = np.abs(scipy.fftpack.fft(MexicanWavelet)/npnts)


    # time-domain plotting
    plt.subplot(211)
    plt.plot(timevec,MexicanWavelet,'k')
    plt.xlabel('Time (sec.)')
    plt.ylabel('Amplitude')
    plt.title('Mexican wavelet in time domain')

    # frequency-domain plotting
    plt.subplot(212)
    plt.plot(hz,MexicanPow[:len(hz)],'k')
    plt.xlim([0,freq*3])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Mexican wavelet in frequency domain')
    plt.show()



def DoG_wavelet_example():
    """
    .. image:: _static/images/wavelet/DoG_wavelet.png

    """
    ## general simulation parameters
    fs = 1024
    npnts = fs*5 # 5 seconds

    # centered time vector
    timevec = np.arange(0,npnts)/fs
    timevec = timevec - np.mean(timevec)

    # for power spectrum
    hz = np.linspace(0,fs/2,int(np.floor(npnts/2)+1))
    freq = 4 # peak frequency


    ## Difference of Gaussians (DoG)
    # (approximation of Laplacian of Gaussian)

    # define sigmas
    sPos = .1
    sNeg = .5

    # create the two GAussians
    gaus1 = np.exp( (-timevec**2) / (2*sPos**2) ) / (sPos*np.sqrt(2*np.pi))
    gaus2 = np.exp( (-timevec**2) / (2*sNeg**2) ) / (sNeg*np.sqrt(2*np.pi))

    # their difference is the DoG
    DoG = gaus1 - gaus2


    # amplitude spectrum
    DoGPow = np.abs(scipy.fftpack.fft(DoG)/npnts)


    # time-domain plotting
    plt.subplot(211)
    plt.plot(timevec,DoG,'k')
    plt.xlabel('Time (sec.)')
    plt.ylabel('Amplitude')
    plt.title('DoG wavelet in time domain')

    # frequency-domain plotting
    plt.subplot(212)
    plt.plot(hz,DoGPow[:len(hz)],'k')
    plt.xlim([0,freq*3])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('DoG wavelet in frequency domain')
    plt.show()


if __name__ == "__main__":
    morlet_wavelet_example()
    haar_wavelet_example()
    mexican_wavelet_example()
    DoG_wavelet_example()
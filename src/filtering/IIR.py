
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy



def IRR_filter():
    """

    **1. Creation of the filter kernel using butter funtion**

    The IRR filter are composed of two kernel, often call "a" and "b".

    .. image:: _static/images/filtering/IIR_kernel.png

    **2. Analysis of the frequency response**

    .. image:: _static/images/filtering/IIR_frequency_response_wrong.png

    As IRR kernel are from very little order (comparing to FIR)

    e.g. butter filter from order 4 means the Kernel signal will be composed
        of only 4*2+1 points. 
        These lead in only 6 point in the frequency domain.

    A better way to evaluate the kernel function is to filter a basic impulse response (arr = [0, 0, 1, 0, 0])
    and take a look at its frequency response

    .. image:: _static/images/filtering/IIR_filter_an_Impulse.png

    .. image:: _static/images/filtering/IRR_frequency_response.png

    .. image:: _static/images/filtering/IRR_frequency_response_log.png
    """
    # filter parameters
    srate   = 1024 # hz
    nyquist = srate/2
    frange  = np.array([20,45])

    # create filter coefficients
    fkernB, fkernA = signal.butter(5, frange/nyquist, btype='bandpass')

    #filter
    plt.figure()
    plt.subplot(121)
    plt.plot(fkernA,'rs-',label='A')
    plt.xlabel('Time points')
    plt.ylabel('Filter coeffs.')
    plt.title('Butter Kernel A')
    plt.legend()

    plt.subplot(122)
    plt.plot(fkernB*1e5,'ks-',label='B')
    plt.xlabel('Time points')
    plt.ylabel('Filter coeffs.')
    plt.title('Butter Kernel B')
    plt.legend()

    
    ######  FIR approach ######

    # power spectrum of filter coefficients
    filtpow = np.abs(scipy.fftpack.fft(fkernB))**2
    hz = np.linspace(0,srate/2,int(np.floor(len(fkernB)/2)+1))


    plt.figure()
    plt.stem(hz,filtpow[0:len(hz)],'ks-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power spectrum filter coeffs.')
    plt.show()


    ## how to evaluate an IIR filter: filter an impulse
    # generate the impulse
    impres = np.zeros(1001)
    impres[501] = 1

    # apply the filter
    fimp = signal.lfilter(fkernB,fkernA,impres,axis=-1)

    # compute power spectrum
    fimpX = np.abs(scipy.fftpack.fft(fimp))**2
    hz = np.linspace(0,nyquist,int(np.floor(len(impres)/2)+1))


    # plot
    plt.figure()
    plt.plot(impres,'k',label='Impulse')
    plt.plot(fimp,'r',label='Filtered')
    plt.xlim([1,len(impres)])
    plt.ylim([-.06,.06])
    plt.legend()
    plt.xlabel('Time points (a.u.)')
    plt.title('Filtering an impulse')
    plt.show()

    plt.figure()
    plt.plot(hz,fimpX[0:len(hz)],'ks-')
    plt.plot([0,frange[0],frange[0],frange[1],frange[1],nyquist],[0,0,1,1,0,0],'r')
    plt.xlim([0,100])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation')
    plt.title('Frequency response of filter (Butterworth)')
    plt.show()

    plt.figure()
    plt.plot(hz,10*np.log10(fimpX[0:len(hz)]),'ks-')
    plt.xlim([0,100])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation (dB)')
    plt.title('Frequency response of filter (Butterworth)')
    plt.show()

    print('end')



if __name__ == "__main__":
    IRR_filter()
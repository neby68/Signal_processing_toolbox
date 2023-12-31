import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import copy


####################  FIR  ##########################

def FIR_filter_example():
    """
    Example of FIR filter for a narrowband filter

    **1. Creation of the filter kernel using FIR funtion**

    - Creation of the input of the FIR function

    .. image:: _static/images/filtering/FIR_input.png

    - Output of the FIR function is the kernel

    .. image:: _static/images/filtering/FIR_kernel.png

    **2. Analysis of the frequency response**
    
    - Analysis of the frequency response in comparison of the ideal response

    .. image:: _static/images/filtering/FIR_frequency_response.png

    - In dB (it allows to see more details)

    .. image:: _static/images/filtering/FIR_frequency_response_log.png

    """
    # filter parameters
    srate   = 1024 # hz
    nyquist = srate/2
    frange  = [20,45]
    transw  = .1
    order   = int( 5*srate/frange[0] )

    # order must be odd
    if order%2==0:
        order += 1

    # define filter shape
    shape = [ 0, 0, 1, 1, 0, 0 ]
    frex  = [ 0, frange[0]-frange[0]*transw, frange[0], frange[1], frange[1]+frange[1]*transw, nyquist ]

    # filter kernel
    filtkern = signal.firls(order,frex,shape,fs=srate)

    plt.figure()
    plt.title('Shape input for of FIR ')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.plot(shape)

    # time-domain filter kernel
    plt.figure()
    plt.plot(filtkern)
    plt.xlabel('Time points')
    plt.title('Filter kernel (firls)')
    plt.show()

    # compute the power spectrum of the filter kernel
    filtpow = np.abs(scipy.fftpack.fft(filtkern))**2
    # compute the frequencies vector and remove negative frequencies
    hz      = np.linspace(0,srate/2,int(np.floor(len(filtkern)/2)+1))
    filtpow = filtpow[0:len(hz)]


    # plot amplitude spectrum of the filter kernel
    plt.figure()
    plt.plot(hz,filtpow,'ks-',label='Actual')
    plt.plot(frex,shape,'ro-',label='Ideal')
    plt.xlim([0,frange[0]*4])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.legend()
    plt.title('Frequency response of filter (firls)')
    plt.show()


    # Same as above but logarithmically scaled
    plt.figure()
    plt.plot(hz,10*np.log10(filtpow),'ks-',label='Actual')
    # plt.plot([frange[0],frange[0]],[-40,5],'ro-',label='Ideal')
    plt.xlim([0,frange[0]*4])
    plt.ylim([-40,5])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain (log)')
    plt.legend()
    plt.title('Frequency response of filter (firls)')
    plt.show()




#%%

####################  FIR win ##########################
def FIRwin_eample():
    # filter parameters
    srate   = 1024 # hz
    nyquist = srate/2
    frange  = [20,45]
    transw  = .1
    order   = int( 5*srate/frange[0] )

    # force odd order
    if order%2==0:
        order += 1

    ### --- NOTE: Python's firwin corresponds to MATLAB's fir1 --- ###

    # filter kernel
    filtkern = signal.firwin(order,frange,fs=srate,pass_zero=False)

    plt.figure()
    plt.title('Shape input for of FIRwin ')
    plt.plot(shape)

    # time-domain filter kernel
    plt.figure()
    plt.plot(filtkern)
    plt.xlabel('Time points')
    plt.title('Filter kernel (firwin)')
    plt.show()

    # compute the power spectrum of the filter kernel
    filtpow = np.abs(scipy.fftpack.fft(filtkern))**2
    # compute the frequencies vector and remove negative frequencies
    hz      = np.linspace(0,srate/2,int(np.floor(len(filtkern)/2)+1))
    filtpow = filtpow[0:len(hz)]

    # plot amplitude spectrum of the filter kernel
    plt.figure()
    plt.plot(hz,filtpow,'ks-',label='Actual')
    plt.plot([0,frange[0],frange[0],frange[1],frange[1],nyquist],[0,0,1,1,0,0],'ro-',label='Ideal')
    plt.xlim([0,frange[0]*4])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.legend()
    plt.title('Frequency response of filter (firwin)')
    plt.show()

    # Same as above but logarithmically scaled
    plt.figure()
    plt.plot(hz,10*np.log10(filtpow),'ks-',label='Actual')
    # plt.plot([frange[0],frange[0]],[-100,5],'ro-',label='Ideal')
    plt.xlim([0,frange[0]*4])
    plt.ylim([-80,5])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter gain')
    plt.legend()
    plt.title('Frequency response of filter (firwin)')
    plt.show()

    plt.print('end')


if __name__ == "__main__":
    FIR_filter_example()
    # FIRwin_filter_example()
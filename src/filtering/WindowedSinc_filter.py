import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from scipy import signal
import scipy




# def sinc_window(x: np.ndarray, fc:int):
#     return np.sinc(2 * fc * x)

# def inverted_sinc_window(x, fc):
#     return 1 - np.sinc(2 * fc * x)

# def sinc_passband(x, fc_low, fc_high):
#     return np.sinc(2 * fc_high * x) - np.sinc(2 * fc_low * x)



def sinc_filter_example():
    r"""
    .. math::
        \text{sinc}(t) = \frac{\sin(2 \pi f_c t)}{t}

    with

    .. math::
        f_c : \text{cut-off frequency},
        t : \text{timestamps}

    .. image:: _static/images/filtering/Sinc_kernel_and_frequency_response.png

    """

    # simulation params
    srate = 1000
    time  = np.arange(-4,4,1/srate)
    pnts  = len(time)
    hz = np.linspace(0,srate/2,int(np.floor(pnts/2)+1))
    # create sinc function
    f = 5
    sincfilt = np.sin(2*np.pi*f*time) / time

    # adjust NaN and normalize filter to unit-gain
    sincfilt[~np.isfinite(sincfilt)] = np.max(sincfilt)
    sincfilt = sincfilt/np.sum(sincfilt)

    # plot the sinc filter
    plt.figure()
    plt.subplot(121)
    plt.plot(time,sincfilt,'k', label="sync")
    plt.ylabel('Power')
    plt.xlabel('Time (s)')
    plt.title('sinc kernel')
    
    # plot the power spectrum
    plt.subplot(122)
    plt.title('sinc frequency response')
    pw = np.abs(scipy.fftpack.fft(sincfilt))
    plt.plot(hz,pw[:len(hz)],'k', label="sync")
    plt.xlim([0,f*3])
    plt.yscale('log')
    plt.plot([f,f],[0,1],'r--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.legend()
    plt.show()


def sinc_window_filter_example():
    """

    The sinc filter can be improved by multipling a window with itself.

    Different window exist with different caracteristics:

    * Hann (hanning)
    * Hamming
    * Gaus


    .. image: _static/images/filtering/Sinc_different_window.png

    .. image: _static/images/filtering/sinc_kernel_with_different_window_zoomed.png
    
    .. image: _static/images/filtering/sinc_kernel_with_different_window.png

    .. image: _static/images/filtering/sincw_frequency_response.png

    """
    ####### Window sinc filter ##########

    # simulation params
    srate = 1000
    time  = np.arange(-4,4,1/srate)
    pnts  = len(time)
    hz = np.linspace(0,srate/2,int(np.floor(pnts/2)+1))

    # create sinc function
    f = 5
    sincfilt = np.sin(2*np.pi*f*time) / time

    # adjust NaN and normalize filter to unit-gain
    sincfilt[~np.isfinite(sincfilt)] = np.max(sincfilt)
    sincfilt = sincfilt/np.sum(sincfilt)

    ## with different windowing functions

    sincfiltW = np.zeros((4,pnts))
    tapernames = ['no window','Hann','Hamming','Gauss']
    
    sincfiltW[0,:]= sincfilt

    # with Hann taper
    # sincfiltW[0,:] = sincfilt * np.hanning(pnts)
    hannw = .5 - np.cos(2*np.pi*np.linspace(0,1,pnts))/2
    sincfiltW[1,:] = sincfilt * hannw

    # with Hamming taper
    #sincfiltW[1,:] = sincfilt * np.hamming(pnts)
    hammingw = .54 - .46*np.cos(2*np.pi*np.linspace(0,1,pnts))
    sincfiltW[2,:] = sincfilt * hammingw

    # with Gaussian taper
    sincfiltW[3,:] = sincfilt * np.exp(-time**2)

    #plot the windows
    plt.figure()
    plt.title("different window")
    plt.plot(time, hannw,label="hannw")
    plt.plot(time, hammingw,label="hammingw")
    plt.plot(time, np.exp(-time**2),label="gaus")
    plt.xlabel("Time (s)")
    plt.ylabel("Power")
    plt.legend()
    plt.show()

    # plot the kernel and frequency response
    plt.figure()
    plt.title("Sinc kernel with different window")
    for filti in range(0,len(sincfiltW)):
        plt.plot(time,sincfiltW[filti,:],
                  label=tapernames[filti])
    plt.ylabel('Power')
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Frequency response of window sinc')
    for filti in range(0,len(sincfiltW)):
        pw = np.abs(scipy.fftpack.fft(sincfiltW[filti,:]))
        plt.plot(hz,pw[:len(hz)],label=tapernames[filti])
        plt.xlim([f-3,f+10])
        plt.yscale('log')
        
    plt.plot([f,f],[0,1],'r--')
    plt.ylabel("Filter gain (dB)")
    plt.xlabel("Frequency (hz)")
    plt.legend()
    plt.show()



def filter_data_with_sinc_window_filter_example():
    """

    **Example of the windowed sinc low pass filter on real signal:**

    .. image: _static/images/filtering/Sinc_signal_filtered.png

    .. image: _static/images/filtering/sincw_frequency_response_on_real_signal.png

    """
    ## apply the filter to noise
    
    # simulation params
    srate = 1000
    time  = np.arange(-4,4,1/srate)
    pnts  = len(time)
    hz = np.linspace(0,srate/2,int(np.floor(pnts/2)+1))
    
    # create sinc function
    f = 5
    sincfilt = np.sin(2*np.pi*f*time) / time

    # adjust NaN and normalize filter to unit-gain
    sincfilt[~np.isfinite(sincfilt)] = np.max(sincfilt)
    sincfilt = sincfilt/np.sum(sincfilt)

    # generate data as integrated noise
    data = np.cumsum( np.random.randn(pnts) )

    # reflection
    datacat = np.concatenate( (data,data[::-1]) ,axis=0)

    # apply filter (zero-phase-shift)
    dataf = signal.lfilter(sincfilt,1,datacat)
    dataf = signal.lfilter(sincfilt,1,dataf[::-1])

    # flip forwards and remove reflected points
    dataf = dataf[-1:pnts-1:-1]

    # compute spectra of original and filtered signals
    powOrig = np.abs(scipy.fftpack.fft(data)/pnts)**2
    powFilt = np.abs(scipy.fftpack.fft(dataf)/pnts)**2


    # plot
    plt.figure()
    plt.title('singal filtered')
    plt.plot(time,data,label='Original')
    plt.plot(time,dataf,label='Windowed-sinc filtred (hanning)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


    # plot original and filtered spectra
    plt.figure()
    plt.title("Frequency response")
    plt.plot(hz,powOrig[:len(hz)],label='Original')
    plt.plot(hz,powFilt[:len(hz)],label='Windowed-sinc filtred')
    plt.xlim([0,40])
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # sinc_filter_example()
    # sinc_window_filter_example()
    filter_data_with_sinc_window_filter_example()





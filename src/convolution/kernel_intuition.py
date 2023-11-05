import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import scipy.io as sio
import copy
import pylab as pl
import time


def convolution_intuition_kernel_size():
    """

    **Effect of the kernel size**

    **1. Large gaussian width**

    .. image:: _static/images/convolution/convolution_first_intuition_large_kernel.png


    **2. Narrow gaussian width**

    .. image:: _static/images/convolution/convolution_first_intuition_narrow_kernel.png


    """
    ## first example to build intuition
    signal1 = np.concatenate( (np.zeros(30),np.ones(2),np.zeros(20),np.ones(30),2*np.ones(10),np.zeros(30),-np.ones(10),np.zeros(40)) ,axis=0)
    kernel  = np.exp( -np.linspace(-2,2,20)**2/0.001)
    kernel  = kernel/sum(kernel)
    N = len(signal1)

    plt.figure()
    plt.subplot(311)
    plt.plot(kernel,'k')
    plt.xlim([0,N])
    plt.title('Kernel')

    plt.subplot(312)
    plt.plot(signal1,'k')
    plt.xlim([0,N])
    plt.title('Signal')

    plt.subplot(313)
    plt.plot( np.convolve(signal1,kernel,'same') ,'k')
    plt.xlim([0,N])
    plt.title('Convolution result')

    plt.show()



def convolution_intuition_kernel_mean_value():
    """

    **Effect of the kernel mean_value**


    **1. Kernel mean value is positive**

    .. raw:: html

        <video width="800" height="360" controls>
            <source src="_static/images/convolution/kernel_intuiition_positive.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>

    
        
    **2. Kernel mean value is equal0**

    .. raw:: html
    
        <video width="800" height="360" controls>
            <source src="_static/images/convolution/kernel_intuiition_0.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>

    
        
    **3. Kernel mean value is negative**

    .. raw:: html
    
        <video width="800" height="360" controls>
            <source src="_static/images/convolution/kernel_intuiition_negative.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>

    """
    return


if __name__ == "__main__":
    convolution_intuition_kernel_size()
    # convolution_intuition_kernel_mean_value()
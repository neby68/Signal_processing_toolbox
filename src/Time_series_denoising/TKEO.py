import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
import os



def TKEO(sig: np.ndarray) -> np.ndarray:
    """Teager kaisor operator on a given signal

    Args:
        sig (np.ndarray): signal to filter

    Returns:
        np.ndarray: filterd signal
    """
    filtered_sig = sig.copy()
    for i in range(1, len(sig)-2):
        filtered_sig[i] = sig[i]**2 - (sig[i-1]*sig[i+1])

    #TODO replace by vector operation for performance optimisation
    # filtered_sig2 = sig.copy()
    # filtered_sig2[1:-1] = sig[1:-1]**2 - (sig[2:]*sig[0:-2])
    return filtered_sig


def TKEO_example():
    """
    Teager kaisor operator example

    Context:
        EMG data is often noisy,
        We want to focus on a particular events that append.
    Method:
        TKEO will amplify/highly phases with high energy changes.

    .. image:: _static/images/TimeSeriesDenoising/TKEO.png

    
    """
    #%%import signal
    data_path = os.path.join(os.path.dirname(__file__), "data")
    emgdata= sio.loadmat(os.path.join(data_path, 'emg4TKEO.mat'))

    sig = emgdata["emg"].T
    time = emgdata["emgtime"]
    fs = emgdata["fs"]

    #filtering
    filtered_sig = TKEO(sig)

    #%%plot
    plt.figure()
    plt.plot(sig/max(sig), label="sig")
    plt.plot(filtered_sig/max(filtered_sig), label="filtered sig")#normalized the vector as TKEO is the square of the sig
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude or energy')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude or energy (normalised unit)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    TKEO_example()

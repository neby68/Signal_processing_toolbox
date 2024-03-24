import numpy as np
import scipy
import matplotlib.pyplot as plt



def linear_detrending(sig: np.ndarray) -> np.ndarray:
    """linear detrending function

    Args:
        sig (np.ndarray): signal to be detrended

    Returns:
        np.ndarray: detrended signal
    """
    return scipy.signal.detrend(sig)



def linear_detrending_example():
    """example of linear detrending usage

    Remove the general trend of the data.

    The iIdea is to fit a line that represent the golbal trend of the signal (green) and remove it from the signal. 
    The mean value of the signal after detrending should be 0

    .. image:: _static/images/TimeSeriesDenoising/linear_detrending.png


    """
    #create signal
    n = 2000
    y_max = 50
    y_min = 0
    sig = np.cumsum(np.random.randn(n)) + np.linspace(y_min, y_max,n)

    detrended_sig = linear_detrending(sig)

    plt.figure()
    plt.title('linear detrending')
    plt.plot(sig, color ="green", label="sig")
    plt.plot(np.linspace(y_min, y_max,n), "--", color ="green", label="trend of sig" )
    plt.plot(detrended_sig, color ="orange", label="detrended sig")
    plt.plot(np.linspace(0,0,n), "--", color ="orange", label="trend of detrended sig = 0" )
    plt.xlabel('Indexes')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()
    # print('end')

if __name__ == "__main__":
    linear_detrending_example()
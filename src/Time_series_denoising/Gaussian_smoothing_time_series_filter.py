import numpy as np
import scipy
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import plotly.express as px
import os

def gaussian_smoothing_filter(signal: np.ndarray, 
                              s_rate: int=1000, k: int = 40, fwhm: int = 25
                              ) -> Tuple[np.ndarray, int, int] :
    """gaussian smoothing filter

    Args:
        signal (np.ndarray): signal to be filtered
        s_rate (int, optional): sample frequancy. Defaults to 1000.
        k (int, optional): half window size. Defaults to 40.
        fwhm (int, optional): full width at half mawimum of the gaussian function. Defaults to 25.

    Returns:
        Tuple[np.ndarray, int, int]:
            filtered_sig : signal filtered
            g : gaussian function used
            gtime : gaussian timestamps
    """

    n = len(signal)
    
    #create gaussian function
    gtime = np.arange(-k,k+1)*1000/s_rate
    g = np.exp( -(4*np.log(2)*gtime**2) / (fwhm**2))

    #normalise gaussian to unit energy --> usefull to not shift the signal afterwards
    g_normalized = g/sum(g)

    #Filtering
    filtered_sig = signal.copy()
    for i in range(k+1, n-k-1):
        filtered_sig[i] = sum(signal[i-k:i+k+1]*g_normalized)

    return filtered_sig, g




def gaussian_smoothing_filter_example():
    """Example of the gaussian smoothing filter

    Two main faktor influence the gaussian smoothing filter:
        - the full width at half maximum (fwhm)
            - influene the width of the guassian
        - the half window size (k)
            - evenly split between right and left
            - length of the window is therefore always an odd number
            - influence the number of indexes of the gaussian kernel

    The goal is to find the a good ratio between k and fwhm,
    knowing that k is also the sample window for filtering 
    (the bigger k is the smoother the signal would be),
    so that the kurve look like a bell and to not have too much near 0 values on the right and left side

    **The fwhm is higlighted in the figure below:**
    
    .. image:: _static/images/TimeSeriesDenoising/Gaussian_fwhm_example.png

    The next two figures highlight the influence of the fwhm and k on:
        1. the gaussian shape (=gaussian kernel)
        2. and filtering of a example noisy signal

    .. raw:: html

        <!-- include the contents of the HTML file -->
        <iframe src="_static/images/TimeSeriesDenoising/gaussian_examples_k_fwhm.html" width="900" height="600"></iframe>

    .. raw:: html

        <!-- include the contents of the HTML file -->
        <iframe src="_static/images/TimeSeriesDenoising/Runnig_gaussian_filter_k_fwhm.html" width="900" height="600"></iframe>
    """
    #%%create signal
    s_rate = 1000 
    time = np.arange(0,1,1/1000)
    n = len(time)
    p = 15

    #noise level, measured in standard deviation
    noise_amp = 5

    #amplitude modulator and noise level
    ampl = np.interp(np.linspace(0,p,n),np.arange(0,p),np.random.rand(p)*30)
    noise = noise_amp* np.random.rand(len(time))
    signal = ampl + noise


    #%%Create Gaussian function

    #full-width half-maximum: the key Gaussian parameter
    fwhm = 25

    #centered normalised time vector in ms
    k = 40
    df_sig = pd.DataFrame()
    df_sig["raw_signal"] = signal
    df_gaussian = pd.DataFrame()
    df_gaussian = pd.DataFrame(index=np.arange(-100,101,1))
    fwhm_dict = {"min": {},
                 "max": {},
                 "value": {}}

    # fig = px.
    for k in [5, 50, 100]:
        for fwhm_coef in [0.3, 0.6, 0.9]:
            fwhm = k*fwhm_coef
            fsig, g = gaussian_smoothing_filter(signal, s_rate, k = k, fwhm = fwhm)
            df_gaussian[f"gaussian_k_{k}_fwhm_{fwhm}"] = np.nan
            df_gaussian.loc[0-k:0+k, f"gaussian_k_{k}_fwhm_{fwhm}"] = g
            df_sig[f"f_sig_k_{k}_fwhm_{fwhm}"] = fsig
            
            # # #determine experimentale fwhm
            # half_time = int(len(g)/2)
            # fwhm_dict["min"]["k_{k}_fwhm_{fwhm}"] = abs(g[:half_time] - 0.5).argmin()
            # fwhm_dict["max"]["k_{k}_fwhm_{fwhm}"] = half_time + abs(g[half_time:] - 0.5).argmin()
            # fwhm_dict["value"]["k_{k}_fwhm_{fwhm}"] = (fwhm_dict["min"]["k_{k}_fwhm_{fwhm}"]
            #                                            - fwhm_dict["max"]["k_{k}_fwhm_{fwhm}"])/s_rate

    # Ajouter des lignes horizontales

    fig = px.line(df_gaussian, labels={'x': 'Indexes', 'y': 'gaussian'}, 
                  title="Gaussian with different k and fwhm")
    
    fig2 = px.line(df_sig, labels={'x': 'Indexes', 'y': 'Signal'}, 
                  title="Signal filtered with guassian smmoothing filter varying k and fwhm")

    # #store html plot
    # fig.write_html(os.path.join( os.path.dirname(__file__), 
    #                             r"../../docs/sphinx/_static/images/TimeSeriesDenoising/gaussian_examples_k_fwhm.html"))
    # fig2.write_html(os.path.join( os.path.dirname(__file__), 
    #                         r"../../docs/sphinx/_static/images/TimeSeriesDenoising/Runnig_gaussian_filter_k_fwhm.html"))


    # #plot gaussian
    # plt.figure()
    # plt.title(f"Gaussian with k: {k} and fwhm: {fwhm}")
    # plt.plot(gtime, g)
    # plt.hlines(g[fwhm_exp_x1], gtime[fwhm_exp_x1], gtime[fwhm_exp_x2], color ="purple",label=f"fwhm {fwhm_exp}")
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Gain')
    # plt.legend()
    # plt.show()


    # #%%plot
    # plt.figure()
    # plt.plot(signal, label = "raw signal")
    # plt.plot(filtered_sig, label = "filtered signal")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    gaussian_smoothing_filter_example()


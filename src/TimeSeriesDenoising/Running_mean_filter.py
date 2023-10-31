import numpy as np
import scipy
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import plotly.express as px


def running_mean_filter(signal: np.ndarray, k: int=20) -> np.ndarray:
    """runnign_mean_filter

    Args:
        signal (np.ndarray): signal to filter
        k (int, optional): half window size. Defaults to 20.
    Returns:
        np.ndarray: filtered signal
    """
    filtered_sig = signal.copy()
    n = len(signal)
    for i in range(k+1, n-k-1):
        filtered_sig[i] = np.mean(signal[i-k:i+k])
    return filtered_sig






def running_mean_filter_example():
    """
    example of running mean filter.
    
    One faktor influence the running mean filter:
        - the half window size (k)
            - evenly split between right and left
            - length of the window is therefore always an odd number
            - influence the number of indexes of the gaussian kernel
    
    **The effect of k is highlighted on the figure below:**

    .. raw:: html

       <!-- include the contents of the HTML file -->
       <iframe src="_static/images/TimeSeriesDenoising/Runnig_mean_filter.html" width="900" height="600"></iframe>
       
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

    
    df_signal = pd.DataFrame()
    df_signal["raw_signal"] = signal
    #%%filtering
    for k in [2, 5, 10, 50, 100]:   
        df_signal[f"fsignal k={k}"] = running_mean_filter(signal, k=k)


    #%%plot
    fig = px.line(df_signal, labels={'x': 'Indexes', 'y': 'Signal'})
    # fig.write_html("Runnig_mean_filter.html")

    print('end')


if __name__ =="__main__":
    running_mean_filter_example()
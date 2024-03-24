import numpy as np
import matplotlib.pyplot as plt




def median_filter_for_outliers(sig: np.ndarray, outlier_arr: np.ndarray,
                                k: int=20) -> np.ndarray:
    """filter signals outliers using median value on a given window

    Args:
        sig (np.ndarray): signal to be filtered
        k (int, optional): half window size. Defaults to 20.

    Returns:
        np.ndarray: filtered signal
    """
    #TODO replace copy for performance optimisation
    n = len(sig)
    filtered_sig = sig.copy()
    for i, _ in enumerate(outlier_arr):
        lower_lim = max(1, outlier_arr[i]-k)
        upper_lim = min(outlier_arr[i]+k, n)
        filtered_sig[outlier_arr[i]] = np.median(sig[lower_lim : upper_lim])
    return filtered_sig



def median_filter_for_outliers_example():
    """example of filtering signal outliers using median value on a given window
    Median filter is less sensisitive to outlier - unusually high/low values., than mean/gaussian filtering

    Median filter is a nonlinear filter
    It should be applied on selected data points and not on all data points
    → i.e define a threshold and replace all the value above it with a median value

    .. image:: _static/images/TimeSeriesDenoising/Median_filter.png

    """
    #%%create signal
    n=2000
    sig = np.cumsum(np.random.randn(n))




    #%%noise

    #proportion of noise
    prop_noise = 0.05

    #choose randomly some points with the above given proportion
    noise_points = np.random.permutation(n)[:int(n*prop_noise)]

    #replace those points with noisy values
    sig[noise_points] = 50 + np.random.rand(len(noise_points))*100

    plt.figure()
    plt.hist(sig, 50)

    #define a threshold
    threshold = 45
    outlier_arr = np.where(sig>threshold)[0]
    filtered_sig = median_filter_for_outliers(sig, outlier_arr, k=20)

    plt.figure()
    plt.title("Example of a median filter")
    plt.hlines(threshold,0, n, label="threshold", linestyles="dashed", color="green")
    plt.plot(sig, label="sig")
    plt.plot(filtered_sig, "--", label="sig filtered")
    plt.xlabel('Indexes')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    median_filter_for_outliers_example()
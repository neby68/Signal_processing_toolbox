import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def generate_chirp_signal(pnts: int, srate: int, chirp_type: str = 'bipolar', k: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a chirp signal.
    A chirp signal is a frequency modulate wave.

    .. image:: _static/images/time_series_noise_stimultation/chirps.png

    Args:
        pnts (int): Number of data points.
        srate (int): Sampling rate.
        chirp_type (str, optional): Type of chirp signal ('bipolar' or 'multipolar'). Default is 'bipolar'.
        k (int, optional): Number of poles for frequencies in case of 'multipolar' chirp. Default is 10.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Time vector, instantaneous frequency, and generated chirp signal.
    """
    time = np.arange(pnts) / srate

    if chirp_type == 'bipolar':
        freqmod = np.linspace(5, 15, pnts)
    elif chirp_type == 'multipolar':
        rdm = np.random.rand(k)
        freqmod = 20 * np.interp(np.linspace(5, 15, pnts), np.linspace(5, 15, k), rdm)
    else:
        raise ValueError("Invalid chirp_type. Use 'bipolar' or 'multipolar'.")

    cumulative_freqmod = np.cumsum(freqmod)
    signal = np.sin(2 * np.pi * (time + cumulative_freqmod) / srate)

    return time, freqmod, signal



def plot_chirp_signal(time: np.ndarray, freqmod: np.ndarray, signal: np.ndarray) -> None:
    """
    Plot the chirp signal.

    Args:
        time (np.ndarray): Time vector.
        freqmod (np.ndarray): Instantaneous frequency vector.
        signal (np.ndarray): Chirp signal vector.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    plt.subplot(211)
    plt.plot(time, freqmod, 'r', linewidth=3)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Instantaneous frequency')

    plt.subplot(212)
    plt.plot(time, signal, 'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (a.u.)')
    plt.title('Chirp Signal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Simulation details
    pnts = 10000
    srate = 1024

    # Generate and plot bipolar chirp signal
    time, freqmod, signal = generate_chirp_signal(pnts, srate, chirp_type='bipolar')
    plot_chirp_signal(time, freqmod, signal)

    # Generate and plot multipolar chirp signal
    time, freqmod, signal = generate_chirp_signal(pnts, srate, chirp_type='multipolar', k=10)
    plot_chirp_signal(time, freqmod, signal)
    print('end')

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union

def simulate_eeg_data(pnts: int = 4567, srate: int = 987, peakfreq: int = 14, fwhm: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate EEG data with non-stationary narrowband activity via filtered noise.

    Parameters:
    - pnts (int): Number of data points.
    - srate (int): Sampling rate.
    - peakfreq (int): Peak frequency of the narrowband activity.
    - fwhm (int): Full width at half maximum of the narrowband activity.

    Returns:
    - hz (numpy.ndarray): Frequency values.
    - signal (numpy.ndarray): Simulated EEG data.
    """

    # frequencies
    hz = np.linspace(0, srate, pnts)

    # create frequency-domain Gaussian
    s = fwhm * (2 * np.pi - 1) / (4 * np.pi)  # normalized width
    x = hz - peakfreq  # shifted frequencies
    fg = np.exp(-0.5 * (x / s) ** 2)  # gaussian

    # Fourier coefficients of random spectrum
    fc = np.random.rand(pnts) * np.exp(1j * 2 * np.pi * np.random.rand(pnts))

    # taper with Gaussian
    fc = fc * fg

    # go back to time domain to get EEG data
    signal = 2 * np.real(np.fft.ifft(fc))

    # plotting
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(hz, np.abs(fc), 'k')
    plt.xlim([0, peakfreq * 3])
    plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude (a.u.)')
    plt.title('Frequency domain')

    plt.subplot(212)
    plt.plot(np.arange(pnts) / srate, signal, 'b')
    plt.title('Time domain')
    plt.xlabel('Time (s)'), plt.ylabel('Amplitude')

    plt.show()

    return hz, signal


if __name__ == "__main__":
    # Example usage:
    pnts = 100
    srate = 20
    peakfreq = 10
    fwhm = 5
    frequency_values, simulated_eeg_data = simulate_eeg_data(
                                                            pnts=pnts, 
                                                            srate=srate, 
                                                            peakfreq=peakfreq, 
                                                            fwhm=fwhm
                                                            )
    print('end')

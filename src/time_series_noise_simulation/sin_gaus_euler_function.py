import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_sine_wave(freq: float, srate: int, ampl: float, phas: float) -> None:
    r"""
    Plots a sine wave.

    
    Sin wave formula:
        
        .. math::
            a\sin({2 \pi f t + \theta})

    with:
        - a : amplitude
        - f : frequency
        - t : time
        - theta : phase shift
    
    Args:
        freq (float): Frequency of the sine wave in Hz.
        srate (int): Sampling rate in Hz.
        ampl (float): Amplitude of the sine wave.
        phas (float): Phase of the sine wave in radians.

    Returns:
        None
    """
    time = np.arange(-1, 1 + 1/srate, 1/srate)
    sinewave = ampl * np.sin(2 * np.pi * freq * time + phas)

    plt.figure()
    plt.plot(time, sinewave)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])
    plt.xlabel('Time (s)')
    plt.title('Sine Wave Plot')
    plt.show()


def plot_sum_of_sine_waves(frex: List[float], amplit: List[float], phases: List[float]) -> None:
    r"""
    Plots the sum of multiple sine waves.

    Args:
        frex (list[float]): List of frequencies for each sine wave.
        amplit (list[float]): List of amplitudes for each sine wave.
        phases (list[float]): List of phases for each sine wave in radians.

    Returns:
        None
    """
    srate = 1000
    time = np.arange(-1, 1 + 1/srate, 1/srate)
    sine_waves = np.zeros((len(frex), len(time)))

    for fi in range(len(frex)):
        sine_waves[fi, :] = amplit[fi] * np.sin(2 * np.pi * time * frex[fi] + phases[fi])

    plt.figure()
    plt.plot(time, np.sum(sine_waves, axis=0))
    plt.title('Sum of Sine Waves')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (arb. units)')
    plt.show()


def plot_each_sine_wave(frex: List[float], amplit: List[float], phases: List[float]) -> None:
    r"""
    Plots each individual sine wave.

    Args:
        frex (list[float]): List of frequencies for each sine wave.
        amplit (list[float]): List of amplitudes for each sine wave.
        phases (list[float]): List of phases for each sine wave in radians.

    Returns:
        None
    """
    srate = 1000
    time = np.arange(-1, 1 + 1/srate, 1/srate)

    plt.figure()
    for fi in range(len(frex)):
        plt.subplot(len(frex), 1, fi+1)
        plt.plot(time, amplit[fi] * np.sin(2 * np.pi * time * frex[fi] + phases[fi]))
        plt.axis([time[0], time[-1], -max(amplit), max(amplit)])
    plt.show()


def plot_gaussian(ptime: float, ampl: float, fwhm: float) -> None:
    r"""
    Plots a Gaussian curve.

    Gaus formula:
    
    .. math::
        a e^{\frac{- (t-m)^2}{2 s^2}}

    with:
        - m : time point at peak
        - t : time
        - s : width

    .. math::
        a e^{\frac{-4 \ln{2} t^2}{fwhm^2}}

            
    with:
        - fwhm : full width at half maximum (s)
        - t : time

    fwhm is a more more easy tunable and understainable parameter to tun guassian function as s


    Args:
        ptime (float): Peak time of the Gaussian curve.
        ampl (float): Amplitude of the Gaussian curve.
        fwhm (float): Full-width at half-maximum of the Gaussian curve.

    Returns:
        None
    """
    time = np.arange(-2, 2 + 1/1000, 1/1000)
    gwin = ampl * np.exp(-(4 * np.log(2) * (time - ptime)**2) / fwhm**2)

    gwinN = gwin / max(gwin)
    midp = np.argmin(np.abs(time - 0))
    pst5 = midp - 1 + np.argmax(gwinN[midp:])
    pre5 = np.argmax(gwinN[:midp])
    empfwhm = time[pst5] - time[pre5]

    plt.figure()
    plt.plot(time, gwin, 'k', linewidth=2)
    plt.plot(time[[pre5, pst5]], gwin[[pre5, pst5]], 'ro--', markerfacecolor='k')
    plt.plot(time[[pre5, pre5]], [0, gwin[pre5]], 'r:')
    plt.plot(time[[pst5, pst5]], [0, gwin[pst5]], 'r:')
    plt.title(f'Requested FWHM: {fwhm}s, empirical FWHM: {empfwhm}s')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def plot_eulers_formula(M: float, k: float) -> None:
    r"""
    Plots Euler's formula.

    .. math::
        M e^{i k } = M(\cos{k} + i\sin{k})

    with:
        - M : distance from the origin
        - k : angle in respect to the positive real axis
        

    Args:
        M (float): Magnitude of the complex number.
        k (float): Phase angle of the complex number in radians.

    Returns:
        None
    """
    meik = M * np.exp(1j * k)

    plt.figure(figsize=(10, 5))

    # Polar plane
    plt.subplot(121, projection='polar')
    plt.polar([0, np.angle(meik)], [0, np.abs(meik)], 'r')
    plt.polar(np.angle(meik), np.abs(meik), 'ro')
    plt.title('Polar plane')

    # Cartesian (rectangular) plane
    plt.subplot(122)
    plt.plot(np.real(meik), np.imag(meik), 'ro')
    plt.plot([0, np.real(meik)], [0, np.imag(meik)], 'gs')
    # plt.axis([-1, 1, -1, 1] * np.abs(meik))
    plt.axis('square')
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.grid(True)
    plt.title('Cartesian (rectangular) plane')

    plt.show()


if __name__ == "__main__":
    # Test each function
    freq = 2
    srate = 1000
    ampl = 2
    phas = np.pi/3
    plot_sine_wave(freq, srate, ampl, phas)

    frex = [3, 10, 5, 15, 35]
    amplit = [5, 15, 10, 5, 7]
    phases = [np.pi/7, np.pi/8, np.pi, np.pi/2, -np.pi/4]
    plot_sum_of_sine_waves(frex, amplit, phases)

    plot_each_sine_wave(frex, amplit, phases)

    ptime = 1
    ampl = 45
    fwhm = 0.9
    plot_gaussian(ptime, ampl, fwhm)

    M = 2.4
    k = 3 * np.pi/4
    plot_eulers_formula(M, k)

import numpy as np
import matplotlib.pyplot as plt


def type_of_noise_plot():
    """
    Example of different noise types:
        1. White noise:
            Equal power across all frequencies.
        2. Pink noise:
            Power decreases at higher frequencies.
        3. Brown noise:
            Power decreases rapidly at higher frequencies.

    .. image:: _static/images/noise/noise_type_description.png

    
    """
    # Création des fréquences
    hz = np.linspace(0, 1, 1000)

    # Création des spectres de bruit
    white_noise = np.ones_like(hz)
    pink_noise = 1 / np.sqrt(hz)
    brown_noise = 1 / hz

    # Tracé du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(hz, white_noise, label='White Noise')
    plt.plot(hz, pink_noise, label='Pink Noise')
    plt.plot(hz, brown_noise, label='Brown Noise')

    plt.xscale('log')
    plt.yscale('log')

    plt.title('Spectral Density of Different Noises')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    type_of_noise_plot()
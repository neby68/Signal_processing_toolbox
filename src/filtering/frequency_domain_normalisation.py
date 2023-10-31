import numpy as np
import matplotlib.pyplot as plt

# Paramètres du signal
srate = 1000  # Taux d'échantillonnage en Hz
time = np.arange(0, 1, 1/srate)  # Temps de 0 à 1 seconde
freq = 5  # Fréquence du signal en Hz
signal = np.sin(2 * np.pi * freq * time)  # Signal sinusoïdal pur

# Calcul de la FFT
fft_result = np.fft.fft(signal)
frequency = np.fft.fftfreq(len(signal), d=1/srate)

# Normalisation par la longueur du signal
magnitude_normalized = np.abs(fft_result) / len(signal)

# Normalisation sans la longueur du signal
magnitude_unnormalized = np.abs(fft_result)

# Magnitude au carré
power_spectrum = np.abs(fft_result)**2

# Tracé des graphiques
plt.figure(figsize=(15, 10))

# Signal temporel
plt.subplot(4, 1, 1)
plt.plot(time, signal)
plt.title('Signal Temporel')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')

# FFT avec normalisation par la longueur du signal
plt.subplot(4, 1, 2)
plt.plot(frequency, magnitude_normalized)
plt.title('FFT avec Normalisation par la Longueur du Signal')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Magnitude Normalisée')

# FFT sans normalisation par la longueur du signal
plt.subplot(4, 1, 3)
plt.plot(frequency, magnitude_unnormalized)
plt.title('FFT sans Normalisation par la Longueur du Signal')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Magnitude Non Normalisée')

# FFT avec magnitude au carré (puissance spectrale)
plt.subplot(4, 1, 4)
plt.plot(frequency, power_spectrum)
plt.title('FFT avec Magnitude au Carré (Puissance Spectrale)')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Puissance')

plt.tight_layout()
plt.show()

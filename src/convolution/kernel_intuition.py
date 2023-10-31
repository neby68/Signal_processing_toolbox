import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Fonctions de convolution et de multiplication en fréquence
def convolution(f, g, dt):
    h = np.convolve(f, g, mode='full') * dt
    return h[:len(f)]

def plot_convolution_theorem():
    # Créer des signaux et leur convolution
    t = np.linspace(-5, 5, 1000)
    dt = t[1] - t[0]
    f = np.exp(-t**2)
    g = np.sinc(t)

    h = convolution(f, g, dt)

    # Transformées de Fourier
    F = np.fft.fft(f) * dt
    G = np.fft.fft(g) * dt
    H = F * G

    # Tracé
    plt.figure(figsize=(12, 8))

    # Signaux temporels
    plt.subplot(321)
    plt.plot(t, f, label='$f(t)$', color='blue')
    plt.plot(t, g, label='$g(t)$', color='green')
    plt.title('Time Domain Signals')
    plt.legend()

    # Convolution temporelle
    plt.subplot(322)
    plt.plot(t, h, label='$(f * g)(t)$', color='red')
    plt.title('Convolution in Time Domain')
    plt.legend()

    # Transformée de Fourier des signaux
    plt.subplot(323)
    plt.plot(np.fft.fftfreq(len(t), dt), np.abs(F), label='$\mathcal{F}\{f(t)\}$', color='blue')
    plt.plot(np.fft.fftfreq(len(t), dt), np.abs(G), label='$\mathcal{F}\{g(t)\}$', color='green')
    plt.title('Frequency Domain Signals')
    plt.legend()

    # Multiplication en fréquence
    plt.subplot(324)
    plt.plot(np.fft.fftfreq(len(t), dt), np.abs(H), label='$\mathcal{F}\{(f * g)(t)\}$', color='red')
    plt.title('Multiplication in Frequency Domain')
    plt.legend()

    # Nyquist Frequency
    nyquist_freq = 1 / (2 * dt)
    plt.subplot(325)
    plt.plot(np.fft.fftfreq(len(t), dt), np.abs(F), label='$\mathcal{F}\{f(t)\}$', color='blue')
    plt.plot(np.fft.fftfreq(len(t), dt), np.abs(G), label='$\mathcal{F}\{g(t)\}$', color='green')
    plt.axvline(x=nyquist_freq, color='purple', linestyle='--', label='Nyquist Frequency')
    plt.title('Nyquist Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Appeler la fonction pour générer le schéma
plot_convolution_theorem()
print('end')

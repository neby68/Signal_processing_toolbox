import numpy as np
import scipy
import matplotlib.pyplot as plt



def time_synchronous_average_example():
    # create en event
    event_length = 100
    event = np.diff(np.exp( -np.linspace(-2, 2, event_length+1)**2))
    event = event/max(event)

    plt.figure()
    plt.title("single event")
    plt.plot(event)

    #create a signal by adding multiple event and noise
    signal_length = 10000
    nb_event = 30
    idx_arr = np.random.randint(0, high=signal_length-event_length, size=signal_length-event_length)[0:event_length]

    signal = np.zeros(signal_length)
    for n in range(0, nb_event):
        signal[idx_arr[n]: idx_arr[n]+event_length] = event

    signal += np.random.rand(signal_length)

    plt.figure()
    plt.title("raw signal")
    plt.plot(signal)

    #store all the events with noise in a matrix
    event_matrix = np.zeros((nb_event, event_length))
    for n in range(0, nb_event):
        event_matrix[n,:] = signal[idx_arr[n]: idx_arr[n]+event_length]

    #averaging all the events to reduce noises
    time_synchronous_averaging = np.mean(event_matrix, axis=0)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title(f"{nb_event} events")
    plt.plot(event_matrix)
    plt.subplot(2, 1, 2)
    plt.title(f"average of the events")
    plt.plot(time_synchronous_averaging)
    plt.show()

    print('end')


# if __name__ == "__main__":
#     time_synchronous_average_example()

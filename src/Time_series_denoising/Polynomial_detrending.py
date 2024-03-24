import numpy as np
import scipy
import matplotlib.pyplot as plt



def polynomial_detrending_with_bayes_criterion(signal: np.ndarray, order_min:int=2,
                                               order_max:int=25) -> np.ndarray:
    """polynomial detrending. Polynomial order defined based on bayes criterion 

    Args:
        signal (np.ndarray): signal to be detrended
        order_min (int, optional): min order of the polynomial. Defaults to 2.
        order_max (int, optional): max order of the polynomial. Defaults to 25.

    Returns:
        np.ndarray: detrended signal
    """
    n = len(signal) 
    t = np.arange(0,n)
    orders = np.arange(order_min, order_max)
    epsilon_arr = np.zeros(len(orders))

    for i, order in enumerate(orders):
        y = np.polyval(np.polyfit(t, signal, order), t)
        epsilon_arr[i] = sum( (y - signal)**2)/n

    #  Bayes information criterion
    bic = n*np.log(epsilon_arr) + orders*np.log(n)
    best_order = orders[np.argmin(bic)]

    plt.figure()
    plt.title("bayes criterion")
    plt.plot(orders, bic, label="order")
    plt.scatter(best_order, bic[best_order], color='red', label="best order")
    plt.xlabel("order")
    plt.ylabel("bic")
    plt.legend()
    plt.show()

    polynomial_coefs = np.polyfit(t, signal, best_order)
    y = np.polyval(polynomial_coefs, t)

    # plt.figure()
    # plt.plot(x_new, signal)
    # plt.plot(x_new, y)
    # plt.plot(x_new, signal - y)

    return signal - y




def polynomial_detrending_with_bayes_criterion_example():
    """


    **Find best polynomial order with Bayes information criterion ():**


    Bayes information criterion (bic):
    Give an information about  how close the y and y_fit are.
    We search for the minimal distance (red point on the figure below)

    Formula:

    .. math::
        bic =n \ln(\epsilon) + k \ln(n)
    
    .. math::
        \epsilon = n^{-1} \sum^{n}_{i=1}(y_{fit, i} - y_i)^2\n
    
    with:\n
    k = polynomial order\n
    y = raw signal\n
    y_fit = predicted/fitted signal (=polynomial)\n
    n = y length\n

        
    .. image:: _static/images/TimeSeriesDenoising/bayes_information_criterion.png
    
    **Detrending:**
        detreding is basically :math:`y - y_{fit}`.

    .. image:: _static/images/TimeSeriesDenoising/polynomial_detrending.png

    """
    #%% create a signal with slow drift and high frequency noises
    n = 10000
    t = np.arange(0,n)
    k = 10
    x = np.linspace(1,n,k)
    x_new = np.linspace(1,n,n)
    slow_drift = scipy.interpolate.interp1d(x, 100*np.random.rand(k), kind="cubic")
    signal = slow_drift(x_new) +20* np.random.rand(n)


    #%%find polynomial by giving the polynomial order
    polynomial_order = 5
    polynomial_coefs = np.polyfit(t, signal, polynomial_order)
    y = np.polyval(polynomial_coefs, t)

    plt.figure()
    plt.title(f"detrending using fix polynomial order : {polynomial_order}")
    plt.plot(x_new, slow_drift(x_new), label="drift")
    plt.plot(x_new, signal, label="signal")
    plt.plot(x_new, y, label = "polynomial")
    plt.plot(x_new, signal-y, label = "detrended signal")
    plt.legend()
    plt.show()


    detrended_sig = polynomial_detrending_with_bayes_criterion(signal, order_min=5, order_max=40)
    
    plt.figure()
    plt.title("detrending using bayes criterion")
    plt.plot(x_new, signal, label="signal")
    plt.plot(x_new, detrended_sig, label="detrended signal")
    plt.xlabel("Indexes")
    plt.ylabel("Signal")
    plt.legend()
    plt.show()


def understand_polynomials_orders():
    """ understand polynomial orders 
    """
    #%% Understand polynomials
    polynomial_order = 3
    x = np.linspace(-15, 15, 100)
    y = np.zeros(len(x))

    for order in range(0, polynomial_order):
        y += np.random.random()*x**order

    plt.figure()
    plt.plot(y)
    plt.show()

if __name__ == "__main__":
    polynomial_detrending_with_bayes_criterion_example()
    print('stop')
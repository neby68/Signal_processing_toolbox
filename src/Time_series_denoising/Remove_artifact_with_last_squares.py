import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import sys

root_path = os.path.join(os.path.dirname(__file__), "../.." )
sys.path.append(root_path)



def remove_artifact_with_last_square_between_two_signals(sig: np.ndarray,
                                                sig_artifact: np.ndarray) -> np.ndarray:
    """remove an artifact from a signal using the last square method.
    The artifact is the sig_artifact.

    Args:
        sig (np.ndarray): signal containing artifact that must be removed
        sig_artifact (np.ndarray): signal artifact

    Returns:
        np.ndarray: signal without artifact
    """
    #build the least-squares model as intercept and EOG from this trial
    X = np.column_stack((np.ones(len(sig)), sig_artifact))
    
    #compute regression coefficients for EEG channel
    beta = ((X.T@X)**(-1))@(X.T@sig)
    # beta = np.linalg.lstsq(X, sig, rcond=None)[0]

    #predicted_data
    predicted_data = X@beta

    residual_data = (sig - predicted_data).T
    return residual_data


def remove_artifact_with_last_square_between_two_signals_example():
    r"""
    Example of removeing an artifact from a signal using the last square method:
    
    Context:
        EEG signal can be conain artifact because of the eyes movement (EOG signal).
    Goal:
        Remove the eye movement artifact

    Method:
        Use the last square template matching to remove the EOG artifact.

        last square template matching:

            .. math::
                \beta = (X^TX)^{-1}X^Ty

            .. math::
                res = y - X\beta

    .. image:: _static/images/TimeSeriesDenoising/Artifact_with_EEG_EOG.png

    **color map of the EOG, EEG, and residual signal:**
    
    .. image:: _static/images/TimeSeriesDenoising/color_map_eeg_eog.png
    
    """
    file_data_path = os.path.join(root_path, r"data\TimeSeriesDenoising\templateProjection.mat")
    if not os.path.isfile(file_data_path):
        print("\n\tFile data could not be found. Please check that you have access to it\n")
        return
    data= sio.loadmat(file_data_path)
    eeg_data = data["EEGdat"] #data 
    eye_data = data["eyedat"] #artifact

    residual_data = np.zeros(eeg_data.shape)

    for i in range(0, residual_data.shape[1]):
        residual_data[:,i] = remove_artifact_with_last_square_between_two_signals(eeg_data[:,i], eye_data[:,i])


    plt.figure()
    plt.title("Removing EOG artifact from EEG signal")
    plt.plot(np.mean(eeg_data, axis=1), label="EEG signal")
    plt.plot(np.mean(eye_data, axis=1), label="EOG signal")
    plt.plot(np.mean(residual_data, axis=1), label="residual")
    plt.xlabel("Indexes")
    plt.ylabel("Signal")
    plt.legend()
    plt.show()

    plt.figure()
    # plt.title("Colormap of EOG artifact, EEG and residual signal")
    plt.subplot(131)
    plt.imshow(eye_data)
    plt.ylabel('Time')
    plt.xlabel("Trial")
    plt.title('EOG')

    plt.subplot(132)
    plt.imshow(eeg_data)
    plt.ylabel('Time')
    plt.xlabel("Trial")
    plt.title('EEG')

    plt.subplot(133)
    plt.imshow(residual_data)
    plt.ylabel('Time')
    plt.xlabel("Trial")
    plt.title('Residual')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    remove_artifact_with_last_square_between_two_signals_example()
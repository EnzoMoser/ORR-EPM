"""
Tyson Reimer
University of Manitoba
June 19th, 2019
"""


import pickle
import numpy as np


###############################################################################


def load_pickle(path):
    """Loads a pickle file

    Parameters
    ----------
    path : str
        The full path to the .pickle file that will be loaded

    Returns
    -------
    loaded_var :
        The loaded variable, can be array_like, int, str, dict_to_save,
        etc.
    """

    with open(path, 'rb') as handle:
        loaded_var = pickle.load(handle)

    return loaded_var


def save_pickle(var, path):
    """Saves the var to a .pickle file at the path specified

    Parameters
    ----------
    var : object
        A variable that will be saved
    path : str
        The full path of the saved .pickle file
    """

    with open(path, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dak_csv(csv_path):
    """Load the permittivity data from a DAK-generated .csv file

    Parameters
    ----------
    csf_path : str
        Path to the DAK-generated .csv file

    Returns
    -------
    fs : array_like
        The frequencies, in [Hz], that were used in the measurement
    perm : array_like
        Complex-valued relative permittivity obtained from the DAK
        measurement
    """

    # Load the .csv file
    raw_data = np.genfromtxt(csv_path,
                             delimiter=',',
                             skip_header=10,
                             dtype=float)

    # Get the frequencies, and convert from [MHz] to [Hz]
    fs = raw_data[:, 0] * 1e6

    # Get the complex permittivity
    perm = raw_data[:, 1] - 1j * raw_data[:, 2]

    return fs, perm

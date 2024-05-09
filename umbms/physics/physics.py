"""
Tyson Reimer
University of Manitoba
July 25th, 2023
"""

import numpy as np

###############################################################################

__EPS_0 = 8.854187e-12  # Vacuum permittivity, in [F/m]
__MU_0 = 1.256637e-6  # Vacuum permeability, in [N/A^2]

###############################################################################


def _perm_cond_to_complex_perm(perm, cond, fs):
    """Convert real relative perm/cond to complex perm

    Parameters
    ----------
    perm : array_like
        Relative real-valued permittivity
    cond : array_like
        Conductivity, in [S/m]
    fs : array_like
        Indexing frequencies, in [Hz]

    Returns
    -------
    complex_perm : array_like
        The complex-valued relative permittivity
    """

    # Calculate the relative complex permittivity
    complex_perm = perm - 1j * cond / (__EPS_0 * 2 * np.pi * fs)

    return complex_perm


def get_reflectance(perm_a, perm_b):
    """Calculate the reflectance, r, for an interface of two materials

    Parameters
    ----------
    perm_a : array_like
        The relative complex-valued permittivities of material A. May be
        indexed by frequency
    perm_b : array_like
        The relative complex-valued permittivities of material B. May be
        indexed by frequency

    Returns
    -------
    reflectance : array_like
        The reflectance, may be indexed by frequency, if inputs are
    """

    # Get the impedence of materials A and B
    nu_a = np.sqrt((__MU_0 / __EPS_0) / perm_a)
    nu_b = np.sqrt((__MU_0 / __EPS_0) / perm_b)

    reflectance = (nu_b - nu_a) / (nu_b + nu_a)  # Calc reflectance

    return reflectance


def get_ref_coeff(perm_a, perm_b):
    """

    Parameters
    ----------
    perm_a : array_like
        The relative complex-valued permittivities of material A. May be
        indexed by frequency
    perm_b : array_like
        The relative complex-valued permittivities of material B. May be
        indexed by frequency

    Returns
    -------
    reflection_coeff : array_like
        The reflection coefficient, may be indexed by frequency,
        if inputs are
    """

    # Calculate the reflection coefficient
    reflection_coeff = np.abs(get_reflectance(perm_a=perm_a,
                                              perm_b=perm_b))**2

    return reflection_coeff


def get_trans_coeff(perm_a, perm_b):
    """

    Parameters
    ----------
    perm_a : array_like
        The relative complex-valued permittivities of material A. May be
        indexed by frequency
    perm_b : array_like
        The relative complex-valued permittivities of material B. May be
        indexed by frequency

    Returns
    -------
    transmission_coeff : array_like
        The transmission coefficient, may be indexed by frequency,
        if inputs are
    """


    # Calculate the transmission coefficient
    transmission_coeff = 1 - get_ref_coeff(perm_a=perm_a, perm_b=perm_b)

    return transmission_coeff


def calc_speed(fs, perms, conds):
    """Calculate propagation speed using permittivity, conductivity

    Parameters
    ----------
    fs : array_like
        Frequencies used to define permittivities and conductivities,
        in [Hz]
    perms : array_like
        Real part of the relative permittivity
    conds : array_like
        Conductivity, in [S/m]

    Returns
    -------
    speed : array_like
        Propagation speed, in [m/s], defined at each frequency in fs
    """

    # Beta parameter
    beta = (2 * np.pi * fs
            * np.sqrt((__MU_0 * __EPS_0 * perms / 2)
                      * (np.sqrt(1
                                 + (conds
                                    / (2 * np.pi * fs * perms * __EPS_0))**2)
                         + 1)
                      )
            )

    speed = 2 * np.pi * fs / beta

    return speed

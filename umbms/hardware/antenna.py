"""
Tyson Reimer
University of Manitoba
November 07, 2018
"""

import os
import numpy as np
from umbms import get_proj_path
from umbms.loadsave import load_pickle

from umbms.recon.extras import get_pix_angs

###############################################################################


_valid_beam_freqs = np.arange(2, 9.1, 0.5)


###############################################################################


def apply_ant_t_delay(pix_ts):
    """Add the antenna time delay to the 1-way pixel time delays

    Parameters
    ----------
    pix_ts : array_like, NxMxL
        Array of the 1-way time-of-response for each pixel, N is
        number of antenna positions, MxL is size of 2D image, units
        of [s]

    Returns
    -------
    cor_ts : array_like, NxMxL
        The 1-way response times for each pixel after accounting
        for the 1-way antenna time delay, units of [s]
    """

    # Add the 1-way antenna time delay, in units of [s]
    # (Value added should be 0.19 ns, based on analysis from May 29,
    # 2023)
    cor_ts = pix_ts + 0.19e-9

    return cor_ts


def to_phase_center(meas_rho):
    """Shift the measured rho of the antenna to its phase center

    Parameters
    ----------
    meas_rho : float
        The measured rho of the antenna; measured from the front edge
        of the antenna stand, in [cm]

    Returns
    -------
    cor_rho : float
        The corrected rho of the antenna, i.e., the rho corresponding
        to the phase center of the antenna, in [cm]
    """

    # Add the length corresponding to the distance from the front edge
    # of the antenna stand to the phase center of the antenna, in [cm]
    # (Value added should be 2.4 cm, based on analysis from May 29,
    # 2023)
    cor_rho = meas_rho + 2.4

    return cor_rho


def get_ant_beam(roi_rho, ant_rho, m_size, n_ant_pos,
                 ini_ant_ang=-136.0, pattern_freq=5.5,
                 use_all_fs=False,
                 all_fs=None):
    """

    Parameters
    ----------
    roi_rho : float
        The radius of the ROI, in [cm]
    ant_rho : float
        The polar radius of the antenna trajectory in the scan,
        corrected to the phase center of the antenna, in [cm]
    m_size : int
        Number of pixels along one dimension of the image space
    n_ant_pos : int
        Number of antenna positions used in the scan
    ini_ant_ang : float
        Initial polar coordinate phi of the antenna, in [deg]
    pattern_freq : float
        The frequency at which the beam pattern will be used.
    use_all_fs : bool
        If True, will return the beam pattern at all frequencies
    all_fs :
        The frequencies at which to calculate the beam pattern

    Returns
    -------
    beam : array_like
        The beam pattern effect at each pixel for one-way propagation
        in linear magnitude.
    """

    # Assert the frequency is valid
    assert pattern_freq in _valid_beam_freqs, "Invalid pattern_freq"

    # Get pixel angles in [deg]
    pix_angs = get_pix_angs(ant_rho=ant_rho, m_size=m_size, roi_rho=roi_rho,
                            n_ant_pos=n_ant_pos, ini_ant_ang=ini_ant_ang)

    if not use_all_fs:  # If using only one frequency

        # Load the interpolated model of the antenna beam pattern
        # (obtained from the antenna data sheet), convert pattern_freq
        # from [GHz] to [MHz]
        interp_model = load_pickle(
            os.path.join(get_proj_path(),
                         'docs/hardware/beam_patterns/%dMHz_H.pickle'
                         % (1000 * pattern_freq)))

        # Get the beam factor in dB
        beam_fac = np.interp(x=pix_angs, xp=interp_model[:, 1],
                             fp=interp_model[:, 0])

        beam_fac = 10**(beam_fac / 10)  # Convert to linear magnitude

    else:  # If using all frequencies

        # Beam frequencies, in [MHz]
        beam_fs = np.arange(start=2000, stop=9001, step=500)

        # Init arr for return
        beam_fac = np.zeros([len(all_fs), n_ant_pos, m_size, m_size])

        # For each target frequency...
        for ii in range(len(all_fs)):

            # Find the closest beam frequency
            f_here = beam_fs[np.argmin(np.abs(all_fs[ii] - beam_fs * 1e6))]

            # Load the model
            interp_model = load_pickle(
            os.path.join(get_proj_path(),
                         'docs/hardware/beam_patterns/%dMHz_H.pickle'
                         % (f_here)))

            # Get the beam and convert from dB to linear magnitude
            beam_fac[ii, :, :, :] = 10**(
                np.interp(x=pix_angs,
                          xp=interp_model[:, 1],
                          fp=interp_model[:, 0])
                / 10
            )

    return beam_fac


def get_ant_gain(scan_fs, normalize=True):
    """

    Parameters
    ----------
    scan_fs : array_like
        Frequencies used in the scan, in [Hz]
    normalize

    Returns
    -------
    gain
    """

    # Assert frequencies are between 2-9 GHz
    assert (np.min(scan_fs) >= 2e9) and (np.max(scan_fs) <= 9e9), \
        "Scan fs not in valid bounds"

    # Convert frequencies from [Hz] to [GHz]
    interp_fs = scan_fs / 1e9

    # Load data from datasheet, used for interpolation
    interp_model = load_pickle(
        os.path.join(get_proj_path(), 'docs/hardware/gain/ant_gain.pickle')
    )

    # Use interpolation to get the gain model
    gain = np.interp(x=interp_fs, xp=interp_model[:, 0], fp=interp_model[:, 1])

    if normalize:
        gain /= np.max(gain)  # Normalize to have max of unity

    return gain

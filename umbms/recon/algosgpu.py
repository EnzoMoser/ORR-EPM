"""
Tyson Reimer
University of Manitoba
April 10th, 2024
"""

import time
import numpy as np

from umbms import null_logger

try:  # Try to import cupy
    import cupy as cp

except ImportError:  # If cant import cupy
    cupy_available = False

from umbms.hardware.antenna import get_ant_beam, get_ant_gain
from umbms.recon.enhphys import sphere_wave_attn


###############################################################################


# List of the valid physics model enhancements
_valid_enh_phys = [
    'spherical',
    'beam',
    'gain',
    'complex_k',
    'speed',
]


###############################################################################

def _orr_gpu(ini_img,
             freqs,
             m_size,
             fd,
             roi_rho,
             phase_fac,
             ant_rho=-10.0,
             step_size=0.3,
             speed=0.0,
             enh_phys=None,
             logger=null_logger,
             ):
    """Perform optimization-based radar reconstruction, via grad desc

    Parameters
    ----------
    ini_img : array_like
        Initial image estimate
    freqs : array_like
        The frequencies used in the scan, in [Hz].
        NOTE: Before passing to this function, the frequency vector
        should be under-sampled. In the bed system, the frequency
        spacing should be <100 MHz.
    m_size : int
        The number of pixels along one dimension of the reconstructed
        image
    fd : array_like
        The measured frequency domain data
    roi_rho : float
        The radius of the reconstructed image-space, in [cm]
    phase_fac : array_like
        Phase factor array for efficient computation
    ant_rho : float, optional
        The polar coordinate rho of the antenna used in the scan,
        after accounting for the phase center of the antenna. Only
        used if physics enhanced modelling is used, in [cm]
    step_size : float, optional
        The step size to use for gradient descent. Changes based on
        the enhanced physics modelling factors included in enh_phys
            gain : 0.3
            spherical : 6e4
            beam : 0.3
            complex_k : 8e1, 1.5e1
            sec_scat :
            complex_k, beam, spherical : 3e6
    enh_phys : list, optional
        List of physics enhancements to be used
    logger :
        Logging object

    Returns
    -------
    img : array_like
        Reconstructed image
    """

    # Assertion message to check length of frequency vector
    fq_msg = ("Frequecy vector too large, len(freqs) should be "
              "less than 100, len(freqs) = %d" % len(freqs))

    # Ensure frequency vector is sufficiently small - this is checking
    # to see if frequency sampling is correct
    assert len(freqs) < 100, "%s" % fq_msg

    # Assertion message to check length if frequency domain data is
    # sufficiently small
    fd_msg = ("fd array too large, shape should be approx [72, 73], "
              "np.size(fd) = %d" % np.size(fd)
              )

    assert np.size(fd) < (100 * 72), "%s" % fd_msg

    tot_t0 = time.time()

    phase_fac = cp.asarray(phase_fac)  # Convert to GPU arr

    # Init beam, gain, and spherical scattering factors
    beam = cp.ones_like(phase_fac)
    gain = cp.ones([len(freqs), ])
    spherical = cp.ones(np.shape(phase_fac))

    if enh_phys is not None:  # If *using* some physics enhancements...

        # Assert that the antenna rho is set (only used for
        # enhanced physics modelling)
        assert ant_rho != -10.0, "enh_phys used but ant_rho not defined"

        for ii in enh_phys:  # Ensure each enhancement is valid
            assert ii in _valid_enh_phys, \
                "%s not valid enh phys, must be in %s" % (ii, _valid_enh_phys)

        if 'beam' in enh_phys:  # If using beam model...

            # Get the beam model - the antenna beam effect at each pixel
            # *for each frequency*
            beam = get_ant_beam(roi_rho=roi_rho,
                                ant_rho=ant_rho,
                                m_size=m_size,
                                n_ant_pos=np.size(fd, axis=1),
                                use_all_fs=True,
                                all_fs=freqs)

            beam = cp.asarray(beam)  # Convert to GPU array

        if 'gain' in enh_phys:  # If using gain model...

            # Get the antenna gain factor
            gain = get_ant_gain(scan_fs=freqs)

            gain = cp.asarray(gain)  # Convert to GPU array

        # If using spherical wave scattering model...
        if 'spherical' in enh_phys:

            # Get the spherical-scattering factor
            spherical = sphere_wave_attn(ant_rho=ant_rho,
                                         roi_rho=roi_rho,
                                         m_size=m_size,
                                         n_ant_pos=np.size(fd, axis=1))

            spherical = cp.asarray(spherical)  # Convert to GPU array

        # Define a variable for efficient computation that combines
        # the gain, beam, spherical scattering, and phase factors
        v_var = (gain[:, None, None, None] ** 2
                 * beam ** 2
                 * spherical
                 * phase_fac ** 2
                 )

    else:  # If enh_phys is None:

        # Define the variable for efficient computation
        v_var = phase_fac**2

    fd = cp.asarray(fd)  # Convert to GPU array

    # Get the area of each individual pixel, in [m^2]
    d_pix = np.diff(np.linspace(-roi_rho, roi_rho, m_size))[0]
    d_pix /= 100  # Convert from [cm] to [m]
    dv = d_pix**2  # Find area of each pixel

    img = cp.asarray(ini_img)  # Initialize the image

    # Forward project to data domain
    fwd = cp.sum(img[None, None, :, :]
                 * v_var,
                 axis=(2, 3)
                 ) * dv

    # Store the initial cost function value
    cost_funcs = [float(cp.sum(cp.abs(fwd - fd)**2))]

    rel_changes = []  # Init list for storing changes

    # Initialize the number of steps performed in gradient descent
    step = 0

    # Initialize the relative change in the cost function
    cost_rel_change = 1

    logger.info('\tInitial cost value:\t%.4f' % cost_funcs[0])

    # Perform gradient descent until the relative change in the cost
    # function is less than 1%
    while cost_rel_change > 0.01:

        step_start_t = time.time()

        logger.info('\t\tStep %d...' % (step + 1))
        tot_t_now = time.time() - tot_t0
        logger.info("\t\t\t(Total time so far: %d min %d sec"
                    % (tot_t_now // 60, tot_t_now % 60))

        # Calculate the full gradient w.r.t. the reflectivities
        ref_derivs = -1 * cp.sum(
            (cp.conj(v_var)
             * (fd[:, :, None, None] - fwd[:, :, None, None]))
            + v_var
            * cp.conj(fd[:, :, None, None] - fwd[:, :, None, None]),
            axis=(0,1)
        )

        # Update image estimate
        img -= step_size * cp.real(ref_derivs)

        fwd = cp.sum(img[None, None, :, :]
                     * v_var,
                     axis=(2, 3)
                     ) * dv

        # Normalize the forward projection
        cost_funcs.append(float(cp.sum(cp.abs(fwd - fd) ** 2)))

        logger.info('\t\t\tCost func:\t%.4f' % (cost_funcs[step + 1]))

        # Calculate the relative change in the cost function
        cost_rel_change = float(cp.abs((cost_funcs[step]
                                        - cost_funcs[step + 1])
                                       / cost_funcs[step]))

        # Store the relative change
        rel_changes.append(cost_rel_change)

        logger.info('\t\t\t\tCost Func ratio:\t%.4f%%'
                    % (100 * cost_rel_change))

        logger.info("\t\tStep took %.3f sec..." % (time.time() - step_start_t))

        step += 1  # Increment the step counter

    img = cp.asnumpy(img)  # Convert back to numpy array

    return img

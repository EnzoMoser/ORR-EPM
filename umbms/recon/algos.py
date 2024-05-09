"""
Tyson Reimer
University of Manitoba
June 4th, 2019
"""

import time
import numpy as np
import multiprocessing as mp
from scipy.signal import fftconvolve

from functools import partial

from umbms import null_logger

from umbms.hardware.antenna import get_ant_beam, get_ant_gain
from umbms.recon.enhphys import sphere_wave_attn
from umbms.recon.algosgpu import _orr_gpu


###############################################################################

# List of the valid physics model enhancements
_valid_enh_phys = [
    'spherical',  # Model spherical scattering from points in breast
    'beam',  # Model antenna beam pattern (horizontal)
    'gain',  # Model antenna frequency-dependent gain
    'complex_k',  # Model loss / conductivity -based attenuation
    'speed',  # Model path-dependent propagation time delay
]

###############################################################################


def orr_recon(ini_img,
              freqs,
              m_size,
              fd,
              roi_rho,
              phase_fac,
              ant_rho=-10.0,
              step_size=0.03,
              speed=0.0,
              enh_phys=None,
              use_gpu=False,
              logger=null_logger):
    """Perform optimization-based radar reconstruction, via grad desc

    Parameters
    ----------
    ini_img : array_like
        Initial image estimate
    freqs : array_like
        The frequencies used in the scan, in [Hz]
    m_size : int
        The number of pixels along one dimension of the reconstructed image
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

    if use_gpu:  # If using GPU for computation

        img = _orr_gpu(
            ini_img=ini_img,
            freqs=freqs,
            m_size=m_size,
            fd=fd,
            roi_rho=roi_rho,
            phase_fac=phase_fac,
            ant_rho=ant_rho,
            step_size=step_size,
            speed=speed,
            enh_phys=enh_phys,
            logger=logger,
        )

    else:  # If using CPU for computation

        img = _orr_cpu(
            ini_img=ini_img,
            freqs=freqs,
            m_size=m_size,
            fd=fd,
            roi_rho=roi_rho,
            phase_fac=phase_fac,
            ant_rho=ant_rho,
            step_size=step_size,
            speed=speed,
            enh_phys=enh_phys,
            logger=logger,
        )

    return img


def _orr_cpu(ini_img,
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
    logger.info("Initializing ORR...")

    beam = np.ones_like(phase_fac)
    gain = np.ones([len(freqs), ])
    spherical = np.ones(np.shape(phase_fac))

    if enh_phys is not None:  # If *using* some physics enhancements...

        # Assert that the antenna rhow as set (only used for
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

        if 'gain' in enh_phys:  # If using gain model...

            # Get the antenna gain factor
            gain = get_ant_gain(scan_fs=freqs)

        # If using spherical wave scattering model...
        if 'spherical' in enh_phys:

            # Get the spherical-scattering factor
            spherical = sphere_wave_attn(ant_rho=ant_rho,
                                         roi_rho=roi_rho,
                                         m_size=m_size,
                                         n_ant_pos=np.size(fd, axis=1))


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

    # Get the area of each individual pixel, in [m^2]
    d_pix = np.diff(np.linspace(-roi_rho, roi_rho, m_size))[0]
    d_pix /= 100  # Convert from [cm] to [m]
    dv = d_pix**2

    img = ini_img  # Set the image to the initial estimate

    # Use the forward model
    fwd = np.sum(img[None, None, :, :]
                 * v_var,
                 axis=(2, 3)
                 ) * dv

    # Store the initial cost function value
    cost_funcs = [float(np.sum(np.abs(fwd - fd)**2))]

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
        ref_derivs = -1 * np.sum(
            (np.conj(v_var)
             * (fd[:, :, None, None] - fwd[:, :, None, None]))
            + v_var
            * np.conj(fd[:, :, None, None] - fwd[:, :, None, None]),
            axis=(0, 1)
        )

        # Update image estimate
        img -= step_size * np.real(ref_derivs)

        fwd = np.sum(img[None, None, :, :]
                     * v_var,
                     axis=(2, 3)
                     ) * dv

        # Normalize the forward projection
        cost_funcs.append(float(np.sum(np.abs(fwd - fd) ** 2)))

        logger.info('\t\t\tCost func:\t%.4f' % (cost_funcs[step + 1]))

        # Calculate the relative change in the cost function
        cost_rel_change = float(np.abs((cost_funcs[step]
                                        - cost_funcs[step + 1])
                                       / cost_funcs[step]))

        # Store the relative change
        rel_changes.append(cost_rel_change)

        logger.info('\t\t\t\tCost Func ratio:\t%.4f%%'
                    % (100 * cost_rel_change))

        logger.info("\t\tStep took %.3f sec..." % (time.time() - step_start_t))

        step += 1  # Increment the step counter

    return img


###############################################################################


def fd_das(fd_data, phase_fac, freqs, n_cores=2,
           do_parallel=False):
    """Compute frequency-domain DAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension
    freqs : array_like, Nx1
        The frequencies used in the scan
    n_cores : int
        Number of cores used for parallel processing
    do_parallel : bool
        If True, will use parallel computation to calculate the image,
        looping parallel-ly over all frequencies. Intention: set
        to True if using more than 100 frequencies, otherwise set
        to False.

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """

    if not do_parallel:  # If *not* doing parallel computation

        # Reconstruct the image
        img = np.sum(
            fd_data[:, :, None, None]
            * np.power(phase_fac[None, :, :, :],
                       -2 * freqs[:, None, None, None]),
            axis=(0,1)
        )

    else:

        n_fs = np.size(freqs)  # Find number of frequencies used

        # Correct for to/from propagation
        new_phase_fac = phase_fac**(-2)

        # Create func for parallel computation
        parallel_func = partial(_parallel_fd_das_func, fd_data, new_phase_fac,
                                freqs)

        workers = mp.Pool(n_cores)  # Init worker pool

        iterable_idxs = range(n_fs)  # Indices to iterate over

        # Store projections from parallel processing
        back_projections = np.array(workers.map(parallel_func, iterable_idxs))

        # Reshape
        back_projections = np.reshape(back_projections,
                                      [n_fs, np.size(phase_fac, axis=1),
                                       np.size(phase_fac, axis=2)])

        workers.close()  # Close worker pool

        # Sum over all frequencies
        img = np.sum(back_projections, axis=0)

    return img


def _parallel_fd_das_func(fd_data, new_phase_fac, freqs, ff):
    """Compute projection for given frequency ff

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    new_phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension, corrected for DAS
    ff : int
        Frequency index

    Returns
    -------
    this_projection : array_like, KxK
        Back-projection of this particular frequency-point
    """

    # Get phase factor for this frequency
    this_phase_fac = new_phase_fac ** freqs[ff]

    # Sum over antenna positions
    this_projection = np.sum(this_phase_fac * fd_data[ff, :, None, None],
                             axis=0)

    return this_projection


###############################################################################


def fd_dmas(fd_data, phase_fac, freqs, n_cores=2):
    """Compute frequency-domain DAS reconstruction

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension
    freqs : array_like, Nx1
        The frequencies used in the scan
    n_cores : int
        Number of cores used for parallel processing

    Returns
    -------
    img : array_like, KxK
        Reconstructed image, K pixels by K pixels
    """

    # Find number of antenna positions
    n_ants = np.size(phase_fac, axis=0)

    # Correct for to/from propagation
    new_phase_fac = phase_fac**(-2)

    # Create func for parallel computation
    parallel_func = partial(_parallel_fd_dmas_func, fd_data, new_phase_fac,
                            freqs)

    workers = mp.Pool(n_cores)  # Init worker pool

    iterable_idxs = range(n_ants)  # Indices to iterate over

    # Store projections from parallel processing
    back_projections = np.array(workers.map(parallel_func, iterable_idxs))

    # Reshape
    back_projections = np.reshape(back_projections,
                                  [n_ants, np.size(phase_fac, axis=1),
                                   np.size(phase_fac, axis=2)])

    workers.close()  # Close worker pool

    # Init image to return
    img = np.zeros([np.size(phase_fac, axis=1), np.size(phase_fac, axis=1)],
                   dtype=complex)

    # Loop over each antenna position
    for aa in range(n_ants):

        # For each other antenna position
        for aa_2 in range(aa + 1, n_ants):

            # Add the pair-wise multiplication
            img += (back_projections[aa, :, :] * back_projections[aa_2, :, :])

    return img


def _parallel_fd_dmas_func(fd_data, new_phase_fac, freqs, aa):
    """Compute projection for given frequency ff

    Parameters
    ----------
    fd_data : array_like, NxM
        Frequency-domain data, complex-valued, N frequency points and M
        antenna positions
    new_phase_fac : array_like, MxKxK
        Phase factor, M antenna positions and K pixels along each
        dimension, corrected for DAS
    aa : int
        Antenna position index

    Returns
    -------
    this_projection : array_like, KxK
        Back-projection of this particular frequency-point
    """

    # Get phase factor for this antenna position
    this_phase_fac = new_phase_fac[aa, :, :] ** freqs[:, None, None]

    # Sum over antenna positions
    this_projection = np.sum(this_phase_fac * fd_data[:, aa, None, None],
                             axis=0)

    return this_projection

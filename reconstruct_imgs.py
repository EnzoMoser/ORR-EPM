"""
Tyson Reimer
University of Manitoba
June 20th, 2023
"""

import time

import os
import numpy as np
import multiprocessing as mp
import matplotlib

# Use 'tkagg' to display plot windows, use 'agg' to *not* display
# plot windows
matplotlib.use('agg')
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger, null_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.recon.algos import fd_dmas
from umbms.recon.algos import orr_recon
from umbms.recon.extras import get_pix_ts, get_pix_ds

from umbms.tdelay.propspeed import estimate_speed

from umbms.hardware.antenna import apply_ant_t_delay, to_phase_center

from umbms.plot.imgs import plot_img

from umbms.tdelay.partitioned import get_phase_fac_partitioned


###############################################################################

# Directory to load UM-BMID Gen-3 data
__D_DIR = os.path.join(get_proj_path(), '../um_bmid/datasets/gen-three/simple-clean/python-data/')

# Output dir for saving
__O_DIR = os.path.join(get_proj_path(), "output/gen-three/")
verify_path(__O_DIR)

# The frequency parameters from the scan
__INI_F = 1e9
__FIN_F = 9e9
__N_FS = 1001
__SCAN_FS = np.linspace(__INI_F, __FIN_F, __N_FS)

__M_SIZE = 150  # Number of pixels along 1-dimension for reconstruction
__ROI_RAD = 8  # ROI radius, in [cm]

__C = 2.99792458e8

# The approximate radius of each adipose phantom in our dataset,
# defined as the median radii
__ADI_RADS = {
    'A1': 3.87,
    'A2': 5.00,
    'A3': 5.66,
    'A11': 4.48,
    'A12': 4.57,
    'A13': 4.84,
    'A14': 5.19,
    'A15': 5.53,
    'A16': 5.74,
}

###############################################################################


def get_mat_perms(f_str):
    """Load material permittivities

    Parameters
    ----------
    f_str : str
        Path to material property file

    Returns
    -------
    fs : array_like
        Frequencies at which measurement was made
    perms : array_like
        Material permittivities
    conds : array_like
        Material conductivities, in [S/m]
    """

    csv_data = np.genfromtxt(fname=f_str,
                             skip_header=10,
                             dtype=float,
                             delimiter=',')

    fs = csv_data[:, 0] * 1e6  # Convert from [MHz] to [Hz]
    perms = csv_data[:, 1]
    conds = csv_data[:, 3]

    return fs, perms, conds


def interp_perms(data_fs, perms, conds, tar_fs):
    """Interpolate measured permittivity to a set of target freqs

    Parameters
    ----------
    data_fs : array_like
        Frequencies at which measured perms/conds are defined
    perms : array_like
        Measured real part of the relative permittivity
    conds : array_like
        Measured conductivity, [S/m]
    tar_fs : array_like
        Target frequencies

    Returns
    -------
    interp_perms : array_like
        Interpolated permittivities
    interp_conds : array_like
        Interpolated conductivities, in [S/m]
    """

    interp_perms = np.interp(x=tar_fs,
                            xp=data_fs,
                            fp=perms)
    interp_conds = np.interp(x=tar_fs,
                             xp=data_fs,
                             fp=conds)

    return interp_perms, interp_conds


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Flags for doing DAS / DMAS / ORR reconstructions
    do_das = True
    do_dmas = False
    do_orr = False

    # If True, ORR will use GPU for computation
    orr_use_gpu = False

    # Load the frequency-domain S11 and metadata
    s11 = load_pickle(os.path.join(__D_DIR, 'fd_data_gen_three_s11.pickle'))
    md = load_pickle(os.path.join(__D_DIR, 'metadata_gen_three.pickle'))

    # Get the scan IDs
    scan_ids = np.array([ii['id'] for ii in md])

    tar_fs = __SCAN_FS >= 2e9  # Bool arr for target frequencies

    # Store the frequencies used for reconstruction
    recon_fs = __SCAN_FS[tar_fs]

    # Down-sample frequencies for computation benefit
    # Goal: Retain as many frequencies as possible while
    # ensuring frequency spacing is no more than 100 MHz
    recon_fs = recon_fs[::12]

    s11 = s11[:, tar_fs, :]  # Keep only target frequencies
    s11 = s11[:, ::12, :]  # Down-sample for computation benefit

    # Load glycerin DAK dielectric data
    #gly_fs, gly_eps, gly_conds = get_mat_perms(
    #    f_str=os.path.join(
    #        get_proj_path(), 'data/dielectrics/glycerin.csv'
    #    ),
    #)

    # Interpolate to the scan frequencies
    #gly_eps, gly_conds = interp_perms(data_fs=gly_fs,
    #                                  perms=gly_eps,
    #                                  conds=gly_conds,
    #                                  tar_fs=recon_fs)

    n_expts = 1 #len(md)  # Find number of expts

    # Get the unique ID of each experiment / scan
    expt_ids = [mm['id'] for mm in md]

    # Get the unique IDs of the adipose-only and adipose-fibroglandular
    # (healthy) reference scans for each experiment/scan
    adi_ref_ids = [mm['adi_ref_id'] for mm in md]
    fib_ref_ids = [mm['fib_ref_id'] for mm in md]

    # List of physics enhanced modelling factors to use
    phys_enhancements = [
        # 'beam',
        # 'spherical',
        # 'gain',
        # 'sec_scat',
        # 'complex_k',
        # 'speed',
    ]

    if len(phys_enhancements) >= 1:  # If using some enhancements...

        # The output dir, where the reconstructions will be stored
        out_dir = os.path.join(__O_DIR, '-'.join(phys_enhancements)
                               + "-median/")
        verify_path(out_dir)

    else:  # If not using enhancements...

        out_dir = os.path.join(__O_DIR, "base-median-adi-rads/")
        verify_path(out_dir)

    # Define output dir for adipose and healthy references
    adi_o_dir = os.path.join(out_dir, 'adi/')
    verify_path(adi_o_dir)
    healthy_o_dir = os.path.join(out_dir, "healthy/")
    verify_path(healthy_o_dir)

    # Init lists for storing reconstructions using adipose reference
    all_das_adi = []
    all_dmas_adi = []
    all_orr_adi = []

    # Init lists for storing reconstructions using healthy reference
    all_das_fib = []
    all_dmas_fib = []
    all_orr_fib = []

    eps_0 = 8.85e-12  # Vacuum permittivity

    step_size = 0.3  # Select the step size

    # Init lists for storing metadata
    adi_md_to_save = []
    fib_md_to_save = []

    for ii in range(n_expts):  # For each scan / experiment

        # Save arrays every 20th image, in case of PC crash
        if ii % 20 == 0:

            # Save that DAS reconstruction to a .pickle file
            save_pickle(all_das_adi,
                        os.path.join(adi_o_dir,
                                     'up_to_idx%d_das_adi.pickle' % ii))
            save_pickle(all_das_fib,
                        os.path.join(healthy_o_dir,
                                     'up_to_idx%d_das_fib.pickle' % ii))

             # Save DMAS images
            save_pickle(all_dmas_adi,
                        os.path.join(adi_o_dir,
                                     'up_to_idx%d_dmas_adi.pickle' % ii))
            save_pickle(all_dmas_fib,
                        os.path.join(healthy_o_dir,
                                     'up_to_idx%d_dmas_fib.pickle' % ii))

            # Save that ORR reconstruction to a .pickle file
            save_pickle(all_orr_adi,
                        os.path.join(adi_o_dir,
                                     'up_to_idx%d_orr_adi.pickle' % ii))
            save_pickle(all_orr_fib,
                        os.path.join(healthy_o_dir,
                                     'up_to_idx%d_orr_fib.pickle' % ii))

            # Delete the previously-saved .pickle to save disk space
            if os.path.exists(
                    os.path.join(adi_o_dir,
                                 "up_to_idx%d_das_adi.pickle" % (ii - 20))):

                # Remove the old file
                os.remove(os.path.join(adi_o_dir,
                                 "up_to_idx%d_das_adi.pickle" % (ii - 20)))
                os.remove(os.path.join(adi_o_dir,
                                       "up_to_idx%d_das_fib.pickle" % (
                                                   ii - 20)))

            # Delete the previously-saved DMAS files to save disk space
            if os.path.exists(
                    os.path.join(adi_o_dir,
                                 "up_to_idx%d_dmas_adi.pickle" % (
                                         ii - 20))):
                os.remove(os.path.join(healthy_o_dir,
                                       "up_to_idx%d_dmas_fib.pickle" % (
                                                   ii - 20)))
                os.remove(os.path.join(healthy_o_dir,
                                       "up_to_idx%d_dmas_fib.pickle" % (
                                                   ii - 20)))

            # Delete the previously-saved ORR files to save disk space
            if os.path.exists(
                    os.path.join(adi_o_dir,
                                 "up_to_idx%d_orr_adi.pickle" % (
                                         ii - 20))):

                os.remove(os.path.join(adi_o_dir,
                                 "up_to_idx%d_orr_adi.pickle" % (ii - 20)))
                os.remove(os.path.join(healthy_o_dir,
                                       "up_to_idx%d_orr_fib.pickle" % (
                                                   ii - 20)))

        logger.info('Scan [%3d / %3d]...' % (ii + 1, n_expts))

        # Get the frequency domain data and metadata of this experiment
        tar_fd = s11[ii, :, :]
        tar_md = md[ii]

        # If the scan had a fibroglandular shell (indicating it was of
        # a complete tumour-containing or healthy phantom)
        if 'F' in tar_md['phant_id']:

            logger.info("\tDoing adipose-only reference reconstructions...")

            adi_md_to_save.append(tar_md)  # Store scan metadata

            # Get metadata for plotting
            scan_rad = tar_md['ant_rad']
            tum_x = tar_md['tum_x']
            tum_y = tar_md['tum_y']
            tum_rad = 0.5 * (tar_md['tum_diam'])
            adi_rad = __ADI_RADS[tar_md['phant_id'].split('F')[0]]

            # Correct for the antenna time delay
            # NOTE: Only the new antenna was used in UM-BMID Gen-3
            ant_rad = to_phase_center(meas_rho=scan_rad)

            # Estimate the propagation speed in the imaging domain
            speed = estimate_speed(adi_rad=adi_rad,
                                   ant_rho=ant_rad,
                                   )

            # If *not* using path-dependent time-delay modelling
            if not ('speed' in phys_enhancements):

                # Get the one-way propagation times for each pixel,
                # for each antenna position
                pix_ts = get_pix_ts(ant_rho=ant_rad,
                                    m_size=__M_SIZE,
                                    roi_rad=__ROI_RAD,
                                    speed=speed
                                    )

                # Apply the antenna time delay
                pix_ts = apply_ant_t_delay(pix_ts=pix_ts)

                # Get the phase factor for efficient computation
                phase_fac = np.exp(-1j
                                   * 2 * np.pi * recon_fs[:, None, None, None]
                                   * pix_ts[None, :, :, :]
                                   )

            # If using 'speed' and 'complex_k' enhancements
            elif (('speed' in phys_enhancements)
                  and ('complex_k' in phys_enhancements)):

                workers = mp.Pool(10)  # Workers for parallel processing

                # Get complex frequency-dependent propagation constant k
                k = 2 * np.pi * recon_fs * np.sqrt(
                    gly_eps - 1j * gly_conds / (2 * np.pi * recon_fs * eps_0)
                ) / __C

                # Calculate 1-way phase factors *NOTE: Without
                # antenna t-delay
                phase_factors = get_phase_fac_partitioned(
                    ant_rho=ant_rad,
                    m_size=__M_SIZE,
                    roi_rho=__ROI_RAD,
                    air_k=2 * np.pi * recon_fs / __C,
                    breast_k=k,
                    ini_a_ang=-136.0,
                    adi_rad=adi_rad,
                    phant_x=0,
                    phant_y=0,
                    worker_pool=workers
                )

                # For DMAS
                pix_ts = phase_factors

                # Define the phase factor using calculated
                # phase factors *and* account for the 1-way antenna
                # time delay of 0.19 ns
                phase_fac = (np.exp(-1j * phase_factors)
                             * np.exp(-1j * 2 * np.pi
                                      * recon_fs[:, None, None, None]
                                      * 0.19e-9
                                      )
                             )

            # If using 'speed' but NOT 'complex_k' enhancements
            elif (('speed' in phys_enhancements)
                  and not ('complex_k' in phys_enhancements)):

                workers = mp.Pool(10)  # Workers for parallel processing

                # Retain only the real part of k
                k = np.real(2 * np.pi * recon_fs * np.sqrt(
                    gly_eps - 1j * gly_conds / (2 * np.pi * recon_fs * eps_0)
                ) / __C)

                # Calculate 1-way phase factors *NOTE: Without
                # antenna t-delay
                phase_factors = get_phase_fac_partitioned(
                    ant_rho=ant_rad,
                    m_size=__M_SIZE,
                    roi_rho=__ROI_RAD,
                    air_k=2 * np.pi * recon_fs / __C,
                    breast_k=k,
                    ini_a_ang=-136.0,
                    adi_rad=adi_rad,
                    phant_x=0,
                    phant_y=0,
                    worker_pool=workers
                )

                # For DMAS
                pix_ts = phase_factors

                # Define the phase factor using calculated
                # phase factors *and* account for the 1-way antenna
                # time delay of 0.19 ns
                phase_fac = (np.exp(-1j * phase_factors)
                             * np.exp(-1j * 2 * np.pi
                                      * recon_fs[:, None, None, None]
                                      * 0.19e-9
                                      )
                             )

            else:  # If using only 'complex_k' enhancement

                # Get adipose / air areas
                adi_area = np.pi * adi_rad ** 2
                scan_area = np.pi * ant_rad ** 2

                # Determine average permittivity and conductivity based
                # on areas *at each frequency*
                avg_perm = (adi_area * gly_eps
                            + 1 * (scan_area - adi_area)) / scan_area
                avg_cond = (adi_area * gly_conds
                            + 0 * (scan_area - adi_area)) / scan_area

                # Get complex propagation constant k
                k = 2 * np.pi * recon_fs * np.sqrt(
                    avg_perm - 1j * avg_cond / (2 * np.pi * recon_fs * eps_0)
                ) / __C

                # Get the map of pixel distances, convert from
                # [cm] to [m]
                pix_ds = get_pix_ds(ant_rho=ant_rad,
                                    m_size=__M_SIZE,
                                    roi_rad=__ROI_RAD,
                                    ini_ant_ang=-136.0) / 100

                alpha = -2 * np.imag(k)  # Get attenuation parameter

                # Get the one-way propagation times for each pixel,
                # for each antenna position
                pix_ts = get_pix_ts(ant_rho=ant_rad,
                                    m_size=__M_SIZE,
                                    roi_rad=__ROI_RAD,
                                    speed=speed
                                    )

                # Apply the antenna time delay
                pix_ts = apply_ant_t_delay(pix_ts=pix_ts)

                # Get phase factor
                phase_fac = (
                    np.exp(-alpha[:, None, None, None]
                           * pix_ds[None, :, :, ]
                           * 0.5  # To account for squaring inside orr
                           )
                    * np.exp(-1j
                             * 2 * np.pi * recon_fs[:, None, None, None]
                             * pix_ts[None, :, :, :]
                             )
                )

            # Define phase factor for DMAS
            dmas_fac = np.exp(-1j * 2 * np.pi * pix_ts)

            # Get the adipose-only reference data for this scan
            adi_fd = s11[expt_ids.index(tar_md['adi_ref_id']), :, :]

            # Subtract reference
            adi_cal_cropped = (tar_fd - adi_fd)

            # If the scan does include a tumour
            if ~np.isnan(tar_md['tum_diam']):

                # Set a str for plotting
                plt_str = "%.1f cm tum in " \
                          "Class %d %s, ID: %d" % (tar_md['tum_diam'],
                                                   tar_md['birads'],
                                                   tar_md['phant_id'],
                                                   tar_md['id'])

            else:  # If the scan does NOT include a tumour
                plt_str = "Class %d %s, ID: %d" % (tar_md['birads'],
                                                   tar_md['phant_id'],
                                                   tar_md['id'])

            if do_das:  # If doing DAS reconstructions

                das_t0 = time.time()
                logger.info("\tStarting DAS recon...")

                # Reconstruct a DAS image
                das_adi_recon = np.sum(
                    adi_cal_cropped[:, :, None, None]
                    * phase_fac**(-2),
                    axis=(0,1)
                )

                # Store DAS image
                all_das_adi.append(das_adi_recon * np.ones_like(das_adi_recon))

                # Plot the DAS reconstruction
                plot_img(img=np.abs(das_adi_recon),
                         tar_xs=[tum_x],
                         tar_ys=[tum_y],
                         tar_rads=[tum_rad],
                         roi_rho=__ROI_RAD,
                         title='%s\nAdi Cal' % plt_str,
                         save_fig=True,
                         save_str=os.path.join(adi_o_dir,
                                               'idx_%d_adi_cal_das.png'
                                               % ii),
                         save_close=True
                         )
                logger.info("\t\tFinished DAS in %.1f sec."
                            % (time.time() - das_t0))

            if do_dmas:  # If doing DMAS reconstructions

                dmas_t0 = time.time()
                logger.info("\tStarting DMAS recon...")

                # Reconstruct a DMAS image
                dmas_adi_recon = fd_dmas(fd_data=adi_cal_cropped,
                                         phase_fac=dmas_fac,
                                         freqs=recon_fs,
                                         n_cores=10,
                                         )

                # Store DMAS image
                all_dmas_adi.append(dmas_adi_recon
                                    * np.ones_like(dmas_adi_recon))

                # Plot the DMAS reconstruction
                plot_img(img=np.abs(dmas_adi_recon),
                         tar_xs=[tum_x],
                         tar_ys=[tum_y],
                         tar_rads=[tum_rad],
                         roi_rho=__ROI_RAD,
                         title='%s\nAdi Cal' % plt_str,
                         save_fig=True,
                         save_str=os.path.join(adi_o_dir,
                                               'idx_%d_adi_cal_dmas.png'
                                               % ii),
                         save_close=True
                         )

                logger.info("\t\tFinished DMAS in %.1f sec."
                            % (time.time() - dmas_t0))

            if do_orr:  # If do ORR images...

                # Start timer
                orr_t0 = time.time()

                logger.info("\tStarting ORR recon...")

                # Reconstruct an ORR image
                adi_orr = orr_recon(
                    ini_img=np.zeros([__M_SIZE, __M_SIZE], dtype=complex),
                    freqs=recon_fs,
                    m_size=__M_SIZE,
                    fd=adi_cal_cropped,
                    roi_rho=__ROI_RAD,
                    phase_fac=phase_fac,
                    step_size=step_size,
                    enh_phys=phys_enhancements,
                    ant_rho=ant_rad,
                    logger=null_logger,
                    speed=speed,
                    use_gpu=orr_use_gpu,
                )

                # Store the image
                all_orr_adi.append(adi_orr * np.ones_like(adi_orr))

                # Plot the ORR image
                plot_img(img=np.abs(adi_orr),
                         tar_xs=[tum_x],
                         tar_ys=[tum_y],
                         tar_rads=[tum_rad],
                         roi_rho=__ROI_RAD,
                         title='%s\nAdi Cal' % plt_str,
                         save_fig=True,
                         save_str=os.path.join(adi_o_dir,
                                               'idx_%d_adi_cal_orr.png'
                                               % ii),
                         save_close=True)

                plt.close('all')

                # Report completion
                logger.info("\t\tFinished ORR in %.1f sec."
                            % (time.time() - orr_t0))

            # If the scan contained a tumour do extra reconstruction
            if ~np.isnan(tar_md['tum_diam']):

                logger.info("\tDoing healthy reference scan reconstructions...")

                # Get the data for the adipose-fibroglandular reference
                # scan
                fib_fd = s11[expt_ids.index(tar_md['fib_ref_id']), :, :]

                fib_md_to_save.append(tar_md)  # Store metadata

                # Subtract reference
                fib_cal_cropped = (tar_fd - fib_fd)

                if do_das:  # If doing DAS reconstruction

                    das_t0 = time.time()
                    logger.info("\tStarting DAS recon...")

                    # Produce DAS reconstruction, save, plot
                    das_fib_recon = np.sum(
                        fib_cal_cropped[:, :, None, None]
                        * phase_fac**(-2),
                        axis=(0,1)
                    )

                    # Store DAS reconstruction
                    all_das_fib.append(das_fib_recon
                                       * np.ones_like(das_fib_recon))

                    logger.info("\t\tFinished DAS in %.1f sec."
                                % (time.time() - das_t0))

                    # Plot DAS image
                    plot_img(img=np.abs(das_fib_recon),
                             tar_xs=[tum_x],
                             tar_ys=[tum_y],
                             tar_rads=[tum_rad],
                             roi_rho=__ROI_RAD,
                             title='%s\nFib Cal' % plt_str,
                             save_fig=True,
                             save_str=os.path.join(healthy_o_dir,
                                                   'idx_%d_fib_cal_das.png'
                                                   % ii),
                             save_close=True)

                if do_dmas:  # If doing DMAS

                    dmas_t0 = time.time()
                    logger.info("\tStarting DMAS recon...")

                    # Produce DMAS reconstruction, save, plot
                    dmas_fib_recon = fd_dmas(fd_data=fib_cal_cropped,
                                             phase_fac=dmas_fac,
                                             freqs=recon_fs,
                                             n_cores=10,
                                             )

                    # Store DMAS image
                    all_dmas_fib.append(dmas_fib_recon)

                    # Plot DMAS
                    plot_img(img=np.abs(dmas_fib_recon),
                             tar_xs=[tum_x],
                             tar_ys=[tum_y],
                             tar_rads=[tum_rad],
                             roi_rho=__ROI_RAD,
                             title='%s\nAdi Cal' % plt_str,
                                save_fig=True,
                                save_str=os.path.join(healthy_o_dir,
                                                      'idx_%d_fib_cal_dmas.png'
                                                      % ii),
                                save_close=True)

                    logger.info(
                        "\t\tFinished DMAS in %.1f sec."
                        % (time.time() - dmas_t0))

                if do_orr:  # If doing ORR

                    # Start timer
                    orr_t0 = time.time()

                    # Produce ORR reconstruction, save, plot
                    fib_gd = orr_recon(
                        ini_img=np.zeros([__M_SIZE, __M_SIZE], dtype=complex),
                        freqs=recon_fs,
                        m_size=__M_SIZE,
                        fd=fib_cal_cropped,
                        roi_rho=__ROI_RAD,
                        phase_fac=phase_fac,
                        step_size=step_size,
                        enh_phys=phys_enhancements,
                        ant_rho=ant_rad,
                        logger=null_logger,
                        speed=speed,
                        use_gpu=False,
                    )

                    # Store ORR image
                    all_orr_fib.append(fib_gd * np.ones_like(fib_gd))

                    # Plot ORR image
                    plot_img(img=np.abs(fib_gd),
                             tar_xs=[tum_x],
                             tar_ys=[tum_y],
                             tar_rads=[tum_rad],
                             roi_rho=__ROI_RAD,
                             title='%s\nFib Cal' % plt_str,
                             save_fig=True,
                             save_str=os.path.join(healthy_o_dir,
                                                   'idx_%d_fib_cal_orr.png'
                                                   % tar_md['id']),
                             save_close=True,
                             )

                    plt.close('all')

                    logger.info(
                        "\t\tFinished ORR in %.1f sec."
                        % (time.time() - orr_t0))

    if do_das:  # Save DAS if doing DAS

        save_pickle(all_das_adi, os.path.join(out_dir, "das_adi.pickle"))
        save_pickle(all_das_fib, os.path.join(out_dir, "das_fib.pickle"))

    if do_dmas:  # Save DMAS if doing DMAS
        save_pickle(all_dmas_adi, os.path.join(out_dir, "dmas_adi.pickle"))
        save_pickle(all_dmas_fib, os.path.join(out_dir, "dmas_fib.pickle"))

    if do_orr:  # Save ORR if doing ORR
        save_pickle(all_orr_adi, os.path.join(out_dir, "orr_adi.pickle"))
        save_pickle(all_orr_fib, os.path.join(out_dir, "orr_fib.pickle"))

    # Save metadata
    save_pickle(adi_md_to_save, os.path.join(__O_DIR, "adi_md.pickle"))
    save_pickle(fib_md_to_save, os.path.join(__O_DIR, "fib_md.pickle"))

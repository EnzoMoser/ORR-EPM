"""
Tyson Reimer
University of Manitoba
June 20th, 2023
"""

import time

import os
import numpy as np
import matplotlib

# Use 'tkagg' to display plot windows, use 'agg' to *not* display
# plot windows
matplotlib.use('agg')
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger, null_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.recon.extras import get_pix_ts, get_pix_ds

from umbms.tdelay.propspeed import estimate_speed

from umbms.hardware.antenna import apply_ant_t_delay, to_phase_center

from umbms.plot.imgs import plot_img

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

if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Flags for doing DAS / DMAS / ORR reconstructions
    do_das = True

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

    s11 = s11[0:7, :, 0:1]  # Only keep eight scans and ONLY MEASURE USING TWO ANTENNAS.

    s11 = s11[:, ::12, :]  # Down-sample for computation benefit

    n_expts = 1 #len(md)  # The number of scans to work on

    # Get the unique ID of each experiment / scan
    expt_ids = [mm['id'] for mm in md]

    # Get the unique IDs of the adipose-only and adipose-fibroglandular
    # (healthy) reference scans for each experiment/scan
    adi_ref_ids = [mm['adi_ref_id'] for mm in md]
    fib_ref_ids = [mm['fib_ref_id'] for mm in md]

    out_dir = os.path.join(__O_DIR, "base-median-adi-rads/")
    verify_path(out_dir)

    # Define output dir for adipose and healthy references
    adi_o_dir = os.path.join(out_dir, 'adi/')
    verify_path(adi_o_dir)
    healthy_o_dir = os.path.join(out_dir, "healthy/")
    verify_path(healthy_o_dir)

    # Init lists for storing reconstructions using adipose reference
    all_das_adi = []

    # Init lists for storing reconstructions using healthy reference
    all_das_fib = []

    # Init lists for storing metadata
    adi_md_to_save = []
    fib_md_to_save = []

    for ii in range(n_expts):  # For each scan / experiment
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
                         save_close=False
                         )
                logger.info("\t\tFinished DAS in %.1f sec."
                            % (time.time() - das_t0))

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
                             save_close=False)

    if do_das:  # Save DAS if doing DAS
        save_pickle(all_das_adi, os.path.join(out_dir, "das_adi.pickle"))
        save_pickle(all_das_fib, os.path.join(out_dir, "das_fib.pickle"))

    # Save metadata
    save_pickle(adi_md_to_save, os.path.join(__O_DIR, "adi_md.pickle"))
    save_pickle(fib_md_to_save, os.path.join(__O_DIR, "fib_md.pickle"))

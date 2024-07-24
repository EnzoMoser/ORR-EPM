"""
Tyson Reimer
University of Manitoba
September 23, 2023

Includes contributions by Fatimah Eashour
"""

import os
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle

from umbms.recon.breastmodels import get_roi

import warnings

# Suppress divide by zero warning
warnings.filterwarnings('ignore', 'divide by zero encountered in scalar divide')

###############################################################################

__D_DIR = os.path.join(get_proj_path(), '../um_bmid/datasets/gen-three/simple-clean/python-data/')

# Output dir for saving
__O_DIR = os.path.join(get_proj_path(), "output/orr/g3/")
verify_path(__O_DIR)

# Antenna polar rho coordinate, in [cm], after shifting to phase center
__N_ANTS = 72  # Number of antennas
__ROI_RHO = 8  # ROI radius, in [cm]

__M_SIZE = 150 # Image size


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # do_max_based = False
    do_fib_ref = False

    # Load the frequency-domain S11 and metadata
    s11 = load_pickle(os.path.join(__D_DIR, 'fd_data_gen_three_s11.pickle'))

    orr_str = "beam-spherical-gain-complex_k-median"
    orr_str_speed = "beam-spherical-gain-complex_k-speed-median"

    # -------------------------------------------------------------------------
    # Load ORR-BASE images
    orr_base_fib = np.array(
        load_pickle(
            os.path.join(__O_DIR, "base-median-adi-rads/orr_fib.pickle")
        )
    )
    orr_base_adi = np.array(
        load_pickle(os.path.join(__O_DIR,
                                 "base-median-adi-rads/orr_adi.pickle")
    )
    )

    # -------------------------------------------------------------------------
    # Load all other images, fib references
    das_fib_imgs = np.array(
        load_pickle(os.path.join(__O_DIR,
                                 "base-median-adi-rads/das_fib.pickle"))
    )
    dmas_fib_imgs = np.array(
        load_pickle(os.path.join(__O_DIR,
                                 "base-median-adi-rads/dmas_fib.pickle"))
    )
    orr_fib_imgs = np.array(
        load_pickle(os.path.join(__O_DIR,
                                 "%s/orr_fib.pickle" % orr_str))
    )
    orr_speed_fib_imgs = np.array(
        load_pickle(os.path.join(__O_DIR,
                                 "%s/orr_fib.pickle" % orr_str_speed))
    )

    fib_md = load_pickle(os.path.join(__O_DIR, "fib_md.pickle"))

    # -------------------------------------------------------------------------
    # Load all other images, fib references

    das_adi_imgs = np.array(
        load_pickle(os.path.join(__O_DIR,
                                 "base-median-adi-rads/das_adi.pickle"))
    )
    dmas_adi_imgs = np.array(
        load_pickle(os.path.join(__O_DIR,
                                 "base-median-adi-rads/dmas_adi.pickle"))
    )
    orr_adi_imgs = np.array(
        load_pickle(os.path.join(__O_DIR,
                                 "%s/orr_adi.pickle" % orr_str))
    )
    orr_speed_adi_imgs = np.array(
        load_pickle(os.path.join(__O_DIR,
                                 "%s/orr_adi.pickle" % orr_str_speed))
    )

    adi_md = load_pickle(os.path.join(__O_DIR, "adi_md.pickle"))

    # Find indices of the healthy images
    healthy_idx = np.array(
        [np.isnan(md['tum_diam']) for md in adi_md]
    )

    if do_fib_ref:  # If using healthy reference scans

        # Concatenate healthy and tumour-containing image datasets
        das_imgs = np.concatenate(
            (das_fib_imgs, das_adi_imgs[healthy_idx, :, :]))
        dmas_imgs = np.concatenate(
            (dmas_fib_imgs, dmas_adi_imgs[healthy_idx, :, :]))
        orr_epm_imgs = np.concatenate(
            (orr_fib_imgs, orr_adi_imgs[healthy_idx, :, :]))
        orr_base_imgs = np.concatenate(
            (orr_base_fib, orr_base_adi[healthy_idx, :, :]))
        orr_spd_imgs = np.concatenate(
            (orr_speed_fib_imgs, orr_speed_adi_imgs[healthy_idx, :, :])
        )
        md = np.concatenate((np.array(fib_md), np.array(adi_md)[healthy_idx]))

    else:  # If using adipose-only reference scans

        das_imgs = das_adi_imgs
        dmas_imgs = dmas_adi_imgs
        orr_epm_imgs = orr_adi_imgs
        orr_base_imgs = orr_base_adi
        orr_spd_imgs = orr_speed_adi_imgs
        md = adi_md

    healthy_idx = np.array(
        [np.isnan(mm['tum_diam']) for mm in md]
    )

    # ----- Make big figure ---------------------------------------------------

    o_dir = os.path.join(__O_DIR, "recon-fig/")
    verify_path(o_dir)

    # Select target indices
    tar_idx = 14
    tar_idx2 = 13
    # tar_idx = 7
    # tar_idx2 = 15


    das_img = das_imgs[tar_idx, :, :]
    dmas_img = dmas_imgs[tar_idx, :, :]
    orr_base_img = orr_base_imgs[tar_idx, :, :]
    orr_epm_img = orr_epm_imgs[tar_idx, :, :]
    orr_spd_img = orr_spd_imgs[tar_idx, :, :]

    das_img2 = das_imgs[tar_idx2, :, :]
    dmas_img2 = dmas_imgs[tar_idx2, :, :]
    orr_base_img2 = orr_base_imgs[tar_idx2, :, :]
    orr_epm_img2 = orr_epm_imgs[tar_idx2, :, :]
    orr_spd_img2 = orr_spd_imgs[tar_idx2, :, :]

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14, 6))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    cax = fig.add_axes([0.93, 0.05, 0.02, 0.8])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='inferno'), cax=cax)
    cbar.ax.tick_params(labelsize=16)

    tick_bounds = [-__ROI_RHO, __ROI_RHO, -__ROI_RHO, __ROI_RHO]
    draw_angs = np.linspace(0, 2 * np.pi, 1000)

    roi = get_roi(
        roi_rho=__ROI_RHO,
        m_size=__M_SIZE,
        arr_rho=__ROI_RHO,
    )

    all_imgs = [das_img, dmas_img, orr_base_img, orr_epm_img, orr_spd_img,
                das_img2, dmas_img2, orr_base_img2, orr_epm_img2, orr_spd_img2,
                ]

    norm_facs = [
        np.max(np.abs(das_img)),
        np.max(np.abs(dmas_img)),
        np.max(np.abs(orr_base_img)),
        np.max(np.abs(orr_epm_img)),
        np.max(np.abs(orr_spd_img)),
        np.max(np.abs(das_img)),
        np.max(np.abs(dmas_img)),
        np.max(np.abs(orr_base_img)),
        np.max(np.abs(orr_epm_img)),
        np.max(np.abs(orr_spd_img)),
    ]

    captions = [
        "(a) DAS",
        "(b) DMAS",
        "(c) ORR",
        "(d) ORR-EPM",
        "(e) ORR-EPM-T",
        "(f) DAS",
        "(g) DMAS",
        "(h) ORR",
        "(i) ORR-EPM",
        "(j) ORR-EPM-T",
    ]

    # Load stl (x,y) data BELOW .... ------------------------------------------
    phant_id = md[tar_idx]['phant_id']
    adi_id = phant_id.split("F")[0]
    fib_id = "F" + phant_id.split("F")[1]
    adi_xs, adi_ys = load_pickle(
        os.path.join(get_proj_path(),
                     "data/phantoms-stl/%s_xys_z65mm.pickle" % adi_id)
    )
    fib_xs, fib_ys = load_pickle(
        os.path.join(get_proj_path(),
                     "data/phantoms-stl/%s_xys_z35mm.pickle" % fib_id)
    )

    if ('A14' in [adi_id]) or ("A16" in [adi_id]):
        adi_xs, adi_ys = adi_ys, adi_xs
    if 'F14' in [fib_id]:
        fib_xs, fib_ys = fib_ys, fib_xs

    # Load stl (x,y) data ABOVE .... ------------------------------------------

    tar_rad = md[tar_idx]['tum_diam'] / 2
    tar_x = md[tar_idx]['tum_x']
    tar_y = md[tar_idx]['tum_y']

    for ii, ax in enumerate(axes.ravel()):

        img_to_plt = all_imgs[ii] * np.ones_like(all_imgs[ii])

        img_to_plt = np.abs(img_to_plt) / norm_facs[ii]
        img_to_plt[~roi] = np.nan

        ax.imshow(np.abs(img_to_plt),
                  cmap='inferno',
                  extent=tick_bounds,
                  aspect='equal',
                  vmin=0.0,
                  vmax=1.0,
                  )
        ax.scatter(adi_xs, adi_ys,
                   s=0.001,
                   marker='o',
                   color='w',
                   )
        ax.scatter(fib_xs, fib_ys,
                   s=0.001,
                   marker='o',
                   color='w',
                   )

        if ii < 5:
            # Get the x/y coords for plotting boundary
            plt_xs = tar_rad * np.cos(draw_angs) + tar_x
            plt_ys = tar_rad * np.sin(draw_angs) + tar_y

            # Plot the circular boundary
            ax.plot(plt_xs, plt_ys, 'w',
                    linewidth=2.0,
                    linestyle='--',
                    )

        ax.set_xlim([-__ROI_RHO, __ROI_RHO])
        ax.set_ylim([-__ROI_RHO, __ROI_RHO])

        if ii == 0:

            ax.set_xlabel("x-axis (cm)",
                          fontsize=16,
                          labelpad=-5,
                          )
            ax.set_ylabel("x-axis (cm)", fontsize=16)

        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if ii < 1:
            ax.text(0.5, -0.3,
                    "%s" % captions[ii],
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    )
        else:
            ax.text(0.5, -0.30,
                    "%s" % captions[ii],
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    )
    #
    # plt.savefig(os.path.join(__O_DIR, "big_recon-ex-adi-ref.png"),
    #             bbox_inches='tight',
    #             dpi=300)

    o_dir = os.path.join(__O_DIR, "recon-fig/")
    verify_path(o_dir)

    # tar_idx = 14
    # tar_idx2 = 13
    tar_idx = 7
    # tar_idx2 = 15
    das_img = das_imgs[tar_idx, :, :]
    dmas_img = dmas_imgs[tar_idx, :, :]
    orr_base_img = orr_base_imgs[tar_idx, :, :]
    orr_epm_img = orr_epm_imgs[tar_idx, :, :]
    orr_spd_img = orr_spd_imgs[tar_idx, :, :]

    das_img2 = das_imgs[tar_idx2, :, :]
    dmas_img2 = dmas_imgs[tar_idx2, :, :]
    orr_base_img2 = orr_base_imgs[tar_idx2, :, :]
    orr_epm_img2 = orr_epm_imgs[tar_idx2, :, :]
    orr_spd_img2 = orr_spd_imgs[tar_idx2, :, :]

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(14, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    cax = fig.add_axes([0.93, 0.05, 0.02, 0.8])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='inferno'), cax=cax)
    cbar.ax.tick_params(labelsize=16)

    tick_bounds = [-__ROI_RHO, __ROI_RHO, -__ROI_RHO, __ROI_RHO]
    draw_angs = np.linspace(0, 2 * np.pi, 1000)

    roi = get_roi(
        roi_rho=__ROI_RHO,
        m_size=__M_SIZE,
        arr_rho=__ROI_RHO,
    )

    all_imgs = [das_img, dmas_img, orr_base_img, orr_epm_img, orr_spd_img,
                # das_img2, dmas_img2, orr_base_img2, orr_epm_img2, orr_spd_img2,
                ]

    norm_facs = [
        np.max(np.abs(das_img)),
        np.max(np.abs(dmas_img)),
        np.max(np.abs(orr_base_img)),
        np.max(np.abs(orr_epm_img)),
        np.max(np.abs(orr_spd_img)),
        np.max(np.abs(das_img)),
        np.max(np.abs(dmas_img)),
        np.max(np.abs(orr_base_img)),
        np.max(np.abs(orr_epm_img)),
        np.max(np.abs(orr_spd_img)),
    ]

    captions = [
        "(a) DAS",
        "(b) DMAS",
        "(c) ORR",
        "(d) ORR-EPM",
        "(e) ORR-EPM-T",
        "(f) DAS",
        "(g) DMAS",
        "(h) ORR",
        "(i) ORR-EPM",
        "(j) ORR-EPM-T",
    ]

    # Load stl (x,y) data BELOW .... ------------------------------------------
    phant_id = md[tar_idx]['phant_id']
    adi_id = phant_id.split("F")[0]
    fib_id = "F" + phant_id.split("F")[1]
    adi_xs, adi_ys = load_pickle(
        os.path.join(get_proj_path(),
                     "data/phantoms-stl/%s_xys_z65mm.pickle" % adi_id)
    )
    fib_xs, fib_ys = load_pickle(
        os.path.join(get_proj_path(),
                     "data/phantoms-stl/%s_xys_z35mm.pickle" % fib_id)
    )

    if ('A14' in [adi_id]) or ("A16" in [adi_id]):
        adi_xs, adi_ys = adi_ys, adi_xs
    if 'F14' in [fib_id]:
        fib_xs, fib_ys = fib_ys, fib_xs
    # Load stl (x,y) data ABOVE .... ------------------------------------------

    tar_rad = md[tar_idx]['tum_diam'] / 2
    tar_x = md[tar_idx]['tum_x']
    tar_y = md[tar_idx]['tum_y']

    for ii, ax in enumerate(axes.ravel()):

        img_to_plt = all_imgs[ii] * np.ones_like(all_imgs[ii])

        img_to_plt = np.abs(img_to_plt) / norm_facs[ii]
        img_to_plt[~roi] = np.nan

        ax.imshow(np.abs(img_to_plt),
                  cmap='inferno',
                  extent=tick_bounds,
                  aspect='equal',
                  vmin=0.0,
                  vmax=1.0,
                  )
        ax.scatter(adi_xs, adi_ys,
                   s=0.001,
                   marker='o',
                   color='w',
                   )
        ax.scatter(fib_xs, fib_ys,
                   s=0.001,
                   marker='o',
                   color='w',
                   )

        if ii < 5:
            # Get the x/y coords for plotting boundary
            plt_xs = tar_rad * np.cos(draw_angs) + tar_x
            plt_ys = tar_rad * np.sin(draw_angs) + tar_y

            # Plot the circular boundary
            ax.plot(plt_xs, plt_ys, 'w',
                    linewidth=2.0,
                    linestyle='--',
                    )

        ax.set_xlim([-__ROI_RHO, __ROI_RHO])
        ax.set_ylim([-__ROI_RHO, __ROI_RHO])

        if ii == 0:
            ax.set_xlabel("x-axis (cm)",
                          fontsize=16,
                          labelpad=-5,
                          )
            ax.set_ylabel("x-axis (cm)", fontsize=16)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if ii < 1:
            ax.text(0.5, -0.3,
                    "%s" % captions[ii],
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    )
        else:
            ax.text(0.5, -0.30,
                    "%s" % captions[ii],
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    )
    #
    # plt.savefig(os.path.join(__O_DIR, "big_recon-ex-healthy.png"),
    #             bbox_inches='tight',
    #             dpi=300)

    ###########################################################################
    # ----------------------------------------------------------------------- #
    ###########################################################################


    ###########################################################################
    # ----------------------------------------------------------------------- #
    ###########################################################################

    das_imgs = np.concatenate(
        (das_fib_imgs, das_adi_imgs[healthy_idx, :, :]))
    dmas_imgs = np.concatenate(
        (dmas_fib_imgs, dmas_adi_imgs[healthy_idx, :, :]))
    orr_epm_imgs = np.concatenate(
        (orr_fib_imgs, orr_adi_imgs[healthy_idx, :, :]))
    orr_base_imgs = np.concatenate(
        (orr_base_fib, orr_base_adi[healthy_idx, :, :]))
    orr_spd_imgs = np.concatenate(
        (orr_speed_fib_imgs, orr_speed_adi_imgs[healthy_idx, :, :])
    )
    md = np.concatenate((np.array(fib_md), np.array(adi_md)[healthy_idx]))

    # tar_idx = 12
    # tar_idx2 = 13
    tar_idx = 7
    # tar_idx2 = 15
    das_img = das_imgs[tar_idx, :, :]
    dmas_img = dmas_imgs[tar_idx, :, :]
    orr_base_img = orr_base_imgs[tar_idx, :, :]
    orr_epm_img = orr_epm_imgs[tar_idx, :, :]
    orr_spd_img = orr_spd_imgs[tar_idx, :, :]

    das_img2 = das_imgs[tar_idx2, :, :]
    dmas_img2 = dmas_imgs[tar_idx2, :, :]
    orr_base_img2 = orr_base_imgs[tar_idx2, :, :]
    orr_epm_img2 = orr_epm_imgs[tar_idx2, :, :]
    orr_spd_img2 = orr_spd_imgs[tar_idx2, :, :]

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(14, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    cax = fig.add_axes([0.93, 0.35, 0.02, 0.3])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='inferno'), cax=cax)
    cbar.ax.tick_params(labelsize=16)

    tick_bounds = [-__ROI_RHO, __ROI_RHO, -__ROI_RHO, __ROI_RHO]
    draw_angs = np.linspace(0, 2 * np.pi, 1000)

    roi = get_roi(
        roi_rho=__ROI_RHO,
        m_size=__M_SIZE,
        arr_rho=__ROI_RHO,
    )

    all_imgs = [das_img, dmas_img, orr_base_img, orr_epm_img, orr_spd_img,
                # das_img2, dmas_img2, orr_base_img2, orr_epm_img2, orr_spd_img2,
                ]

    norm_facs = [
        np.max(np.abs(das_img)),
        np.max(np.abs(dmas_img)),
        np.max(np.abs(orr_base_img)),
        np.max(np.abs(orr_epm_img)),
        np.max(np.abs(orr_spd_img)),
        np.max(np.abs(das_img)),
        np.max(np.abs(dmas_img)),
        np.max(np.abs(orr_base_img)),
        np.max(np.abs(orr_epm_img)),
        np.max(np.abs(orr_spd_img)),
    ]

    captions = [
        "(a) DAS",
        "(b) DMAS",
        "(c) ORR",
        "(d) ORR-EPM",
        "(e) ORR-EPM-T",
        "(f) DAS",
        "(g) DMAS",
        "(h) ORR",
        "(i) ORR-EPM",
        "(j) ORR-EPM-T",
    ]

    # Load stl (x,y) data BELOW .... ------------------------------------------
    phant_id = md[tar_idx]['phant_id']
    adi_id = phant_id.split("F")[0]
    fib_id = "F" + phant_id.split("F")[1]
    adi_xs, adi_ys = load_pickle(
        os.path.join(get_proj_path(),
                     "data/phantoms-stl/%s_xys_z65mm.pickle" % adi_id)
    )
    fib_xs, fib_ys = load_pickle(
        os.path.join(get_proj_path(),
                     "data/phantoms-stl/%s_xys_z35mm.pickle" % fib_id)
    )

    if ('A14' in [adi_id]) or ("A16" in [adi_id]):
        adi_xs, adi_ys = adi_ys, adi_xs
    if 'F14' in [fib_id]:
        fib_xs, fib_ys = fib_ys, fib_xs
    # Load stl (x,y) data ABOVE .... ------------------------------------------

    tar_rad = md[tar_idx]['tum_diam'] / 2
    tar_x = md[tar_idx]['tum_x']
    tar_y = md[tar_idx]['tum_y']

    for ii, ax in enumerate(axes.ravel()):

        img_to_plt = all_imgs[ii] * np.ones_like(all_imgs[ii])

        img_to_plt = np.abs(img_to_plt) / norm_facs[ii]
        img_to_plt[~roi] = np.nan

        ax.imshow(np.abs(img_to_plt),
                  cmap='inferno',
                  extent=tick_bounds,
                  aspect='equal',
                  vmin=0.0,
                  vmax=1.0,
                  )
        ax.scatter(adi_xs, adi_ys,
                   s=0.001,
                   marker='o',
                   color='w',
                   )
        ax.scatter(fib_xs, fib_ys,
                   s=0.001,
                   marker='o',
                   color='w',
                   )

        if ii < 5:
            # Get the x/y coords for plotting boundary
            plt_xs = tar_rad * np.cos(draw_angs) + tar_x
            plt_ys = tar_rad * np.sin(draw_angs) + tar_y

            # Plot the circular boundary
            ax.plot(plt_xs, plt_ys, 'w',
                    linewidth=2.0,
                    linestyle='--',
                    )

        ax.set_xlim([-__ROI_RHO, __ROI_RHO])
        ax.set_ylim([-__ROI_RHO, __ROI_RHO])

        if ii == 0:
            ax.set_xlabel("x-axis (cm)",
                          fontsize=16,
                          labelpad=-5,
                          )
            ax.set_ylabel("x-axis (cm)", fontsize=16)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if ii < 1:
            ax.text(0.5, -0.3,
                    "%s" % captions[ii],
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    )
        else:
            ax.text(0.5, -0.30,
                    "%s" % captions[ii],
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20,
                    )
    # #
    # plt.savefig(os.path.join(__O_DIR, "big_recon-ex-healthy-ref.png"),
    #             bbox_inches='tight',
    #             dpi=300)



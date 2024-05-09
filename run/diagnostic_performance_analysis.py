"""
Tyson Reimer
University of Manitoba
August 30th, 2021
"""

import matplotlib
matplotlib.use('tkagg')
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import os
import numpy as np

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle

from umbms.iqms.accuracy import get_loc_err
from umbms.iqms.contrast import get_scr

###############################################################################

__D_DIR = os.path.join(get_proj_path(), 'data/umbmid/g3/')

# Output dir for saving
__O_DIR = os.path.join(get_proj_path(), "output/orr/g3/")
verify_path(__O_DIR)

# Scan frequency parameters
__INI_F = 1e9
__FIN_F = 9e9
__N_FS = 1001

# Image size
__M_SIZE = 150
__ROI_RAD = 8

# Approximate radius of each adipose shell in our array
__ADI_RADS = {
    'A1': 5,
    'A2': 6,
    'A3': 7,
    'A11': 6,
    'A12': 5,
    'A13': 6.5,
    'A14': 6,
    'A15': 5.5,
    'A16': 7
}

# Assumed tumour radius for healthy reconstructions where the SCR
# threshold is exceeded
__HEALTHY_RAD = 1.5

# The str indicating the reference type for the tumour-containing scans
# must be in ['adi', 'fib']
__TUM_REF_STR = 'adi'

###############################################################################

# Define RGB colours for plotting DAS/DMAS/ORR
das_col = [0, 0, 0]
dmas_col = [80, 80, 80]
gd_col = [160, 160, 160]
das_col = [ii / 255 for ii in das_col]
dmas_col = [ii / 255 for ii in dmas_col]
gd_col = [ii / 255 for ii in gd_col]


def get_auc(sens, specs):
    """Calculate the area-under-the curve of the ROC curve

    Parameters
    ----------
    sens : array_like
        The diagnostic sensitivity at each decision threshold,
        as a percentage
    specs : array_like
        The diagnostic specificity at each decision threshold,
        as a percentage

    Returns
    -------
    auc : float
        The AUC of the ROC curve
    d_auc : float
        The numerical uncertainty in the AUC
    """

    tps = sens / 100  # Calculate the true positive rate as a decimal

    # Calculate the false positive rate as a decimal
    fps = (100 - specs) / 100

    sort_idxs = np.argsort(fps)  # Order appropriately to make ROC curve

    # Sort TPR and FPR
    tps = tps[sort_idxs]
    fps = fps[sort_idxs]

    auc = 0  # Init AUC using center rectangle integration
    auc1 = 0  # Init AUC using left rectangle integration
    auc2 = 0  # Init AUC using right rectangle integration

    for ii in range(len(tps) - 1):  # For each decision threshold

        # Calculate rectangle width (for integration)
        d_fps = fps[ii + 1] - fps[ii]

        if d_fps != 0:  # For a nonzero width

            # Calculate the AUCs
            auc += d_fps * 0.5 * (tps[ii + 1] + tps[ii])
            auc1 += d_fps * tps[ii]
            auc2 += d_fps * tps[ii + 1]

    # Convert from fractions to percentages
    auc *= 100
    auc1 *= 100
    auc2 *= 100

    d_auc = auc2 - auc1  # Approximate numerical uncertainty

    if d_auc < 0.5:  # Round up to at least 1%
        d_auc = 1

    return auc, d_auc


def get_le_scr(img, roi_rad, adi_rad, tum_x, tum_y, tum_rad):
    """Calculate the localization error and signal to clutter ratio

    Parameters
    ----------
    img : array_like
        Reconstructed image
    roi_rad : float
        The image radius, in [cm]
    adi_rad : float
        The approximate radius of the breast, in [cm]
    tum_x : float
        The approximate tumour x-position, in [cm]
    tum_y : float
        The approximate tumour y-position, in [cm]
    tum_rad : float
        The approximate tumour radius, in [cm]

    Returns
    -------
    scr : float
        The signal-to-clutter ratio, in [dB]
    le : float
        The localization error, in [cm]
    """

    # Get SCR
    scr = get_scr(
        img=img,
        roi_rad=roi_rad,
        adi_rad=adi_rad,
        tum_rad=tum_rad,
    )

    # Get LE
    le = get_loc_err(
        img=img,
        roi_rho=roi_rad,
        tum_x=tum_x,
        tum_y=tum_y
    )

    return scr, le


def get_sens_spec(scrs, scr_thresh, les, tum_rads, healthy_idx):
    """Calculate the diagnostic sensitivity and specificity

    Parameters
    ----------
    scrs : array_like
        The SCR of the reconstructions
    scr_thresh : float
        The SCR threshold value used to determine the presence of a
        lesion in the reconstructed image
    les : array_like
        The localization errors (LEs) of the reconstructions
    tum_rads : array_like
        The radius of the tumour in each image
    healthy_idx : array_like
        Boolean entries indicating presence / absence of a tumour
        in the phantom

    Returns
    -------
    sens : float
        The diagnostic sensitivity, as a percentage
    spec : float
        The diagnostic specificity, as a percentage
    """

    # Get the true positive count
    tp = np.sum(
        np.logical_and(scrs >= scr_thresh,
                       les <= (tum_rads + 0.5)
                       )[~healthy_idx])

    # Get the false positive count
    fp = np.sum(
        np.logical_and(scrs >= scr_thresh,
                       healthy_idx)
    )

    # Get true negative count
    tn = np.sum(
        np.logical_and(scrs < scr_thresh,
                       healthy_idx)
    )

    # Get false negative count
    fn = np.sum(
        np.logical_or(scrs < scr_thresh,
                      les > (tum_rads + 0.5)
                      )[~healthy_idx]
    )

    sens = tp / (tp + fn) * 100  # Convert to percentage
    spec = tn / (tn + fp) * 100  # Convert to percentage

    return sens, spec


def get_scrs_les(imgs):
    """Get the SCRs and LEs for the reconstructed images

    Parameters
    ----------
    imgs : array_like
        Reconstructed images

    Returns
    -------
    scrs : array_like
        SCRs of each image, in [dB]
    les : array_like
        Localization errors in each image. Note that healthy images
        have a LE of np.nan
    """

    # Init arrays for return
    scrs = np.zeros([np.size(imgs, axis=0)])
    les = np.zeros([np.size(imgs, axis=0)])

    # For each scan, determine the SCR and LE
    for ii in range(np.size(imgs, axis=0)):

        # Get the metadata for this scan
        tar_md = md[ii]

        # Get metadata for plotting
        scan_rad = tar_md['ant_rad']
        tum_x = tar_md['tum_x']
        tum_y = tar_md['tum_y']
        adi_rad = __ADI_RADS[tar_md['phant_id'].split('F')[0]]

        # If the scan had a fibroglandular shell (indicating it was of
        # a complete tumour-containing or healthy phantom)
        if 'F' in tar_md['phant_id'] and ~np.isnan(tar_md['tum_diam']):

            # Get the SCR and localization error for the DAS image
            scrs[ii], les[ii] = get_le_scr(
                img=imgs[ii, :, :],
                roi_rad=__ROI_RAD,
                adi_rad=adi_rad,
                tum_x=tum_x,
                tum_y=tum_y,
                tum_rad=1.5,
            )

        # If the experiment was of a healthy phantom
        elif 'F' in tar_md['phant_id'] and np.isnan(tar_md['tum_diam']):

            # Get the SCR for ORR
            scrs[ii] = get_scr(
                img=np.abs(imgs[ii, :, :]),
                roi_rad=__ROI_RAD,
                adi_rad=adi_rad,
                tum_rad=1.5,
            )

            # Set LE to nan
            les[ii] = np.nan

    return scrs, les


def get_best_pred(imgs, sens, specs, det_thresholds):
    """Get classifier predictions based on best point on ROC curve

    Parameters
    ----------
    sens : array_like
        The diagnostic sensitivity at each decision threshold,
        as a percentage
    specs : array_like
        The diagnostic specificity at each decision threshold,
        as a percentage

    Returns
    -------
    preds : array_like
        The predictions at the decision threshold which is
        closest to the point (0, 1) in the ROC space
    """

    # Find the best index
    best_idx = np.argmin(np.sqrt((1 - sens / 100)**2 + (1 - specs / 100)**2))

    # Threshold corresponding to best point on ROC
    best_thresh = det_thresholds[best_idx]

    print('Best thresh:\t%.3f' % (best_thresh / np.max(imgs)))

    # Get the max intensity in each image
    img_maxes = np.max(np.abs(imgs), axis=(1, 2))

    preds = img_maxes >= best_thresh  # Get best predictions

    return preds


def get_scr_preds(imgs, scrs, scr_thresh, les, tum_rads, healthy_idx):
    """Get classification predictions based on a SCR threshold value

    Parameters
    ----------
    imgs : array_like
        Reconstructed images
    scrs : array_like
        SCR of each reconstruction, in [dB]
    scr_thresh : float
        The SCR threshold used to determine the presence of a lesion in
        the reconstruction, in [dB]
    les : array_like
        The localization errors of each reconstruction, in [cm]
    tum_rads : array_like
        The radius of the tumour in each phantom scan
    healthy_idx : array_like
        Boolean entries indicating presence / absence of a lesion
        in each phantom

    Returns
    -------
    preds : array_like
        The predictions for each SCR threshold
    """

    preds = np.zeros([np.size(imgs, axis=0)])  # Init arr for return

    for ii in range(np.size(imgs, axis=0)):  # For each image

        if healthy_idx[ii]:  # If healthy scan
            preds[ii] = scrs[ii] >= scr_thresh

        else:  # If tumour scan
            preds[ii] = (scrs[ii] >= scr_thresh
                         and les[ii] <= (tum_rads[ii] + 0.5))

    return preds


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    do_max_based = False
    do_fib_ref = True

    # Load the frequency-domain S11 and metadata
    s11 = load_pickle(os.path.join(__D_DIR, 'g3_s11.pickle'))

    orr_str = 'beam-spherical-gain-complex_k-median'
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
        orr_imgs = np.concatenate(
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
        orr_imgs = orr_adi_imgs
        orr_base_imgs = orr_base_adi
        orr_spd_imgs = orr_speed_adi_imgs

        md = adi_md

    # Create array whose entries indicate the absence of a tumour
    healthy_idx = np.array(
        [np.isnan(mm['tum_diam']) for mm in md]
    )

    # Find number of experiments / scans
    n_expts = np.size(das_imgs, axis=0)

    # Calculate the SCRs and LEs for each reconstruction method
    das_scrs, das_les = get_scrs_les(imgs=das_imgs)
    dmas_scrs, dmas_les = get_scrs_les(imgs=dmas_imgs)
    orr_scrs, orr_les = get_scrs_les(imgs=orr_imgs)
    orr_base_scrs, orr_base_les = get_scrs_les(imgs=orr_base_imgs)
    orr_spd_scrs, orr_spd_les = get_scrs_les(imgs=orr_spd_imgs)

    # -------------------------------------------------------------------------
    # -------- Now calculate sensitivities and specificities ------------------
    # -------------------------------------------------------------------------

    # The SCR thresholds to be investigated
    scr_thresholds = np.linspace(0, 30, 10000)

    # Get array of tumour radii
    tum_rads = np.array([mm['tum_diam'] / 2 for mm in md])

    # Init result arrays
    das_res = np.zeros([len(scr_thresholds), 2])
    dmas_res = np.zeros([len(scr_thresholds), 2])
    orr_res = np.zeros([len(scr_thresholds), 2])
    orr_base_res = np.zeros([len(scr_thresholds), 2])
    orr_spd_res = np.zeros([len(scr_thresholds), 2])

    # For each SCR threshold...
    for scr_ii in range(len(scr_thresholds)):

        # Get the SCR threshold here
        scr_thresh = scr_thresholds[scr_ii]

        # Get the sensitivity / specificity results for DAS
        das_res[scr_ii, :] = get_sens_spec(
            scrs=das_scrs,
            scr_thresh=scr_thresh,
            les=das_les,
            tum_rads=tum_rads,
            healthy_idx=healthy_idx
        )

        # Get the sensitivity / specificity results for DMAS
        dmas_res[scr_ii, :] = get_sens_spec(
            scrs=dmas_scrs,
            scr_thresh=scr_thresh,
            les=dmas_les,
            tum_rads=tum_rads,
            healthy_idx=healthy_idx
        )

        # Get the sensitivity / specificity results for ORR
        orr_res[scr_ii, :] = get_sens_spec(
            scrs=orr_scrs,
            scr_thresh=scr_thresh,
            les=orr_les,
            tum_rads=tum_rads,
            healthy_idx=healthy_idx
        )

        # Get the sensitivity / specificity results for ORR-EPM
        orr_base_res[scr_ii, :] = get_sens_spec(
            scrs=orr_base_scrs,
            scr_thresh=scr_thresh,
            les=orr_base_les,
            tum_rads=tum_rads,
            healthy_idx=healthy_idx,
        )

        # Get the sensitivity / specificity results for ORR-EPM-T
        orr_spd_res[scr_ii, :] = get_sens_spec(
            scrs=orr_spd_scrs,
            scr_thresh=scr_thresh,
            les=orr_spd_les,
            tum_rads=tum_rads,
            healthy_idx=healthy_idx,
        )

    # Identify the decision threshold which is closest to the (0, 1)
    # point in the ROC curve for each reconstruction method
    # and report the best senstivity / specificity to the console
    das_best_idx = np.argmin((100 - das_res[:, 0]) ** 2
                             + (100 - das_res[:, 1]) ** 2)
    logger.info("DAS Best Sens / Spec:\t%.2f%%, %.2f%%"
                % (das_res[das_best_idx, 0], das_res[das_best_idx, 1]))
    dmas_best_idx = np.argmin((100 - dmas_res[:, 0]) ** 2
                              + (100 - dmas_res[:, 1]) ** 2)
    logger.info("DMAS Best Sens / Spec:\t%.2f%%, %.2f%%"
                % (dmas_res[dmas_best_idx, 0], dmas_res[dmas_best_idx, 1]))
    orr_base_best_idx = np.argmin((100 - orr_base_res[:, 0]) ** 2
                                  + (100 - orr_base_res[:, 1]) ** 2)
    logger.info("ORR Base Best Sens / Spec:\t%.2f%%, %.2f%%"
                % (orr_base_res[orr_base_best_idx, 0],
                   orr_base_res[orr_base_best_idx, 1]))
    orr_best_idx = np.argmin((100 - orr_res[:, 0]) ** 2
                             + (100 - orr_res[:, 1]) ** 2)
    logger.info("ORR Best Sens / Spec:\t%.2f%%, %.2f%%"
                % (orr_res[orr_best_idx, 0], orr_res[orr_best_idx, 1]))
    orr_spd_best_idx = np.argmin((100 - orr_spd_res[:, 0]) ** 2
                                  + (100 - orr_spd_res[:, 1]) ** 2)
    logger.info("ORR Base Best Sens / Spec:\t%.2f%%, %.2f%%"
                % (orr_spd_res[orr_spd_best_idx, 0],
                   orr_spd_res[orr_spd_best_idx, 1]))

    # Get the best predictions for each reconstruction method
    das_preds = get_scr_preds(
        imgs=das_imgs,
        scrs=das_scrs,
        scr_thresh=scr_thresholds[das_best_idx],
        les=das_les,
        healthy_idx=healthy_idx,
        tum_rads=tum_rads,
    )
    dmas_preds = get_scr_preds(
        imgs=dmas_imgs,
        scrs=dmas_scrs,
        scr_thresh=scr_thresholds[dmas_best_idx],
        les=dmas_les,
        healthy_idx=healthy_idx,
        tum_rads=tum_rads,
    )
    orr_preds = get_scr_preds(
        imgs=orr_imgs,
        scrs=orr_scrs,
        scr_thresh=scr_thresholds[orr_best_idx],
        les=orr_les,
        healthy_idx=healthy_idx,
        tum_rads=tum_rads,
    )
    orr_base_preds = get_scr_preds(
        imgs=orr_base_imgs,
        scrs=orr_base_scrs,
        scr_thresh=scr_thresholds[orr_base_best_idx],
        les=orr_base_les,
        healthy_idx=healthy_idx,
        tum_rads=tum_rads,
    )
    orr_spd_preds = get_scr_preds(
        imgs=orr_spd_imgs,
        scrs=orr_spd_scrs,
        scr_thresh=scr_thresholds[orr_spd_best_idx],
        les=orr_spd_les,
        healthy_idx=healthy_idx,
        tum_rads=tum_rads,
    )

    # Get the ROC AUC for each method and report to logger
    das_auc = get_auc(sens=das_res[:, 0], specs=das_res[:, 1])
    dmas_auc = get_auc(sens=dmas_res[:, 0], specs=dmas_res[:, 1])
    orr_auc = get_auc(sens=orr_res[:, 0], specs=orr_res[:, 1])
    orr_base_auc = get_auc(sens=orr_base_res[:, 0],
                           specs=orr_base_res[:, 1])
    orr_spd_auc = get_auc(sens=orr_spd_res[:, 0],
                          specs=orr_spd_res[:, 1])
    logger.info("DAS AUC: %.1f +/- %.1f" % das_auc)
    logger.info("DMAS AUC: %.1f +/- %.1f" % dmas_auc)
    logger.info("ORR AUC: %.1f +/- %.1f" % orr_auc)
    logger.info("ORR Base AUC: %.1f +/- %.1f" % orr_base_auc)
    logger.info("ORR Spd AUC: %.1f +/- %.1f" % orr_spd_auc)

    # Make ROC plots
    plt.figure(figsize=(10, 8))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=(20))
    plt.gca().set_aspect('equal', adjustable='box')
    cividis = get_cmap('inferno')
    # Plot the ROC curves when healthy references are used
    plt.plot(100 - das_res[:, 1], das_res[:, 0],
             c=cividis(0.0),
             linestyle='-',
             linewidth=2,
             label=r"DAS, Literature Standard "
                   r"%.0f $\mathdefault{\pm}$ %.0f" % das_auc
             )

    plt.plot(100 - dmas_res[:, 1], dmas_res[:, 0],
             c=cividis(0.8),
             linestyle='--',
             linewidth=2,
             label=r"DMAS, Literature Standard AUC: %.0f "
                   r"$\mathdefault{\pm}$ %.0f" % dmas_auc)

    plt.plot(100 - orr_base_res[:, 1], orr_base_res[:, 0],
             c=cividis(0.65),
             linewidth=2,
             linestyle='-',
             label=r"ORR, Novel Method AUC: %.0f $\mathdefault{\pm}$ %.0f"
                   % orr_base_auc
             )

    plt.plot(100 - orr_res[:, 1], orr_res[:, 0],
             c=cividis(0.4),
             linewidth=2,
             linestyle='-',
             label=r"ORR-EPM, AUC: %.0f $\mathdefault{\pm}$ %.0f" % orr_auc)

    plt.plot(100 - orr_spd_res[:, 1], orr_spd_res[:, 0],
             c='k',
             linewidth=2,
             linestyle='--',
             label=r"ORR-EPM-T, AUC: %.0f $\mathdefault{\pm}$ %.0f" % orr_spd_auc)

    # Plot the ROC curve of a random classifier
    plt.plot(np.linspace(0, 100, 100), np.linspace(0, 100, 100),
             c=[0, 108 / 255, 255 / 255], linestyle='--',
             label='Random Classifier')
    plt.grid()
    # Make the legend, axes labels, etc

    plt.xlim([0, 100])
    plt.ylim([0, 100])

    plt.scatter([100 - 95], [85], marker='v', color='r',
                label="Mammography, Mainly Fatty")
    plt.scatter([100 - 90], [85], marker='^', color='r',
                label="Mammography, Scattered")
    plt.scatter([100 - 87], [79], marker='x', color='r',
                label="Mammography, Heterogeneously Dense")
    plt.scatter([100 - 90], [65], marker='o', color='r',
                label="Mammography, Extremely Dense")
    plt.legend(fontsize=16,
               # loc='upper right'
               )
    plt.xlabel("False Positive Rate (%)", fontsize=22)
    plt.ylabel("True Positive Rate (%)", fontsize=22)
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(__O_DIR, "roc_orr.png"),
    #             dpi=300,
    #             transparent=False,
    #             )

    # Save metadata for statistical analysis
    save_pickle(md, os.path.join(__O_DIR, 'md_for_stat_analysis.pickle'))
    save_pickle(das_preds, os.path.join(__O_DIR, "das_scr_preds.pickle"))
    save_pickle(dmas_preds, os.path.join(__O_DIR, "dmas_scr_preds.pickle"))
    save_pickle(orr_base_preds,
                os.path.join(__O_DIR, "orr_base_scr_preds.pickle"))
    save_pickle(orr_preds, os.path.join(__O_DIR, "orr_epm_scr_preds.pickle"))
    save_pickle(orr_spd_preds, os.path.join(
        __O_DIR, "orr_epm_t_scr_preds.pickle"
    ))

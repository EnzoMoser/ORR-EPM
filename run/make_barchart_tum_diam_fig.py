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

from umbms.loadsave import load_pickle


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


def get_sens_by_tum(preds, tum_presence, tum_diams):
    """Get the diagnostic sensitivity for each tumour size

    Parameters
    ----------
    preds : array_like
        The diagnostic predictions for each reconstruction
    tum_presence : array_like
        Boolean entries indicating presence of tumour
    tum_diams : array_like
        The tumour diameter in the phantom used in each scan

    Returns
    -------
    sens : array_like
        The diagnostic sensitivity for each tumour size, as a
        percentage
    """

    # Find where the diagnosis was correct (for tumour-containing
    # images)
    correct_detect = (preds[tum_presence] == 1).astype('int')

    # Find the scans that included tumours
    non_nan_tum_diams = tum_diams[~np.isnan(tum_diams)]

    # Calculate the sensitivities
    sens = 100 * np.array(
        [np.sum(correct_detect[non_nan_tum_diams == 1])
            / np.sum(non_nan_tum_diams == 1),
         np.sum(correct_detect[non_nan_tum_diams == 1.5])
            / np.sum(non_nan_tum_diams == 1.5),
         np.sum(correct_detect[non_nan_tum_diams == 2.0])
            / np.sum(non_nan_tum_diams == 2.0),
         np.sum(correct_detect[non_nan_tum_diams == 2.5])
            / np.sum(non_nan_tum_diams == 2.5),
         np.sum(correct_detect[non_nan_tum_diams == 3.0])
            / np.sum(non_nan_tum_diams == 3.0),
         ]
    )

    return sens


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load metadata
    md = load_pickle(os.path.join(__O_DIR, 'md_for_stat_analysis.pickle'))

    # Load reconstruction predictions
    das_preds = load_pickle(os.path.join(__O_DIR, "das_scr_preds.pickle"))
    dmas_preds = load_pickle(os.path.join(__O_DIR, "dmas_scr_preds.pickle"))
    orr_base_preds = load_pickle(
        os.path.join(__O_DIR, "orr_base_scr_preds.pickle")
    )
    orr_epm_preds = load_pickle(
        os.path.join(__O_DIR, "orr_epm_scr_preds.pickle")
    )
    orr_spd_preds = load_pickle(
        os.path.join(__O_DIR, "orr_epm_t_scr_preds.pickle")
    )

    # Get array indicating tumour presence in each scan
    tum_presence = np.array(
        [~np.isnan(mm['tum_diam']) for mm in md]
    )

    # Get array indicating tumour size in each scan
    tum_diams = np.array(
        [mm['tum_diam'] for mm in md]
    )

    # --------- Do Stat analysis on -------------------------------------------

    das_sens = get_sens_by_tum(preds=das_preds,
                               tum_presence=tum_presence,
                               tum_diams=tum_diams)
    dmas_sens = get_sens_by_tum(preds=dmas_preds,
                                tum_presence=tum_presence,
                                tum_diams=tum_diams
                                )
    orr_base_sens = get_sens_by_tum(preds=orr_base_preds,
                                    tum_presence=tum_presence,
                                    tum_diams=tum_diams
                                    )
    orr_epm_sens = get_sens_by_tum(preds=orr_epm_preds,
                                   tum_presence=tum_presence,
                                   tum_diams=tum_diams
                                   )
    orr_spd_sens = get_sens_by_tum(preds=orr_spd_preds,
                                   tum_presence=tum_presence,
                                   tum_diams=tum_diams
                                   )

    cividis = get_cmap('cividis')

    # Make a bar chart
    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=18)
    plt.yticks(np.arange(0, 101, 10))
    plt.bar(x=[1, 7, 13, 19, 25],
            height=das_sens,
            label="DAS",
            color=cividis(0.0),
            edgecolor='black',
            zorder=2
            )
    plt.bar(x=[2, 8, 14, 20, 26],
            height=dmas_sens,
            label="DMAS",
            color=cividis(0.8),
            edgecolor='black',
            zorder=2
            )
    plt.bar(x=[3, 9, 15, 21, 27],
            height=orr_base_sens,
            label="ORR",
            color=cividis(0.65),
            edgecolor='black',
            zorder=2
            )
    plt.bar(x=[4, 10, 16, 22, 28],
            height=orr_epm_sens,
            label="ORR-EPM",
            color=cividis(0.4),
            edgecolor='black',
            zorder=2
            )
    plt.bar(x=[5, 11, 17, 23, 29],
            height=orr_spd_sens,
            label="ORR-EPM-T",
            color='k',
            edgecolor='black',
            zorder=2
            )
    plt.grid(axis='y', zorder=-1)

    plt.legend(fontsize=18,
               loc='lower right')
    plt.xticks([3, 9, 15, 21, 27],
               [10, 15, 20, 25, 30])
    plt.xlabel("Tumour Diameter (mm)", fontsize=22)
    plt.ylabel("Diagnostic Sensitivity (%)", fontsize=22)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(__O_DIR,
                             'sens_by_tum_diam.png'),
                dpi=300)

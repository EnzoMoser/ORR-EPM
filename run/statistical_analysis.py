"""
Tyson Reimer
University of Manitoba
August 30th, 2021
"""

import matplotlib

matplotlib.use('tkagg')

import os
import numpy as np

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle

import statsmodels.api as sm
from scipy.stats import norm
import pandas as pd

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


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    md = load_pickle(os.path.join(__O_DIR, 'md_for_stat_analysis.pickle'))
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

    # --------- Creating dataframe for stat analysis... -----------------------

    # Load phantom info file for phantom fib % vol and adi vol
    phant_info = np.genfromtxt(os.path.join(get_proj_path(), 'data/umbmid/',
                                            'phant_info.csv'),
                               delimiter=',',
                               dtype=['<U20', '<U20', float, float, float],
                               skip_footer=1,
                               )

    # All phantom IDs (ex: A2F1)
    phant_ids = np.array(['%s%s' % (ii[0], ii[1]) for ii in phant_info])

    phant_densities = dict()  # Init dict for storing fib % vol
    phant_vols = dict()  # Init dict for storing adi vol

    # Loop to store phantom densities and volumes
    for ii in range(len(phant_ids)):  # For each AXFY combo...

        # Store the phantom fibro % vol
        phant_densities[phant_ids[ii]] = 100 * phant_info[ii][2]

        # Store the adipose volume in [100 cm^3]
        phant_vols[phant_ids[ii]] = phant_info[ii][3] / (10**3)

    # Store metadata info as separate arrays
    densities = np.array(
        [phant_densities[mm['phant_id']] for mm in md]
    )
    adi_vols = np.array(
        [phant_vols[mm['phant_id']] for mm in md]
    )

    tum_presence = np.array(
        [~np.isnan(mm['tum_diam']) for mm in md]
    )
    tum_diams = np.array(
        [mm['tum_diam'] for mm in md]
    )
    tum_polar_rad = np.array(
        [np.sqrt(mm['tum_x']**2 + mm['tum_y']**2) for mm in md]
    )
    tum_zs = np.array(
        [mm['tum_z'] for mm in md]
    )

    unhealthy_df = pd.DataFrame()  # Init dataframe for tumour phants
    healthy_df = pd.DataFrame()  # Init dataframe for healthy phants

    # Store in healthy dataframe
    healthy_df['density'] = densities[~tum_presence]
    healthy_df['adi_vol'] = adi_vols[~tum_presence]

    # Store in unhealthy dataframe
    unhealthy_df['density'] = densities[tum_presence]
    unhealthy_df['adi_vol'] = adi_vols[tum_presence]
    unhealthy_df['tum_diam'] = tum_diams[tum_presence]
    unhealthy_df['tum_polar_rad'] = tum_polar_rad[tum_presence]

    # --------- Do Stat analysis on -------------------------------------------

    # Choose predictions to use for statistical analysis
    preds = orr_spd_preds

    unhealthy_df['pred_correct'] = (preds[tum_presence] == 1).astype('int')
    healthy_df['pred_correct'] = (preds[~tum_presence] == 0).astype('int')

    # Create logistic regression model
    healthy_model = sm.GLM.from_formula(
        "pred_correct ~  "
        " adi_vol "
        " + density",
        family=sm.families.Binomial(),
        data=healthy_df)
    healthy_results = healthy_model.fit()

    unhealthy_model = sm.GLM.from_formula(
        "pred_correct ~  "
        " adi_vol "
        " + density"
        " + tum_diam",
        family=sm.families.Binomial(),
        data=unhealthy_df)
    unhealthy_results = unhealthy_model.fit()

    # Report results
    logger.info('HEALTHY RESULTS...')
    logger.info(healthy_results.summary2())
    logger.info('\tp-values:')
    logger.info('\t\t%s' % healthy_results.pvalues)

    # Critical value - look at 95% confidence intervals
    zstar = norm.ppf(0.95)

    # Report odds ratio and significance level results
    for ii in healthy_results.params.keys():

        logger.info('\t%s' % ii)  # Print metadata info

        coeff = healthy_results.params[ii]
        std_err = healthy_results.bse[ii]

        odds_ratio = np.exp(coeff)  # Get odds ratio

        # Get 95% C.I. for odds ratio
        or_low = np.exp(coeff - zstar * std_err)
        or_high = np.exp(coeff + zstar * std_err)


        # Get p-val
        pval = healthy_results.pvalues[ii]

        logger.info('\t\tOdds ratio:\t\t\t%.3f\t(%.3f,\t%.3f)'
                    % (odds_ratio, or_low, or_high))
        logger.info('\t\tp-value:\t\t\t%.3e' % pval)

    # Report results
    logger.info('UNHEALTHY RESULTS...')
    logger.info(unhealthy_results.summary2())
    logger.info('\tp-values:')
    logger.info('\t\t%s' % unhealthy_results.pvalues)

    # Critical value - look at 95% confidence intervals
    zstar = norm.ppf(0.95)

    # Report odds ratio and significance level results
    for ii in unhealthy_results.params.keys():

        logger.info('\t%s' % ii)  # Print metadata info

        coeff = unhealthy_results.params[ii]
        std_err = unhealthy_results.bse[ii]

        odds_ratio = np.exp(coeff)  # Get odds ratio

        # Get 95% C.I. for odds ratio
        or_low = np.exp(coeff - zstar * std_err)
        or_high = np.exp(coeff + zstar * std_err)

        # Get p-val
        pval = unhealthy_results.pvalues[ii]

        logger.info('\t\tOdds ratio:\t\t\t%.3f\t(%.3f,\t%.3f)'
                    % (odds_ratio, or_low, or_high))
        logger.info('\t\tp-value:\t\t\t%.3e' % pval)


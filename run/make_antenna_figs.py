"""
Tyson Reimer
University of Manitoba
June 20th, 2023
"""

import os
import numpy as np
import matplotlib

# Use 'tkagg' to display plot windows, use 'agg' to *not* display
# plot windows
matplotlib.use('tkagg')
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path, get_script_logger

from umbms.loadsave import load_pickle

from umbms.hardware.antenna import (get_ant_gain, to_phase_center)

from umbms.recon.extras import get_pix_angs

###############################################################################

__D_DIR = os.path.join(get_proj_path(), 'data/umbmid/g3/')

# Output dir for saving
__O_DIR = os.path.join(get_proj_path(), "output/figs/")
verify_path(__O_DIR)


# The frequency parameters from the scan
__INI_F = 1e9
__FIN_F = 9e9
__N_FS = 1001
__SCAN_FS = np.linspace(__INI_F, __FIN_F, __N_FS)

__ROI_RAD = 8  # ROI radius, in [cm]
__M_SIZE = int(__ROI_RAD * 2 / 0.1)  # Image size

# The approximate radius of each adipose phantom in our array
__ADI_RADS = {
    'A1': 5,
    'A2': 6,
    'A3': 7,
    'A11': 6,
    'A12': 5,
    'A13': 6.5,
    'A14': 6,
    'A15': 5.5,
    'A16': 7,
}

# Polar coordinate phi of the antenna at each position during the scan
__ANT_PHIS = np.flip(np.linspace(0, 355, 72) + -136.0)

# List of the valid physics model enhancements
_valid_enh_phys = [
    'spherical',
    'beam',
    'gain',
    'sec_scat',
]


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # -------------------------------------------------------------------------
    # ---- BEAM PATTERN FIG

    # Get pixel angles in [rad]
    pix_angs = np.linspace(-90, 90, 1000)

    # Frequencies, [MHz], at which beam pattern is defined
    beam_fs = np.arange(start=2000, stop=9001, step=500)
    beam_fs = beam_fs[::7]
    gain_at_beam_fs = get_ant_gain(
        scan_fs=(beam_fs * 1e6),
        normalize=False,
    )
    gain_at_beam_fs = 10**(gain_at_beam_fs / 10)
    gain_at_beam_fs /= np.max(gain_at_beam_fs)
    tot_beam = np.zeros([__M_SIZE, __M_SIZE])
    recon_pix_angs = get_pix_angs(ant_rho=to_phase_center(21),
                                  m_size=__M_SIZE,
                                  roi_rho=__ROI_RAD,
                                  )

    plt.figure(figsize=(12,6.75))
    plt.rc('font', family='Times New Roman', size=24)
    ax = plt.subplot(111, polar=True)
    cmap = get_cmap('cividis')
    # ax.set_rorigin(-40)
    plt.subplots_adjust(left=0.001, right=0.99, top=0.99, bottom=0.01)
    colors = [cmap(0.0), cmap(0.4), cmap(0.8)]
    for ii in range(len(beam_fs)):

        interp_model = load_pickle(
            os.path.join(get_proj_path(),
                         'docs/hardware/beam_patterns/%dMHz_H.pickle'
                         % (beam_fs[ii])))

        # Get the beam factor in dB
        beam_fac = np.interp(x=pix_angs, xp=interp_model[:, 1],
                             fp=interp_model[:, 0])
        pix_beam_here = np.zeros_like(tot_beam)
        for aa in range(72):
            pix_beam_here += 10**(np.interp(x=recon_pix_angs[aa, :, :],
                                       xp=interp_model[:, 1],
                                       fp=interp_model[:, 0])
                             / 10
                             )
        tot_beam += gain_at_beam_fs[ii] * (pix_beam_here / 72)

        if ii % 2 == 0:
            linestyle = '-'
        else:
            linestyle = '--'

        ax.plot(np.deg2rad(pix_angs + 90),
                beam_fac,
                # color=cmap(ii / (len(beam_fs))),
                color=colors[ii],
                linestyle=linestyle,
                label="%.1f GHz" % (beam_fs[ii] / 1000),
                linewidth=4.0,
                )

    avg_beam = 10 * np.log10(tot_beam / len(beam_fs))

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]),
                  )
    ax.set_xticklabels([r"90$^{\circ}$",
                   r"60$^{\circ}$",
                   r"30$^{\circ}$",
                   r"0$^{\circ}$",
                   r"-30$^{\circ}$",
                   r"-60$^{\circ}$",
                   r"-90$^{\circ}$",
                   ])
    ax.set_rticks([-40, -30, -20, -10, 0],)
    ax.set_xlabel("Beam Pattern (dB)",
                  fontsize=22,
                  labelpad=-150,
                  )
    plt.legend(loc='lower center',
               ncol=4,
               bbox_to_anchor=(0.5, 0.1),
               shadow=True)
    ax.set_position([0.1, -0.45, 0.8, 2.0])
    plt.show()
    # plt.savefig(os.path.join(__O_DIR, "beam.png"),
    #             # bbox_inches='tight',
    #             # dpi=300
    #             )


    # -------------------------------------------------------------------------
    # ---- GAIN FIG

    recon_fs = __SCAN_FS[__SCAN_FS >= 2e9]
    recon_fs = np.linspace(np.min(recon_fs), np.max(recon_fs), 1000)

    # Get antenna gain
    gain = get_ant_gain(
        scan_fs=recon_fs,
        normalize=False,
    )

    plt.figure(figsize=(12,6))
    plt.rc("font", family='Times New Roman', size=24)
    plt.plot(recon_fs / 1e9, gain,
             color='k',
             linewidth=4.0,
             )
    plt.xlabel("Frequency (GHz)", fontsize=26)
    plt.ylabel("Antenna Gain (dBi)", fontsize=26)
    plt.xlim([2, 9])
    plt.grid()
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(__O_DIR, "gain.png"),
    #             dpi=300)
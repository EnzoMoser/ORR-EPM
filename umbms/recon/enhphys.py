"""
Tyson Reimer, Spencer Christie
University of Manitoba
June 08th, 2023
"""


from umbms.recon.extras import get_pix_ds


###############################################################################

# List of the valid physics model enhancements
_valid_enh_phys = [
    'spherical',
    'beam',
    'gain',
    'sec_scat',
    'speed',
    'complex_k'
]

###############################################################################


def sphere_wave_attn(ant_rho, roi_rho, m_size=150, n_ant_pos=72,
                     ini_ant_ang=-136.0):
    """

    Parameters
    ----------
    ant_rho : float
        Antenna rho used in scan, in [cm]
    roi_rho : float
        Radius of the region of interest that will define the spatial
        extent of the image-space, in [cm]
    m_size : int
        Size of image-space along one dimension
    n_ant_pos : int
        Number of antenna positions used in the scan
    ini_ant_ang : float
        Polar angle of initial antenna position

    Returns
    -------
    attn_fac : array_like
        The attenuation factor due to spherical wave attenuation
    """

    # Get the 1-way propagation distances in [cm]
    pix_ds = get_pix_ds(ant_rho=ant_rho, roi_rad=roi_rho,
                        m_size=m_size, n_ant_pos=n_ant_pos,
                        ini_ant_ang=ini_ant_ang,
                        )

    # Calculate the attenuation factor according to a spherical
    # wave model, 1 / r^2
    attn_fac = 1 / (pix_ds)**2

    return attn_fac

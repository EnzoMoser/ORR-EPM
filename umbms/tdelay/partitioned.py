"""
Illia Prykhodko, Tyson Reimer
University of Manitoba
October 18th, 2023
"""

import numpy as np
from functools import partial

from umbms.recon.extras import get_ant_scan_xys, get_pix_xys


###############################################################################


# TODO: Revise function, clean-up code and comments
def find_xy_ant_bound_circle(ant_xs, ant_ys, n_ants, pix_xs, pix_ys,
                             adi_rho, phant_x=0., phant_y=0.):
    """Finds breast boundary intersection coordinates of propag. rays

    Parameters
    ----------
    ant_xs : array_like Mx1
        Antenna x-coordinates, in [cm]
    ant_ys : array_like Mx1
        Antenna y-coordinates, in [cm]
    n_ants : int
        Number of antenna positions
    pix_xs :
        Positions of x-coordinates of each pixel, in [cm]
    pix_ys :
        Positions of y-coordinates of each pixel, in [cm]
    adi_rho : float
        Approximate radius of a phantom, in [cm]
    phant_x : float
        x_coord of the centre of the circle, in [cm]
    phant_y : float
        y_coord of the centre of the circle, in [cm]

    Returns
    ----------
    int_f_xs : array-like NxN
        x-coordinates of each front intersection
    int_f_ys : array-like NxN
        y-coordinates of each front intersection
    int_b_xs : array-like NxN
        x-coordinates of each back intersection
    int_b_ys : array-like NxN
        y-coordinates of each back intersection
    """

    # initializing arrays for storing intersection coordinates
    # front intersection - closer to antenna
    int_f_xs = np.empty([len(ant_xs), len(pix_xs), len(pix_ys)], dtype=float)
    int_f_ys = np.empty_like(int_f_xs)

    # back intersection - farther from antenna
    int_b_xs = np.empty_like(int_f_xs)
    int_b_ys = np.empty_like(int_f_xs)

    for a_pos in range(n_ants):  # For each antenna

        # singular antenna position
        ant_pos_x = ant_xs[a_pos]
        ant_pos_y = ant_ys[a_pos]

        # For each x-pixel
        for px_x, x in zip(pix_xs, range(len(pix_xs))):

            # For each y-pixel
            for px_y, y in zip(pix_ys, range(len(pix_ys))):

                # calculating coefficients of polynomial
                k = (ant_pos_y - px_y) / (ant_pos_x - px_x)
                a = k ** 2 + 1
                b = 2 * (k * px_y - k ** 2 * px_x - phant_x - k * phant_y)
                c = px_x ** 2 * k ** 2 - 2 * k * px_x * px_y + px_y ** 2 \
                    - adi_rho ** 2 + phant_x ** 2 + 2 * k * px_x * phant_y \
                    - 2 * px_y * phant_y + phant_y ** 2

                # calculating the roots
                x_roots = np.roots([a, b, c])
                y_roots = k * (x_roots - px_x) + px_y

                # flag to determine whether there are two real roots
                are_roots = np.logical_and(~np.iscomplex(x_roots)[0],
                                           ~np.iscomplex(x_roots)[1])

                if not are_roots:  # if no roots

                    # pixel coords are stored at both front and back
                    int_f_xs[a_pos, y, x], int_f_ys[a_pos, y, x] = px_x, px_y
                    int_b_xs[a_pos, y, x], int_b_ys[a_pos, y, x] = px_x, px_y

                else:  # if two real roots

                    # distance from centre to pixel
                    d_centre_pix = np.sqrt(
                        (px_x - phant_x) ** 2
                        + (px_y - phant_y) ** 2
                    )

                    # initializing the list of tuples for easier sorting
                    dtype = [('x', float), ('y', float)]
                    values = [(x_roots[0], y_roots[0]),
                              (x_roots[1], y_roots[1])]
                    roots = np.array(values, dtype=dtype)

                    # sort in ascending order wrt x_values
                    roots = np.sort(roots, order='x')

                    is_inside = d_centre_pix <= adi_rho
                    is_left = ant_pos_x < px_x

                    if is_inside:  # if pixel is inside the breast

                        # store one intersection point
                        # sort in ascending order wrt x_values
                        roots = np.sort(roots, order='x')

                        if is_left:  # if antenna is to the left of pixel

                            # store lower x_value as front intersection
                            (int_f_xs[a_pos, y, x],
                             int_f_ys[a_pos, y, x]) = roots[0]
                            # store pixel coords as back intersection
                            (int_b_xs[a_pos, y, x],
                             int_b_ys[a_pos, y, x]) = px_x, px_y

                        else:
                            # store higher x_value as front intersection
                            (int_f_xs[a_pos, y, x],
                             int_f_ys[a_pos, y, x]) = roots[1]
                            # store pixel coords as back intersection
                            (int_b_xs[a_pos, y, x],
                             int_b_ys[a_pos, y, x]) = px_x, px_y

                    else:  # if pixel is outside the breast

                        # calculate distance from antenna to pixel
                        d_pix_ant = np.sqrt((ant_pos_x - px_x) ** 2
                                            + (ant_pos_y - px_y) ** 2)

                        # distance from antenna to adjacent point on circle
                        d_ant_adj = np.sqrt((ant_pos_x - phant_x) ** 2
                                            + (ant_pos_y - phant_y) ** 2
                                            - adi_rho ** 2)

                        # flag to determine whether the pixel is
                        # in front of the breast
                        is_front = d_ant_adj >= d_pix_ant

                        if is_front:  # if pixel is in front

                            # store the same way as for no roots
                            (int_f_xs[a_pos, y, x],
                             int_f_ys[a_pos, y, x]) = px_x, px_y
                            (int_b_xs[a_pos, y, x],
                             int_b_ys[a_pos, y, x]) = px_x, px_y

                        else:  # if pixel is past the breast
                            roots = np.sort(roots, order='x')
                            if is_left:  # if antenna is to the left

                                # store lower x_value as front intersection
                                (int_f_xs[a_pos, y, x],
                                 int_f_ys[a_pos, y, x]) = roots[0]
                                # store higher x_value as back intersection
                                (int_b_xs[a_pos, y, x],
                                 int_b_ys[a_pos, y, x]) = roots[1]

                            else:  # if antenna is to the right

                                # store higher x_value as front intersection
                                (int_f_xs[a_pos, y, x],
                                 int_f_ys[a_pos, y, x]) = roots[1]
                                # store lower x_value as back intersection
                                (int_b_xs[a_pos, y, x],
                                 int_b_ys[a_pos, y, x]) = roots[0]

    return int_f_xs, int_f_ys, int_b_xs, int_b_ys


def _parallel_find_bound_circle_pix(ant_xs, ant_ys, n_ant_pos, pix_xs, pix_ys,
                                    adi_rad, ox, oy, idx):
    """Finds breast boundary intersection coordinates
    with propagation trajectory from antenna position
    to corresponding pixel (for parallel calculation)

    Parameters
    ----------
    ant_xs : array_like Mx1
        Antenna x-coordinates
    ant_ys : array_like Mx1
        Antenna y-coordinates
    n_ant_pos : int
        Number of antenna positions
    pix_xs : array_like
        Positions of x-coordinates of each pixel
    pix_ys : array_like
        Positions of y-coordinates of each pixel
    adi_rad : float
        Approximate radius of a phantom
    ox : float
        x_coord of the centre of the circle
    oy : float
        y_coord of the centre of the circle
    idx : int
        Index of the current parallel iteration

    Returns
    ----------
    int_f_xs : float
        x-coordinates of each front intersection
    int_f_ys : float
        y-coordinates of each front intersection
    int_b_xs : float
        x-coordinates of each back intersection
    int_b_ys : float
        y-coordinates of each back intersection
    """

    a_pos, y, x = np.unravel_index(idx,
                                   [n_ant_pos, np.size(pix_xs),
                                    np.size(pix_ys)])
    px_x = pix_xs[x]
    px_y = pix_ys[y]

    # singular antenna position
    ant_pos_x = ant_xs[a_pos]
    ant_pos_y = ant_ys[a_pos]

    # calculating coefficients of polynomial
    k = (ant_pos_y - px_y) / (ant_pos_x - px_x)
    a = k ** 2 + 1
    b = 2 * (k * px_y - k ** 2 * px_x - ox - k * oy)
    c = px_x ** 2 * k ** 2 - 2 * k * px_x * px_y + px_y ** 2 - adi_rad ** 2 \
        + ox ** 2 + 2 * k * px_x * oy - 2 * px_y * oy + oy ** 2

    # calculating the roots
    x_roots = np.roots([a, b, c])
    y_roots = k * (x_roots - px_x) + px_y

    # flag to determine whether there are two real roots
    are_roots = np.logical_and(~np.iscomplex(x_roots)[0],
                               ~np.iscomplex(x_roots)[1])

    if not are_roots:  # if no roots

        # pixel coords are stored at both front and back
        int_f_xs, int_f_ys = px_x, px_y
        int_b_xs, int_b_ys = px_x, px_y

    else:  # if two real roots

        # distance from centre to pixel
        d_centre_pix = np.sqrt((px_x - ox) ** 2 + (px_y - oy) ** 2)

        # initializing the list of tuples for easier sorting
        dtype = [('x', float), ('y', float)]
        values = [(x_roots[0], y_roots[0]), (x_roots[1], y_roots[1])]
        roots = np.array(values, dtype=dtype)

        # sort in ascending order wrt x_values
        roots = np.sort(roots, order='x')

        is_inside = d_centre_pix <= adi_rad
        is_left = ant_pos_x < px_x

        if is_inside:  # if pixel is inside the breast
            # store one intersection point
            # sort in ascending order wrt x_values
            roots = np.sort(roots, order='x')

            if is_left:  # if antenna is to the left of pixel

                # store lower x_value as front intersection
                int_f_xs, int_f_ys = roots[0]
                # store pixel coords as back intersection
                int_b_xs, int_b_ys = px_x, px_y

            else:
                # store higher x_value as front intersection
                int_f_xs, int_f_ys = roots[1]
                # store pixel coords as back intersection
                int_b_xs, int_b_ys = px_x, px_y

        else:  # if pixel is outside the breast

            # calculate distance from antenna to pixel
            d_pix_ant = np.sqrt((ant_pos_x - px_x) ** 2 +
                                (ant_pos_y - px_y) ** 2)
            # distance from antenna to adjacent point on circle
            d_ant_adj = np.sqrt((ant_pos_x - ox) ** 2 + (ant_pos_y - oy) ** 2
                                - adi_rad ** 2)

            # flag to determine whether the pixel is in front of the breast
            is_front = d_ant_adj >= d_pix_ant

            if is_front:  # if pixel is in front

                # store the same way as for no roots
                int_f_xs, int_f_ys = px_x, px_y
                int_b_xs, int_b_ys = px_x, px_y

            else:  # if pixel is past the breast
                roots = np.sort(roots, order='x')
                if is_left:  # if antenna is to the left

                    # store lower x_value as front intersection
                    int_f_xs, int_f_ys = roots[0]
                    # store higher x_value as back intersection
                    int_b_xs, int_b_ys = roots[1]

                else:  # if antenna is to the right

                    # store higher x_value as front intersection
                    int_f_xs, int_f_ys = roots[1]
                    # store lower x_value as back intersection
                    int_b_xs, int_b_ys = roots[0]

    return np.array([int_f_xs, int_f_ys, int_b_xs, int_b_ys])


###############################################################################


def get_phase_fac_partitioned(ant_rho,
                              m_size,
                              roi_rho,
                              n_ants=72,
                              ini_a_ang=-136.0,
                              adi_rad=0.0,
                              air_k=0.0,
                              breast_k=0.0,
                              int_f_xs=None,
                              int_f_ys=None,
                              int_b_xs=None,
                              int_b_ys=None,
                              phant_x=0.0,
                              phant_y=0.0,
                              worker_pool=None,
                              ):
    """Get one-way pixel phase factor using a binary partition model

    Calculates the phase factor for one-way propagation from each
    antenna position to each pixel, assuming that the breast is
    circular and homogeneous.

    Parameters
    ----------
    ant_rho : float
        Antenna radius, after correcting for the phase delay, in [cm]
    m_size : int
        Size of image-space along one dimension
    roi_rho : float
        Radius of the region of interest that will define the spatial
        extent of the image-space, in [cm]
    n_ants : int
        Number of antenna positions used in the scan
    ini_a_ang : float
        Polar angle of initial antenna position, in [deg]
    adi_rad : float
        Approximate radius of a phantom slice, in [cm]
    air_k : array_like
        The wave number in air, in units of [m^-1], defined at
        each frequency used in the scan
    breast_k : array_like
        The wave number in the breast, in units of [m^-1], defined
        at each frequency used in the scan
    int_f_xs : array_like n_ant_pos x m_size x m_size
        x_coords of front intersection
    int_f_ys : array_like n_ant_pos x m_size x m_size
        y_coords of front intersection
    int_b_xs : array_like n_ant_pos x m_size x m_size
        x_coords of back intersection
    int_b_ys : array_like n_ant_pos x m_size x m_size
        y_coords of back intersection
    phant_x : float
        x_coord of the centre of the phantom, in [cm]
    phant_y : float
        y_coord of the centre of the circle, in [cm]
    worker_pool : multiprocessing.pool.Pool
        Pool of workers for parallel computation

    Returns
    -------
    phase_factor :
        One-way phase factor that describes the response times of all
         pixels in the NxN image-space and (if k complex) the
         attenuation of the signal as it propagates.
    """

    # Get antenna x/y positions during scan
    ant_xs, ant_ys = get_ant_scan_xys(ant_rho=ant_rho,
                                      n_ants=n_ants,
                                      ini_a_ang=ini_a_ang)

    # Create arrays of pixel x/y positions
    pix_xs, pix_ys = get_pix_xys(m_size=m_size, roi_rho=roi_rho)

    # NOTE: The intersection points can be pre-calculated using
    # another function to improve computation speed

    if int_f_xs is None:  # If intersections not yet determined

        if worker_pool is not None:  # if using parallel computation

            # Calculate the intersection points of all rays at the
            # 'front' and 'back' edges of the breast
            int_f_xs, int_f_ys, int_b_xs, int_b_ys = \
                get_circle_intersections_parallel(
                    n_ants=n_ants,
                    m_size=m_size,
                    ant_xs=ant_xs,
                    ant_ys=ant_ys,
                    pix_xs=pix_xs,
                    pix_ys=pix_ys,
                    adi_rho=adi_rad,
                    phant_x=phant_x,
                    phant_y=phant_y,
                    worker_pool=worker_pool
                )

        else:  # If not using parallel computation

            # Calculate the intersection points of all rays at the
            # 'front' and 'back' edges of the breast
            int_f_xs, int_f_ys, int_b_xs, int_b_ys = find_xy_ant_bound_circle(
                ant_xs=ant_xs,
                ant_ys=ant_ys,
                n_ants=n_ants,
                pix_xs=pix_xs[0, :],
                pix_ys=pix_ys[:, 0],
                adi_rho=adi_rad,
                phant_x=phant_x,
                phant_y=phant_y,
                )

    # Calculate the phase factor
    phase_factor = calculate_phase_factor(
        n_ants=n_ants,
        m_size=m_size,
        ant_rho=ant_rho,
        pix_xs=pix_xs,
        pix_ys=pix_ys,
        int_f_xs=int_f_xs,
        int_f_ys=int_f_ys,
        int_b_xs=int_b_xs,
        int_b_ys=int_b_ys,
        air_k=air_k,
        breast_k=breast_k,
        ini_a_ang=ini_a_ang,
    )

    return phase_factor


def get_circle_intersections_parallel(n_ants, m_size, ant_xs, ant_ys,
                                      pix_xs, pix_ys, adi_rho,
                                      phant_x, phant_y,
                                      worker_pool):
    """Finds breast boundary intersection coordinates
    with propagation trajectory from antenna position
    to corresponding pixel (for parallel calculation)

    Parameters
    ----------
    n_ants : int
        Number of antenna positions
    m_size : int
        Size of image-space along one dimension
    ant_xs : array_like Mx1
        Antenna x-coordinates, in [cm]
    ant_ys : array_like Mx1
        Antenna y-coordinates, in [cm]
    pix_xs : array_like m_size x m_size
        Positions of x-coordinates of each pixel, in [cm]
    pix_ys : array_like m_size x m_size
        Positions of y-coordinates of each pixel, in [cm]
    adi_rho : float
        Approximate radius of a phantom, in [cm]
    phant_x : float
        x_coord of the centre of the circle, in [cm]
    phant_y : float
        y_coord of the centre of the circle, in [cm]
    worker_pool :
        Pool of workers for parallel computation

    Returns
    ----------
    int_f_xs : array_like n_ant_pos x m_size x m_size
        x-coordinates of each front intersection
    int_f_ys : array_like n_ant_pos x m_size x m_size
        y-coordinates of each front intersection
    int_b_xs : array_like n_ant_pos x m_size x m_size
        x-coordinates of each back intersection
    int_b_ys : array_like n_ant_pos x m_size x m_size
        y-coordinates of each back intersection
    """

    iterable_idx = range(n_ants * m_size * m_size)
    parallel_func = partial(_parallel_find_bound_circle_pix, ant_xs, ant_ys,
                            n_ants, pix_xs[0, :], pix_ys[:, 0], adi_rho,
                            phant_x, phant_y)

    # asynchronously find all the intersections
    intersections = np.array(worker_pool.map(parallel_func, iterable_idx))

    # the shape of the output is [n_ant_pos * m_size**2, 4]
    # the order is - [front x, front y, back x, back y]
    # C-style reshape each column to [n_ant_pos, m_size, m_size]
    int_f_xs = np.reshape(intersections[:, 0], [n_ants, m_size, m_size])
    int_f_ys = np.reshape(intersections[:, 1], [n_ants, m_size, m_size])
    int_b_xs = np.reshape(intersections[:, 2], [n_ants, m_size, m_size])
    int_b_ys = np.reshape(intersections[:, 3], [n_ants, m_size, m_size])

    return int_f_xs, int_f_ys, int_b_xs, int_b_ys


def calculate_phase_factor(
        n_ants,
        m_size,
        ant_rho,
        pix_xs,
        pix_ys,
        int_f_xs,
        int_f_ys,
        int_b_xs,
        int_b_ys,
        air_k,
        breast_k,
        ini_a_ang,
        ):
    """Calculate the phase factor for given set of phantom intersections

    Parameters:
    -------------
    n_ants : int
        Number of antenna positions used in the scan
    ant_rho : float
        Antenna radius, after correcting for phase center, [cm]
    m_size : int
        Size of image-space along one dimension
    pix_xs : array_like m_size x m_size
        A 2D arr. Each element in the arr contains the x-position of
        that pixel in the model, in meters
    pix_ys : array_like m_size x m_size
        A 2D arr. Each element in the arr contains the y-position of
        that pixel in the model, in meters
    int_f_xs : array-like n_ant_pos x m_size x m_size
        x-coordinates of each front intersection
    int_f_ys : array-like n_ant_pos x m_size x m_size
        y-coordinates of each front intersection
    int_b_xs : array-like n_ant_pos x m_size x m_size
        x-coordinates of each back intersection
    int_b_ys : array-like n_ant_pos x m_size x m_size
        y-coordinates of each back intersection
    air_speed : float
        The estimated propagation speed of the signal in air
    breast_speed : float
        The estimated propagation speed of the signal in phantom

    Returns:
    -------------
    phase_facs :
        One-way phase factor describing the response times and
        (if k complex) attenuation factors
    """

    # Get antenna x/y positions during scan
    ant_xs, ant_ys = get_ant_scan_xys(ant_rho=ant_rho,
                                      n_ants=n_ants,
                                      ini_a_ang=ini_a_ang)

    # Init array for storing pixel time-delays
    phase_facs = np.zeros([len(breast_k), n_ants, m_size, m_size],
                          dtype=complex)

    for aa in range(n_ants):  # For each antenna position

        # Find distances from each pixel to each to 'back' intersection
        pix_to_back_xs = pix_xs - int_b_xs[aa, :, :]
        pix_to_back_ys = pix_ys - int_b_ys[aa, :, :]

        # Find distances from each pixel to each to 'front' intersection
        back_to_front_xs = int_b_xs[aa, :, :] - int_f_xs[aa, :, :]
        back_to_front_ys = int_b_ys[aa, :, :] - int_f_ys[aa, :, :]

        # Find distances from each 'front' intersection to antenna
        front_to_ant_xs = int_f_xs[aa, :, :] - ant_xs[aa]
        front_to_ant_ys = int_f_ys[aa, :, :] - ant_ys[aa]

        # Calculate phase factor in air behind phantom, divide
        # by 100 to convert distance from [cm] to [m]
        air_td_back = (np.sqrt(pix_to_back_xs[None, :, :] ** 2
                               + pix_to_back_ys[None, :, :] ** 2)
                       / 100
                       * air_k[:, None, None])

        # Calculate phase factor in phantom, divide
        # by 100 to convert distance from [cm] to [m]
        breast_td = (np.sqrt(back_to_front_xs[None, :, :] ** 2
                             + back_to_front_ys[None, :, :] ** 2)
                     / 100
                     * breast_k[:, None, None])

        # Calculate phase factor in air in front of phantom, divide
        # by 100 to convert distance from [cm] to [m]
        air_td_front = (np.sqrt(front_to_ant_xs[None, :, :] ** 2
                                + front_to_ant_ys[None, :, :] ** 2)
                        / 100
                        * air_k[:, None, None])

        # Calculate total phase factor
        all_td = air_td_front + breast_td + air_td_back

        # Store results
        phase_facs[:, aa, :, :] = all_td[:, :, :]

    return phase_facs

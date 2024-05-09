"""
Tyson Reimer
University of Manitoba
November 08, 2018
"""

import numpy as np

from umbms.recon.extras import get_pixdist_ratio, get_ant_scan_xys


###############################################################################

# TODO: Revise this code to match rest of repo
def parallel_time_raytrace(pix_angs, pix_dists_from_center, speed_map, ant_xs,
                           ant_ys, ant_x_idxs, ant_y_idxs, pix_width,
                           first_plane_dist, possible_x_idxs, possible_y_idxs,
                           roi_rad, pix_dists_one_dimension, idx):
    """Get the time-of-flight from ant_pos to pix via parallelized ray-tracing

    Computes the propagation time-of-flight for the signal from one antenna
    position to one pixel in the image-space, and back. This function is
    designed to be used via parallel processing, with idx being from an
    iterable.

    Parameters
    ----------
    pix_angs : array_like
        The angle of each pixel off of the central-axis of the antenna, for
        each antenna position, in degrees
    pix_dists_from_center : array_like
        The polar distance from the center of the image-space to each pixel,
        for each antenna position, in meters
    speed_map : array_like
        Map of the estimated propagation speeds in the image-space, in m/s
    ant_xs : array_like
        The x-positions of each antenna position in the scan, in meters
    ant_ys : array_like
        The y-positions of each antenna position in the scan, in meters
    ant_x_idxs : array_like
        The x-coordinates of each antenna position in the scan (i.e., the
        x-indices of the pixels of each antenna position in the scan)
    ant_y_idxs : array_like
        The y-coordinates of each antenna position in the scan (i.e., the
        y-indices of the pixels of each antenna position in the scan)
    pix_width : float
        The physical width of each pixel in the image-space, in meters
    first_plane_dist : float
        The physical location of the first plane used to segment the
        image-space, in meters
    possible_x_idxs : array_like
        The arr of the possible x-indices of the intersected (intersected by
        the ray-of-propagation) pixels
    possible_y_idxs : array_like
        The arr of the possible y-indices of the intersected (intersected by
        the ray-of-propagation) pixels
    roi_rad : float
        The radius of the central circular region-of-interest, in meters -
        the time-of-flight is only computed for pixels within this central
        region, to save computation time
    pix_dists_one_dimension : array_like
        1D arr of the pixel positions, in meters, along either dimension of
        the image-space
    idx : int
        The current pixel-index for which the propagation time will be computed

    Returns
    -------
    1, or the true time-of flight of the signal
    """

    # Find the current antenna position and x/y-indices of the target pixel
    ant_pos, x_idx, y_idx = np.unravel_index(idx, np.shape(pix_angs))

    # If this pixel is both in front of the antenna and within the central
    # region-of-interest
    if (np.abs(pix_angs[ant_pos, x_idx, y_idx]) < 90 and
            (pix_dists_from_center[x_idx, y_idx] < roi_rad)):

        # Find the x/y-positions of the origin of this ray (i.e., the
        # x/y-positions of the antenna here)
        ray_ini_x = ant_xs[ant_pos]
        ray_ini_y = ant_ys[ant_pos]

        # Return the time-of-flight to this pixel
        return get_tof(speed_map, ant_x_idxs[ant_pos],
                       ant_y_idxs[ant_pos], x_idx, y_idx, ray_ini_x,
                       ray_ini_y, pix_dists_one_dimension[x_idx],
                       pix_dists_one_dimension[y_idx], pix_width,
                       first_plane_dist, possible_x_idxs,
                       possible_y_idxs)

    # If the pixel is either behind the antenna OR not in the central
    # region-of-interest, return 1 (the time-of-response is set to 1 second -
    # a very long time)
    else:
        return 1


def parallel_time_attn_raytrace(ref_coeffs, pix_angs, pix_dists_from_center,
                                speed_map, ant_xs, ant_ys, ant_x_idxs,
                                ant_y_idxs, pix_width, first_plane_dist,
                                possible_x_idxs, possible_y_idxs, roi_rad,
                                pix_dists_one_dimension, idx):
    """Get the time-of-flight and attenuation factor for a pixel

    Computes the propagation time-of-flight for the signal from one antenna
    position to one pixel in the image-space, and back, and the attenuation
    factor. This function is designed to be used via parallel processing, with
    idx being from an iterable.

    Parameters
    ----------
    ref_coeffs : array_like
        The reflection coefficients for each pixel in the image-space
    pix_angs : array_like
        The angle of each pixel off of the central-axis of the antenna,
        for each antenna position, in degrees
    pix_dists_from_center : array_like
        The polar distance from the center of the image-space to each pixel,
        for each antenna position, in meters
    speed_map : array_like
        Map of the estimated propagation speeds in the image-space, in m/s
    ant_xs : array_like
        The x-positions of each antenna position in the scan, in meters
    ant_ys : array_like
        The y-positions of each antenna position in the scan, in meters
    ant_x_idxs : array_like
        The x-coordinates of each antenna position in the scan (i.e., the
        x-indices of the pixels of each antenna position in the scan)
    ant_y_idxs : array_like
        The y-coordinates of each antenna position in the scan (i.e., the
        y-indices of the pixels of each antenna position in the scan)
    pix_width : float
        The physical width of each pixel in the image-space, in meters
    first_plane_dist : float
        The physical location of the first plane used to segment the
        image-space, in meters
    possible_x_idxs : array_like
        The arr of the possible x-indices of the intersected (intersected by
        the ray-of-propagation) pixels
    possible_y_idxs : array_like
        The arr of the possible y-indices of the intersected (intersected by
        the ray-of-propagation) pixels
    roi_rad : float
        The radius of the central circular region-of-interest, in meters -
        the time-of-flight is only computed for pixels within this central
        region, to save computation time
    pix_dists_one_dimension : array_like
        1D arr of the pixel positions, in meters, along either dimension of
        the image-space
    idx : int
        The current pixel-index for which the propagation time will be computed

    Returns
    -------
    (1, 1), or (true time of flight, true attenuation factor)
    """

    # Find the current antenna position, and the x/y indices of the target
    # pixel
    ant_pos, x_idx, y_idx = np.unravel_index(idx, np.shape(pix_angs))

    # If the target pixel is in front of the antenna and wihtin the central
    # region of interest
    if np.abs(pix_angs[ant_pos, x_idx, y_idx]) < 90 and \
            (pix_dists_from_center[x_idx, y_idx] < roi_rad):

        # Find the x/y-position of the origin of the ray here (i.e., the
        # x/y positions of the antenna position here)
        ray_ini_x = ant_xs[ant_pos]
        ray_ini_y = ant_ys[ant_pos]

        # Return the time of flight and the attenuation factor
        return get_tof_attn(speed_map, ref_coeffs, ant_x_idxs[ant_pos],
                            ant_y_idxs[ant_pos], x_idx, y_idx, ray_ini_x,
                            ray_ini_y, pix_dists_one_dimension[x_idx],
                            pix_dists_one_dimension[y_idx], pix_width,
                            first_plane_dist, possible_x_idxs,
                            possible_y_idxs)
    else:
        return 1, 1


def get_tof(speed_map, ini_x_idx, ini_y_idx, fin_x_idx, fin_y_idx, ini_x,
            ini_y, fin_x, fin_y, pix_width, first_plane_dist, possible_x_idxs,
            possible_y_idxs):
    """Ray-trace to obtain the time-of-flight (tof) of the return signal

    Uses the Siddon ray-tracing algorithm to compute the time-of-flight of the
    microwave signal from the starting pixel to the end pixel.

    Parameters
    ----------
    speed_map : array_like
        Map of the estimated propagation speeds of the microwave signal in
        the image-space, in m/s
    ini_x_idx : int
        The x-index of the pixel from which the ray originates
    ini_y_idx : int
        The y-index of the pixel from which the ray originates
    fin_x_idx : int
        The x-index of the pixel from which the ray terminates (i.e., the
        target pixel)
    fin_y_idx : int
        The y-index of the pixel from which the ray terminates (i.e., the
        target pixel)
    ini_x : float
        The physical x-position of the pixel from which the ray originates,
        in meters
    ini_y : float
        The physical y-position of the pixel from which the ray originates,
        in meters
    fin_x : float
        The physical x-position of the pixel at which the ray terminates
        (i.e., the target pixel), in meters
    fin_y : float
        The physical y-position of the pixel at which the ray terminates
        (i.e., the target pixel), in meters
    pix_width : float
        The physical width of each pixel in the image-space, in meters
    first_plane_dist : float
        The physical location of zeroth plane used to define the pixel-grid,
        in meters
    possible_x_idxs : array_like
        Array of the possible x-indices of the intersected pixels
    possible_y_idxs : array_like
        Array of the possible y-indices of the intersected pixels

    Returns
    -------
    tof : float
        The time-of-flight of the signal from the start position to the end
        position and back, in seconds
    """

    # Find the difference in the x/y positions of the start and end positions
    x_diff = fin_x - ini_x
    y_diff = fin_y - ini_y

    # If the start/end positions are the same, return 1 (a large
    # time-of-response)
    if x_diff == 0 and y_diff == 0:
        tof = 1

    # If the start/end x-positions are the same, but the start/end y-positions
    # are different
    elif x_diff == 0 and y_diff != 0:

        # If the end y-position is greater than the start y-position
        if y_diff > 0:

            # Get the x/y-indices of the intersected pixels
            intersected_y_idxs = np.arange(ini_y_idx, fin_y_idx + 1)
            intersected_x_idxs = ini_x_idx * np.ones_like(intersected_y_idxs)

            # Sum the propagation times for each intersection
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

        else:  # If the start y-position is greater than the end y-position

            # Get the x/y-indices of the intersected pixels
            intersected_y_idxs = np.arange(fin_y_idx, ini_y_idx + 1)
            intersected_x_idxs = ini_x_idx * np.ones_like(intersected_y_idxs)

            # Sum the propagation times for each intersection
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

    # If the start/end x-positions are different, but the start/end
    # y-positions are the same
    elif x_diff != 0 and y_diff == 0:

        # Find the distance the ray travels one-way
        ray_length = np.abs(x_diff)

        # If the end x-position is greater than the start x-position
        if x_diff > 0:

            # Get the x/y-indices of the intersected pixels
            intersected_x_idxs = np.arange(ini_x_idx, fin_x_idx + 1)
            intersected_y_idxs = ini_y_idx * np.ones_like(intersected_x_idxs)

            # Find the physical width step-size for this ray
            pix_width = ray_length / (len(intersected_x_idxs) + 1)

            # Sum the propagation times for each intersection
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

        else:  # If the start x-position is greater than the end x-position

            # Get the x/y-indices of the intersected pixels
            intersected_x_idxs = np.arange(fin_x_idx, ini_x_idx + 1)
            intersected_y_idxs = ini_y_idx * np.ones_like(intersected_x_idxs)

            # Find the physical width step-size for this ray
            pix_width = ray_length / (len(intersected_x_idxs) + 1)

            # Sum the propagation times for each intersection
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

    # If the start/end x-positions and y-positions are different
    else:

        # Find the alpha values along the x/y dimensions for the
        # intersections, as in the Siddon algorithm
        alpha_xs = (first_plane_dist + possible_x_idxs * pix_width
                    - ini_x) / x_diff
        alpha_ys = (first_plane_dist + possible_y_idxs * pix_width
                    - ini_y) / y_diff

        # Retain values between 0 and 1, corresponding to true pixel
        # intersection
        alpha_xs = alpha_xs[(alpha_xs > 0) & (alpha_xs < 1)]
        alpha_ys = alpha_ys[(alpha_ys > 0) & (alpha_ys < 1)]

        # Remove any non-unique values
        alpha_xys = np.unique(np.concatenate([alpha_xs, alpha_ys]))

        # Find the distance the ray propagates one-way
        ray_length = np.sqrt(x_diff**2 + y_diff**2)

        # Find the length of each intersection
        pixel_intersection_lengths = ray_length * (alpha_xys[1:]
                                                   - alpha_xys[:-1])

        # Find the alpha values at the middle of each pixel
        mid_pixel_alphas = 0.5 * (alpha_xys[1:] + alpha_xys[:-1])

        # Find the x/y-indices of each intersected pixel
        intersected_x_idxs = np.floor((ini_x + mid_pixel_alphas * x_diff
                                       - first_plane_dist)
                                      / pix_width).astype('int') - 1
        intersected_y_idxs = np.floor((ini_y + mid_pixel_alphas * y_diff -
                                       first_plane_dist)
                                      / pix_width).astype('int') - 1

        # Sum the propagation times through each pixel
        tof = np.sum(pixel_intersection_lengths /
                     speed_map[intersected_x_idxs, intersected_y_idxs])

    # Multiply the propagation time by 2 to account for propagation to the
    # target and back to the antenna
    tof *= 2

    return tof


def get_tof_attn(speed_map, ref_coeffs, ini_x_idx, ini_y_idx, fin_x_idx,
                 fin_y_idx, ini_x, ini_y, fin_x, fin_y, pix_width,
                 first_plane_dist, possible_x_idxs, possible_y_idxs):
    """Ray-trace to obtain the time-of-flight (tof) and attenuation factor

    Uses the Siddon ray-tracing algorithm to compute the time-of-flight and
    attenuation factor of the microwave signal from the starting pixel to
    the end pixel.

    Parameters
    ----------
    speed_map : array_like
        Map of the estimated propagation speeds of the microwave signal in
        the image-space, in m/s
    ref_coeffs : array_like
        Map of the reflection coefficients in the image-space
    ini_x_idx : int
        The x-index of the pixel from which the ray originates
    ini_y_idx : int
        The y-index of the pixel from which the ray originates
    fin_x_idx : int
        The x-index of the pixel from which the ray terminates (i.e.,
        the target pixel)
    fin_y_idx : int
        The y-index of the pixel from which the ray terminates (i.e.,
        the target pixel)
    ini_x : float
        The physical x-position of the pixel from which the ray originates,
        in meters
    ini_y : float
        The physical y-position of the pixel from which the ray originates,
        in meters
    fin_x : float
        The physical x-position of the pixel at which the ray terminates
        (i.e., the target pixel), in meters
    fin_y : float
        The physical y-position of the pixel at which the ray terminates
        (i.e., the target pixel), in meters
    pix_width : float
        The physical width of each pixel in the image-space, in meters
    first_plane_dist : float
        The physical location of zeroth plane used to define the pixel-grid,
        in meters
    possible_x_idxs : array_like
        Array of the possible x-indices of the intersected pixels
    possible_y_idxs : array_like
        Array of the possible y-indices of the intersected pixels

    Returns
    -------
    tof : float
        The time-of-flight of the signal from the start position to the end
        position and back, in seconds
    attn : float
        The attenuation factor for the signal from the start position to
        the end position
    """

    # Find the differences in the x/y-positions of the start and end positions
    x_diff = fin_x - ini_x
    y_diff = fin_y - ini_y

    # If the start/end positions are the same
    if x_diff == 0 and y_diff == 0:

        # Return a large value for the propagation time, and return a value
        # indicating no signal attenuation
        tof = 1
        attn = 0

    # If the start/end positions have the same x-position but different
    # y-positions
    elif x_diff == 0 and y_diff != 0:

        # Find the length the ray propagates one-way
        ray_length = np.abs(y_diff)

        # If the y-position of the end position is greater the start position
        if y_diff > 0:

            # Find the x/y-indices of the intersected pixels
            intersected_y_idxs = np.arange(ini_y_idx, fin_y_idx + 1)
            intersected_x_idxs = ini_x_idx * np.ones_like(intersected_y_idxs)

            # Find the distance step-size
            pix_width = ray_length / (len(intersected_y_idxs) + 1)

            # Sum the propagation times from each pixel
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

            # Multiply the attenuation factors through each pixel
            attn = np.prod((1 - ref_coeffs[intersected_x_idxs,
                                           intersected_y_idxs])**2)

        # If the y-position of the end position is lesser than the start
        # position
        else:

            # Find the x/y-indices of the intersected pixels
            intersected_y_idxs = np.arange(fin_y_idx, ini_y_idx + 1)
            intersected_x_idxs = ini_x_idx * np.ones_like(intersected_y_idxs)

            # Find the distance step-size
            pix_width = ray_length / (len(intersected_y_idxs) + 1)

            # Sum the propagation times through each pixel
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

            # Multiply the attenuation factors through each pixel
            attn = np.prod((1 - ref_coeffs[intersected_x_idxs,
                                           intersected_y_idxs])**2)

    # If the x-positions of the start/end positions are the same, but the
    # y-positions are different
    elif x_diff != 0 and y_diff == 0:

        # Find the length the ray propagates one-way
        ray_length = np.abs(x_diff)

        # If the x-position of the end position is greater than the start
        # position
        if x_diff > 0:

            # Find the x/y-indices of the intersected pixels
            intersected_x_idxs = np.arange(ini_x_idx, fin_x_idx + 1)
            intersected_y_idxs = ini_y_idx * np.ones_like(intersected_x_idxs)

            # Find the distance step-size
            pix_width = ray_length / (len(intersected_x_idxs) + 1)

            # Sum the propagation times through each pixel
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

            # Multiply the attenuation factors through each pixel
            attn = np.prod((1 - ref_coeffs[intersected_x_idxs,
                                           intersected_y_idxs])**2)

        # If the x-position of the end position is lesser than the start
        # position
        else:

            # Find the x/y-indices of the intersected pixels
            intersected_x_idxs = np.arange(fin_x_idx, ini_x_idx + 1)
            intersected_y_idxs = ini_y_idx * np.ones_like(intersected_x_idxs)

            # Find the distance step-size
            pix_width = ray_length / (len(intersected_x_idxs) + 1)

            # Sum the propagation times through each pixel
            tof = np.sum(pix_width / speed_map[intersected_x_idxs,
                                               intersected_y_idxs])

            # Multiply the attenuation factors through each pixel
            attn = np.prod((1 - ref_coeffs[intersected_x_idxs,
                                           intersected_y_idxs])**2)

    # If the x-positions and the y-positions of the start and end pixels
    # are both different
    else:

        # Find the alpha values along the x/y dimensions for the
        # intersections, as in the Siddon algorithm
        alpha_xs = (first_plane_dist + possible_x_idxs * pix_width
                    - ini_x) / x_diff
        alpha_ys = (first_plane_dist + possible_y_idxs * pix_width
                    - ini_y) / y_diff

        # Retain values between 0 and 1, corresponding to true
        # pixel intersection
        alpha_xs = alpha_xs[(alpha_xs > 0) & (alpha_xs < 1)]
        alpha_ys = alpha_ys[(alpha_ys > 0) & (alpha_ys < 1)]

        # Remove any non-unique values
        alpha_xys = np.unique(np.concatenate([alpha_xs, alpha_ys]))

        # Find the distance the ray propagates one-way
        ray_length = np.sqrt(x_diff**2 + y_diff**2)

        # Find the intersection lengths of the ray intersections in each pixel
        intersection_lengths = ray_length * (alpha_xys[1:] - alpha_xys[:-1])

        # Find the alpha values at the middle of each pixel
        mid_pixel_alphas = 0.5 * (alpha_xys[1:] + alpha_xys[:-1])

        # Find the x/y-indices of the intersected pixels
        intersected_x_idxs = np.floor((ini_x + mid_pixel_alphas * x_diff
                                       - first_plane_dist)
                                      / pix_width).astype('int') - 1
        intersected_y_idxs = np.floor((ini_y + mid_pixel_alphas * y_diff
                                       - first_plane_dist)
                                      / pix_width).astype('int') - 1

        # Sum the propagation times for the ray through each pixel
        tof = np.sum(intersection_lengths
                     / speed_map[intersected_x_idxs, intersected_y_idxs])

        # Multiply the attenuation factors through each pixel
        attn = np.prod((1 - ref_coeffs[intersected_x_idxs,
                                       intersected_y_idxs])**2)

    # Multiply the propagation time by two to account for propagation
    # to/from the target pixel
    tof *= 2

    return tof, attn


def get_ant_xy_idxs(ant_rad, n_ant_pos, m_size, ini_ant_ang=-130.0):
    """Returns the x,y-pixel indices of each antenna position

    Returns two vectors, containing the x- and y-coordinates
    (pixel indices) of the antenna positions during a scan.

    Parameters
    ----------
    ant_rad : float
        The radius of the trajectory of the antenna during the scan,
        in meters
    n_ant_pos : int
        The number of antenna positions used in the scan
    m_size : int
        The number of pixels along one dimension used to define the
        model
    ini_ant_ang : float
        The initial angle offset (in deg) of the antenna from the
        negative x-axis

    Returns
    -------
    ant_x_idxs : array_like
        The x-coordinates (pixel indices) of each antenna position used
        in the scan
    ant_y_idxs : array_like
        The y-coordinates (pixel indices) of each antenna position used
        in the scan
    """

    # Get ratio between pixel width and distance
    pixdist_ratio = get_pixdist_ratio(m_size, ant_rad)

    # Get the ant x/y positions
    ant_xs, ant_ys = get_ant_scan_xys(ant_rad, n_ant_pos,
                                      ini_a_ang=ini_ant_ang)

    # Convert the antenna x,y positions to x,y coordinates, store as ints so
    # they can be used for indexing later
    ant_x_idxs = np.floor(ant_xs * pixdist_ratio + m_size // 2).astype(int)
    ant_y_idxs = np.floor(ant_ys * pixdist_ratio + m_size // 2).astype(int)

    return ant_x_idxs, ant_y_idxs

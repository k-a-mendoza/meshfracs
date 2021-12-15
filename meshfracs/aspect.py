import numpy as np
from numba import njit


FLOAT_PRECISION = 1e-4
@njit
def correct_patch_aspect(initial_x_patch, initial_y_patch):
    #determine if square
    is_square = _patch_is_rectangular_check(initial_x_patch, initial_y_patch)
    #coordinates are square check
    if is_square:
        new_x_coordinates, new_y_coordinates = _quadrilateral_aspect_ratio_correction(initial_x_patch, initial_y_patch)
    else:
        new_x_coordinates, new_y_coordinates = _quadrilateral_aspect_ratio_correction(initial_x_patch,initial_y_patch)
    return new_x_coordinates, new_y_coordinates

@njit
def _quadrilateral_aspect_ratio_correction(initial_x_patch,initial_y_patch):
    """
    corrects a coordinate_modified patch to ensure the aspect ratio is at least 1:4
    """
    x_midpoint_bottom = np.sum(initial_x_patch[:,0])/2
    x_midpoint_top    = np.sum(initial_x_patch[:,1])/2

    x_midpoint_left   = np.sum(initial_x_patch[0,:])/2
    x_midpoint_right  = np.sum(initial_x_patch[1,:])/2

    y_midpoint_bottom = np.sum(initial_y_patch[:,0])/2
    y_midpoint_top    = np.sum(initial_y_patch[:,1])/2

    y_midpoint_left   = np.sum(initial_y_patch[0,:])/2
    y_midpoint_right  = np.sum(initial_y_patch[1,:])/2

    x_horizontal_delta = x_midpoint_left-x_midpoint_right
    y_horizontal_delta = y_midpoint_left-y_midpoint_right

    x_vertical_delta   = x_midpoint_top-x_midpoint_bottom
    y_vertical_delta   = y_midpoint_top-y_midpoint_bottom


    horizontal_distance = np.sqrt( x_horizontal_delta*x_horizontal_delta + \
                                   y_horizontal_delta*y_horizontal_delta)

    vertical_distance   = np.sqrt( x_horizontal_delta*x_horizontal_delta + \
                                   y_horizontal_delta*y_horizontal_delta)

    aspect_ratio_flag = _aspect_ratio_exceeds_criteria(horizontal_distance, vertical_distance)

    new_x_coordinates = initial_x_patch.copy()
    new_y_coordinates = initial_y_patch.copy()

    if aspect_ratio_flag==1:
        delta_y  = _get_max_coordinate_diff(initial_y_patch)
        new_x_coordinates[0,0] = -vertical_distance/8 + x_midpoint_bottom
        new_x_coordinates[1,0] =  vertical_distance/8 + x_midpoint_bottom
        new_x_coordinates[0,1] = -vertical_distance/8 + x_midpoint_top
        new_x_coordinates[1,1] =  vertical_distance/8 + x_midpoint_top
    elif aspect_ratio_flag==2:
        delta_x  = _get_max_coordinate_diff(initial_x_patch)
        new_y_coordinates[0,0] = -horizontal_distance/8 + y_midpoint_left
        new_y_coordinates[1,0] =  horizontal_distance/8 + y_midpoint_right
        new_y_coordinates[0,1] = -horizontal_distance/8 + y_midpoint_left
        new_y_coordinates[1,1] =  horizontal_distance/8 + y_midpoint_right
    return new_x_coordinates, new_y_coordinates

@njit
def _correct_square_aspect_ratio(initial_x_patch,initial_y_patch):
    """
    returns a modified patch which has an aspect ratio of at most 1:4
    using 2d coordinate grids which have been determined are rectangular

    """
    max_diff_x = _get_max_coordinate_diff(initial_x_patch)
    max_diff_y = _get_max_coordinate_diff(initial_y_patch)
    aspect_ratio_flag = _aspect_ratio_exceeds_criteria(max_diff_x, max_diff_y)
    new_x_coordinates = initial_x_patch.copy()
    new_y_coordinates = initial_y_patch.copy()
    if aspect_ratio_flag==1:
        delta_y  = _get_max_coordinate_diff(initial_y_patch)
        x_center = np.mean(initial_x_patch)
        new_x_coordinates[0,:] = x_center - delta_y/8
        new_x_coordinates[1,:] = x_center + delta_y/8
    elif aspect_ratio_flag==2:
        delta_x  = _get_max_coordinate_diff(initial_x_patch)
        y_center = np.mean(initial_y_patch)
        new_y_coordinates[0,:] = y_center - delta_x/8
        new_y_coordinates[1,:] = y_center + delta_x/8
    return new_x_coordinates, new_y_coordinates


@njit
def _aspect_ratio_exceeds_criteria(diff_x, diff_y, aspect_exceedence=1/4):
    """
    calculates the aspect exceedence condition based on coordinate differences

    Parameters
    ==========
    diff_x : np.ndarray
        x_coordinate cell dimensions

    diff_y : np.ndarray
        y_coordinates cell dimensions

    aspect_exceedence : float
        aspect exceedence criteria

    Returns
    =======
    aspect_exceedence_flag : int
        exceedence flag is 0 if the cell aspect is within the exceedence ratio.
        returns 1 if the x/y ratio is less than the aspect ratio
        returns 2 if the y/x ratio is less than the aspect ratio

    """

    aspect     = diff_x/diff_y

    upper_aspect = 1/aspect_exceedence

    if aspect > aspect_exceedence and aspect < upper_aspect:
        return 0
    elif aspect < aspect_exceedence:
        return 1
    else:
        return 2

@njit
def _determine_interpolate_mode(coordinate_array, sample_spacing=10_000,cutoff_number=350):
    """
    determines the number of interpolation points based on provided sample spacing.
    divides the max coordinate difference by a sample spacing.

    returns 3 if the max coordinate / sample spacing is less than 3, or
    cutoff_number if it exceeds cutoff_number.
    otherwise returns max_diff/sample_spacing as an integer

    Parameters
    ==========
    coordinate_array : np.ndarray
        a 2D array of coordinates
    sample_spacing : float
        spatial sampling distance in meters
    cutoff_number : int
        max number of samples to request from a dimension

    Returns
    =======
    samples : int
        number of samples needed to sample along that dimension.
    """

    max_diff = _get_max_coordinate_diff(coordinate_array)

    samples_needed = max_diff//sample_spacing
    if samples_needed < 3:
        samples_needed = 3
    elif samples_needed > cutoff_number:
        samples_needed = cutoff_number

    return samples_needed

@njit
def _get_max_coordinate_diff(coordinate_array):
    """
    given a 2d numpy array, finds the max difference between elements of the array
    """
    x0  = coordinate_array[0,0]
    x1  = coordinate_array[0,1]

    x2  = coordinate_array[1,0]
    x3  = coordinate_array[1,1]

    max_diff = max([abs(x3-x0),abs(x3-x1),abs(x3-x2),abs(x2-x0),abs(x2-x1),abs(x1-x0)])
    return max_diff

@njit
def _patch_is_rectangular_check(initial_x_patch, initial_y_patch):
    """
    determines if the patch coordinates are square

    """
    x0  = initial_x_patch[0,0]
    x1  = initial_x_patch[0,1]

    x2  = initial_x_patch[1,0]
    x3  = initial_x_patch[1,1]

    y0  = initial_y_patch[0,0]
    y1  = initial_y_patch[1,0]

    y2  = initial_y_patch[0,1]
    y3  = initial_y_patch[1,1]
    return abs(x0-x1)<FLOAT_PRECISION and abs(x2-x3)<FLOAT_PRECISION and \
           abs(y0-y1)<FLOAT_PRECISION and abs(y2-y3)<FLOAT_PRECISION

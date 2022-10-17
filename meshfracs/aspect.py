import numpy as np
from numba import njit


FLOAT_PRECISION = 1e-4
@njit
def correct_patch_aspect(initial_x_patch, initial_y_patch):
    TARGET_RATIO=1/100
    #determine if square
    is_square = patch_is_rectangular(initial_x_patch, initial_y_patch)
    #coordinates are square check
    if is_square:
        new_x_coordinates, new_y_coordinates = correct_square_aspect_ratio(initial_x_patch, initial_y_patch,
                                                TARGET_RATIO=TARGET_RATIO)
    else:
        new_x_coordinates, new_y_coordinates = correct_quadrilateral_aspect_ratio(initial_x_patch,initial_y_patch,
                                                TARGET_RATIO=TARGET_RATIO)
    return new_x_coordinates, new_y_coordinates

FLOAT_PRECISION = 1e-4
@njit
def get_patch_aspect(initial_x_patch, initial_y_patch):

    new_x, new_y = correct_patch_aspect(initial_x_patch, initial_y_patch)
    TARGET_RATIO=1/100
    #determine if square
    return _patch_aspect_only(new_x, new_y,TARGET_RATIO=TARGET_RATIO)



@njit
def correct_quadrilateral_aspect_ratio(initial_x_patch,initial_y_patch,TARGET_RATIO=1/4):
    """
    corrects a coordinate_modified patch to ensure the aspect ratio is at least 1:4
    """
    ADJUST_RATIO = TARGET_RATIO + TARGET_RATIO**3
    midpoint_top    = np.asarray([np.sum(initial_x_patch[:,1]), np.sum(initial_y_patch[:,1])])/2
    midpoint_bottom = np.asarray([np.sum(initial_x_patch[:,0]), np.sum(initial_y_patch[:,0])])/2
    midpoint_left   = np.asarray([np.sum(initial_x_patch[0,:]), np.sum(initial_y_patch[0,:])])/2
    midpoint_right  = np.asarray([np.sum(initial_x_patch[1,:]), np.sum(initial_y_patch[1,:])])/2

    x_dir_delta = midpoint_right-midpoint_left
    y_dir_delta = midpoint_top-midpoint_bottom

    horizontal_distance = np.sqrt(x_dir_delta[0]*x_dir_delta[0] + x_dir_delta[1]*x_dir_delta[1])
    vertical_distance   = np.sqrt(y_dir_delta[0]*y_dir_delta[0] + y_dir_delta[1]*y_dir_delta[1])

    aspect_ratio_flag = aspect_ratio_exceeds_criteria(horizontal_distance, vertical_distance,
                                                        TARGET_RATIO=TARGET_RATIO)
    new_x_coordinates = initial_x_patch.copy()
    new_y_coordinates = initial_y_patch.copy()

    coordinate_adjust_x, coordinate_adjust_y = _correct_coordinates(horizontal_distance, vertical_distance,
                                                                    aspect_ratio_flag,ADJUST_RATIO)
    new_x_coordinates+=coordinate_adjust_x
    new_y_coordinates+=coordinate_adjust_y
    return new_x_coordinates, new_y_coordinates

@njit
def _patch_aspect_only(initial_x_patch,initial_y_patch,TARGET_RATIO=1/4):
    """
    corrects a coordinate_modified patch to ensure the aspect ratio is at least 1:4
    """
    ADJUST_RATIO = TARGET_RATIO + TARGET_RATIO**3
    midpoint_top    = np.asarray([np.sum(initial_x_patch[:,1]), np.sum(initial_y_patch[:,1])])/2
    midpoint_bottom = np.asarray([np.sum(initial_x_patch[:,0]), np.sum(initial_y_patch[:,0])])/2
    midpoint_left   = np.asarray([np.sum(initial_x_patch[0,:]), np.sum(initial_y_patch[0,:])])/2
    midpoint_right  = np.asarray([np.sum(initial_x_patch[1,:]), np.sum(initial_y_patch[1,:])])/2

    x_dir_delta = midpoint_right-midpoint_left
    y_dir_delta = midpoint_top-midpoint_bottom

    horizontal_distance = np.sqrt(x_dir_delta[0]*x_dir_delta[0] + x_dir_delta[1]*x_dir_delta[1])
    vertical_distance   = np.sqrt(y_dir_delta[0]*y_dir_delta[0] + y_dir_delta[1]*y_dir_delta[1])

    aspect_ratio_flag = _aspect_ratio_float(horizontal_distance, vertical_distance,
                                                        TARGET_RATIO=TARGET_RATIO)
    return aspect_ratio_flag

@njit
def correct_square_aspect_ratio(initial_x_patch,initial_y_patch,TARGET_RATIO=1/4):
    """
    returns a modified patch which has an aspect ratio of at most 1:4
    using 2d coordinate grids which have been determined are rectangular

    """
    max_diff_x = get_max_coordinate_diff(initial_x_patch)
    max_diff_y = get_max_coordinate_diff(initial_y_patch)
    aspect_ratio_flag = aspect_ratio_exceeds_criteria(max_diff_x, max_diff_y,TARGET_RATIO=TARGET_RATIO)
    new_x_coordinates = initial_x_patch.copy()
    new_y_coordinates = initial_y_patch.copy()

    coordinate_adjust_x, coordinate_adjust_y = _correct_coordinates(max_diff_x, max_diff_y, 
                                                 aspect_ratio_flag,TARGET_RATIO=TARGET_RATIO)
   
    new_x_coordinates+=coordinate_adjust_x
    new_y_coordinates+=coordinate_adjust_y

    return new_x_coordinates, new_y_coordinates


@njit
def _correct_coordinates(max_diff_x, max_diff_y, aspect_ratio_flag, TARGET_RATIO=1/4):
    ratio = max_diff_x/max_diff_y
    coordinate_adjust_x= np.zeros((2,2))
    coordinate_adjust_y= np.zeros((2,2))
    if aspect_ratio_flag==0:
        return coordinate_adjust_x, coordinate_adjust_y
    if aspect_ratio_flag==1:
        delta = max_diff_y*(TARGET_RATIO - ratio)/2 
        coordinate_adjust_x[0,:]-= delta
        coordinate_adjust_x[1,:]+= delta
    elif aspect_ratio_flag==2:
        ratio = 1/ratio
        delta = max_diff_x*(TARGET_RATIO - ratio)/2
        coordinate_adjust_y[:,0]-= delta
        coordinate_adjust_y[:,1]+= delta

    return coordinate_adjust_x, coordinate_adjust_y


@njit
def aspect_ratio_exceeds_criteria(diff_x, diff_y, TARGET_RATIO=1/4):
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

    upper_aspect = 1/TARGET_RATIO

    if aspect >= TARGET_RATIO and aspect <= upper_aspect:
        return 0
    elif aspect < TARGET_RATIO:
        return 1
    else:
        return 2

@njit
def _aspect_ratio_float(diff_x, diff_y, TARGET_RATIO=1/4):
    return diff_x/diff_y

@njit
def determine_interpolate_mode(coordinate_array, sample_spacing=10_000,cutoff_number=350):
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

    max_diff = get_max_coordinate_diff(coordinate_array)

    samples_needed = max_diff//sample_spacing
    if samples_needed < 3:
        samples_needed = 3
    elif samples_needed > cutoff_number:
        samples_needed = cutoff_number

    return samples_needed

@njit
def get_max_coordinate_diff(coordinate_array):
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
def patch_is_rectangular(initial_x_patch, initial_y_patch):
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

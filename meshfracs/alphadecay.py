
import numpy as np
from numba import jit, njit

@njit
def decay_data_patch(alpha, data_patch, average_value=-4e3):
    """
    adjusts data to a fixed value based on weights provided by alpha

    Parameters
    ==========
    alpha : np.ndarray
        a numpy ndarray of same dimensions or broadcastable to the dimensions of data patch. must contain
        values between 0 and 1 representing how much of the original data to preserve, with 1 implying full
        preservation of original data.
    data_patch : np.ndarray
        a numpy ndarray representing the data to weight.

    average_value : float
        the average value to reduce the data to. the returned data will trend towards this value as alpha decreases
        to zero

    """
    target_patch = np.ones(data_patch.shape) * average_value
    new_patch    = target_patch*(1-alpha) + data_patch*alpha
    return new_patch

@njit
def get_l2norm_alpha(new_x_surface,new_y_surface,starting_distance=2.6e6,ending_distance=4e6):
    """
    creates an alpha weighting scheme based on an L2 norm metric. Weights values 1 within the starting distance,
    ramps values from 1 to zero in between radii, and sets weights to zero outside a given distance

    Parameters
    ==========
    new_x_surface : np.ndarray
        a numpy ndarray representing the x distance in meters from a center point, assumed to be 0,0
    new_y_surface : np.ndarray
        a numpy ndarray representing the y distance in meters from a center point, assumed to be 0,0

    starting_distance : float
        the inner radius in meters. L2 norm distances within this radius result in an alpha weight of 1.
    ending_distance : float
        the outer radius in meters. L2 norm distances outside this distance result in an alpha weight of 0.
        in between starting_distance and ending_distance values ramp linearly from 1 to 0.

    """
    x_length = new_x_surface.ravel()
    y_length = new_y_surface.ravel()

    alpha = np.ones(x_length.shape)
    delta = ending_distance - starting_distance
    distance = np.sqrt(x_length*x_length + y_length*y_length)

    intermediate_alpha = (distance > starting_distance) & (distance < ending_distance)

    ending_alpha       =  distance > ending_distance

    alpha[intermediate_alpha]= 1 - (distance[intermediate_alpha] - starting_distance)/delta

    alpha[ending_alpha]=0

    alpha = alpha.reshape(new_x_surface.shape)

    return alpha

@njit
def get_l1norm_alpha(x_surface,y_surface,x0=-4e6,x1=-2.6e6,x2=2.6e6,x3=4e6,
                                                 y0=-4e6,y1=-2.6e6,y2=2.6e6,y3=4e6):
    """
    creates an alpha weighting scheme based on an L1 norm metric. Weights values 1 within a flat region
    defined by
    x1-x2 & y1-y2.
    Ramps values from 1 to zero in between
    x0-x1, x2-x3, y0-y1, y2-y3
    Sets weights to zero outside these bounds

    Parameters
    ==========
    x_surface : np.ndarray
        a numpy ndarray representing the x distance in meters from a center point, assumed to be 0,0
    y_surface : np.ndarray
        a numpy ndarray representing the y distance in meters from a center point, assumed to be 0,0

    x0 : float
        starting x direction positive ramp
    x1 : float
        ending x direction positive ramp
    x2 : float
        starting x direction negative ramp
    x3 : float
        ending x direction negative ramp
    y0 : float
        starting y direction positive ramp
    y1 : float
        ending y direction positive ramp
    y2 : float
        starting y direction negative ramp
    y3 : float
        ending y direction negative ramp

    Returns
    =======
    alpha_weight_factor : numpy.ndarray
        a 2d array of alpha weight values
    """

    x_length = x_surface.ravel()
    y_length = y_surface.ravel()

    alpha_x = np.ones(x_length.shape)
    alpha_y = np.ones(x_length.shape)

    zero_mask = (x_length < x0) | (x_length > x3) | (y_length < y0) | (y_length > y3)
    ramp_x_positive = (x_length > x0) | (x_length < x1)
    ramp_x_negative = (x_length > x2) | (x_length < x3)
    ramp_y_positive = (y_length > y0) | (y_length < y1)
    ramp_y_negative = (y_length > y2) | (y_length < y3)

    alpha_x[ramp_x_positive] = (x_length[ramp_x_positive]     - x0)/(x1-x0)
    alpha_x[ramp_x_negative] = 1 - (x_length[ramp_x_negative] - x2)/(x3-x2)
    alpha_y[ramp_y_positive] = (y_length[ramp_y_positive]     - y0)/(y1-y0)
    alpha_y[ramp_y_negative] = 1 - (y_length[ramp_y_negative] - y2)/(y3-y2)

    alpha = alpha_x*alpha_y
    alpha[zero_mask]=0

    return alpha

import numpy as np
from numba import jit, njit

from .datainterp import interpolate_new_surface

@njit(fastmath=True)
def get_lowest_index(elevation_grid, sediment_thickness, z_array):
    deepest_point   = np.amin(elevation_grid-sediment_thickness)
    target_index    = 0
    for iz in range(z_array.shape[2]):
        z_min = np.amin(z_array[:,:,iz])
        diff = deepest_point-z_min
        if diff < 0:
            target_index = iz
    return target_index

@njit(fastmath=True)
def get_score_for_column(z_cell, square_xx, square_yy, topo_patch, sed_topo):
    """
    calculates the % solid and sediment at each index within an earth column

    Parameters
    ==========

    z_cell : np.ndarray
        a 2 x 2 x l array specifying the cell corner depths

    x_cell : np.ndarray
        a  2x2 array specifying the cell corner x coordinates

    y_cell : np.ndarray
        a 2x2 array specifying the cell corner y coordinates

    x_grid : np.ndarray
        a nxm array specifying x coordinate locations to evaluate the solid and sediment fractions
        on
    y_grid : np.ndarray
        a nxm array specifying y coordinate locations to evaluate the solid and sediment fractions
        on
    topo_patch : np.ndarray
        an nxm array specifying the z elevation of topography at the x and y grid coordinates
    sed_patch : np.ndarray
        an nxm array specifying the z thickness of sediment at the x and y grid coordinates

    Returns
    =======
    (solid_array, sediment_array) : tuple of npdarrays
        each returned array is of length l
    
    
    """
    FLOAT_PRECISION = 1e-4
    bottomost_z_index        = get_lowest_index(topo_patch, sed_topo, z_cell)
    column_depth             = z_cell.shape[2]
    solid_percent_column     = np.ones(column_depth)
    sediment_percent_column  = np.zeros(column_depth)

    below_sealevel    = np.any(topo_patch<0) # perform solid frac if True if any below sea level
    nonzero_sediment  = np.any(sed_topo>FLOAT_PRECISION)
    
    if not below_sealevel and not nonzero_sediment:
        return solid_percent_column, sediment_percent_column

    sed_surface   = topo_patch - sed_topo

    for z_index in range(bottomost_z_index):
        mesh_cube_z = z_cell[:,:,z_index:z_index+2]
        
        z_upper_surface = interpolate_new_surface(mesh_cube_z[:,:,0], square_xx, square_yy)
        z_lower_surface = interpolate_new_surface(mesh_cube_z[:,:,1], square_xx, square_yy)

        topo_zeroed   = topo_patch      - z_lower_surface 
        sed_zeroed    = sed_surface     - z_lower_surface 
        delta_z       = z_upper_surface - z_lower_surface

        topo_zeroed=_lthan2d(topo_zeroed,0.0,0.0)

        if np.all(sed_surface>=z_upper_surface):
            solid_intermediate  = 1.0
            sed_intermediate    = 0.0
        else:
            if np.all(topo_patch>z_upper_surface):
                solid_intermediate  = 1.0
            else:
                solid_intermediate = topo_zeroed/delta_z
                solid_intermediate = _lthan2d(solid_intermediate,0.0,0.0)
                solid_intermediate = _gthan2d(solid_intermediate,1.0,1.0)
                solid_intermediate = np.mean(solid_intermediate)

            if nonzero_sediment:
                sed_intermediate = 1-sed_zeroed/delta_z
                sed_intermediate = _lthan2d(sed_intermediate,0.0,0.0)
                sed_intermediate = _gthan2d(sed_intermediate,1.0,1.0)
                sed_intermediate = np.mean(sed_intermediate)
            else:
                sed_intermediate=0.0

        sed_intermediate+=(solid_intermediate-1)
        if sed_intermediate<0:
            sed_intermediate=0
        solid_percent_column[z_index]    = solid_intermediate
        sediment_percent_column[z_index] = sed_intermediate

    return solid_percent_column, sediment_percent_column


@njit(fastmath=True)
def _gthan2d(array,cutoff,assignment):
    shape = array.shape
    for ix in range(shape[0]):
        for iy in range(shape[1]):
            if array[ix,iy]>cutoff:
                array[ix,iy]=assignment
    return array


@njit(fastmath=True)
def _lthan2d(array,cutoff,assignment):
    shape = array.shape
    for ix in range(shape[0]):
        for iy in range(shape[1]):
            if array[ix,iy]<cutoff:
                array[ix,iy]=assignment
    return array



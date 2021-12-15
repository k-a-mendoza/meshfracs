import numpy as np
from numba import jit, njit

from datainterp import Regridder
def calculate_volume_fractions(regridder, bathymetry, sediment, z_grid, ix, iy):
    """
    This function calculates volume fractions for each cell in the mesh.

    Parameters
    ==========

    regridder : Regridder
        a regridder object which is used to subdivide a spatial domain

    bathymetry : np.ndarray
        a 2d array of dimensions 8x8 representing the bathymetric depth over a region as provided by GEBCO.
        bathymetry may be positive or negative and is in coordinates of meters z

    sediment : np.ndarray
        a 2d array of dimensions 8x8 representing the sediment depth over a region. sediment ranges from
        [0,n] where n is the maximum thickness of sediment in the dataset.

    z_grid : np.ndarray:
        a 2x2xn array representing the z coordinates of the mesh. ix and iy indexes represent the [0,0,:] coordinate
        of the grid

    ix : int
        the x index of the intended calculation
    iy : int
        the y index of the intended calculation

    Returns
    =======

    calculations_ignored : bool
        True when the sediment+bathymetry is entirely above the provided elevation patch. False if not.

    """

    lowest_index  = get_lowest_index(bathymetry-sediment, z_grid)
    if lowest_index==0:
        return True
    z_column      = z_grid[:,:,:lowest_index+3]

    solid_percent, sediment_percent = get_score_for_column(z_column,  bathymetry,
                                                           sediment, lowest_index, regridder)


    return False


@jit
def get_score_for_column(z_column, x_grid, y_grid, topo_patch, sed_topo, regridder: Regridder):
    """
    calculates sediment fraction and solid fraction for each cell in z_column. routine is unneccesarily verbose for
    performance reasons.

    Parameters
    ==========
    z_column : np.ndarray
        a 2x2x8 array of elevations representing the mesh depths
    bath_patch:
        an 8x8 array of elevations representing the bathymetric+topographic surface within the 2x2 cell
    sediment:
        an 8x8 array of sediment thicknesses representing the bathymetric surface within the 2x2 cell
    lowest_index : int
        the lowest integer along the z axis in which to perform this operation
    regridder: Regridder
        a regridder object

    Returns
    =======
    solid_percent_column: np.ndarray
        a 1-d array representing the solid fraction of the cell at every depth
    sediment_percent_column: np.ndarray
        a 1-d array representing the sediment fraction of the cell at every depth

    """
    FLOAT_PRECISION = 1e-4

    bottomost_z_index = get_lowest_index(topo_patch, sed_topo, z_column)
    
    below_sealevel    = np.any(topo_patch<0) #True if totally above sea level
    nonzero_sediment  = np.any(np.abs(sed_topo)>FLOAT_PRECISION)

    solid_percent_column, sediment_percent_column =_calculate_column_scores(z_column, x_grid, y_grid, bottomost_z_index, topo_patch, sed_topo, regridder,
                             evaluate_solid_percent= below_sealevel, evaluate_sediment_percent=nonzero_sediment)

    
    return solid_percent_column, sediment_percent_column

@njit
def _calculate_column_scores(z_column, x_grid, y_grid, bottomost_z_index, topo_patch, sed_topo, regridder,
                             evaluate_solid_percent=True, evaluate_sediment_percent=True):
    solid_percent_column     = np.ones(bottomost_z_index)
    sediment_percent_column  = np.zeros(bottomost_z_index)
    if not evaluate_sediment_percent and not evaluate_sediment_percent:
        return solid_percent_column, sediment_percent_column

    for z_index in range(0,bottomost_z_index):
        mesh_cube            = z_column[:,:,z_index:z_index+2]
        regridded_mesh_cube  = regridder.get_3d_cube(mesh_cube,z_index,x_grid, y_grid)
        solid_percent, sediment_percent  = _calculate_score(regridded_mesh_cube,
                                                           topo_patch,
                                                           sed_topo, evaluate_solid_percent=evaluate_solid_percent,
                                                           evaluate_sediment_percent=evaluate_sediment_percent)
        if evaluate_solid_percent:
            solid_percent_column[z_index]=solid_percent
        if evaluate_sediment_percent:
            sediment_percent_column[z_index]=sediment_percent

    return solid_percent_column, sediment_percent_column

@njit
def _calculate_score(regridded_mesh_cube, topo_patch, sed_topo, 
                     evaluate_solid_percent=True, evaluate_sediment_percent=True):
    shape3d = regridded_mesh_cube.shape
    total_dim = np.prod(shape3d)
    solid_percent    = 0
    sediment_percent = 0

    for i in range(shape3d[2]):
        z_slice = regridded_mesh_cube[:,:,i]
        if evaluate_solid_percent and not evaluate_sediment_percent:
            solid_percent    += (z_slice<topo_patch).sum()

        elif not evaluate_solid_percent and evaluate_sediment_percent:

            sediment_percent += (z_slice>sed_topo).sum()
        else:
            below_bathymetry  = z_slice<topo_patch
            above_sediment    = z_slice>sed_topo

            solid_percent    += (below_bathymetry).sum()
            sediment_percent += ( above_sediment & below_bathymetry).sum()

    solid_percent    = solid_percent/total_dim
    sediment_percent = sediment_percent/total_dim

    return solid_percent, sediment_percent

@njit
def get_lowest_index(elevation_grid, sediment_thickness, z_array):
    minz_across_xy  = np.min(np.min(z_array,axis=0),axis=0)
    deepest_point   = np.amin(elevation_grid-sediment_thickness)
    deepest_diffs   = deepest_point - minz_across_xy
    smallest_positive_diff = 1000000.0
    for diff in deepest_diffs:
        if diff > 0 and diff < smallest_positive_diff:
            smallest_positive_diff = diff
    iz = np.argwhere(deepest_diffs==smallest_positive_diff)
    return iz




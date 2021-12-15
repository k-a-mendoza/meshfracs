import warnings
warnings.filterwarnings("ignore")
import sys
import os
import numpy as np
import egi_utils.meshing as meshing
import geopandas as gpd
from pyproj import Proj, transform
from numba import jit, njit
import tqdm
import multiprocessing as mp
from multiprocessing import RawArray
import aspect
from dataaccess import GrayverData, ElevationPatch, Coastlines
from datainterp import Regridder
import alphadecay as adecay
import volumefractions
subfolder_extensions = ('solid_fraction','sediment_fraction','ocean_conductivity','sediment_conductivity')
var_dict = {}

def projection_definitions():
    longitude_center = -113
    latitude_center  = 39.75
    dictionary = dict(
    longitude_center = -113,
    latitude_center  = 39.75,
    wgs_84           = Proj('epsg:4326'),
    azimuth_equid    = Proj(proj='aeqd', datum='WGS84', lon_0=longitude_center, lat_0=latitude_center, units='m'))
    return dictionary


def convert_lonlat_into_aeqd(lon, lat):
    definitions = projection_definitions()
    x,y = transform(definitions['wgs_84'],
                    definitions['azimuth_equid'],lon,lat)
    return x, y

def convert_aeqd_to_latlon(x, y):
    definitions = projection_definitions()
    lat, lon = transform(definitions['azimuth_equid'],
                         definitions['wgs_84'],x,y)
    return lat, lon

def get_na_shapefile(shapefile):
    shapes = gpd.read_file(shapefile)
    shapes = shapes[shapes.name=='N. Am']
    return shapes

def return_memory_views_of_arrays(x_array,y_array,z_array):
    d2_grid_shape = x_array.shape
    d3_grid_shape = z_array.shape

    X = RawArray(np.ctypeslib.as_ctypes_type(x_array.dtype),
                d2_grid_shape[0]*d2_grid_shape[1])
    Y = RawArray(np.ctypeslib.as_ctypes_type(x_array.dtype),
                d2_grid_shape[0]*d2_grid_shape[1])
    Z = RawArray(np.ctypeslib.as_ctypes_type(x_array.dtype),
                d3_grid_shape[0]*d3_grid_shape[1]*d3_grid_shape[2])
    # Wrap X as an numpy array so we can easily manipulates its data.
    mp_x_grid = np.frombuffer(X).reshape(d2_grid_shape)
    mp_y_grid = np.frombuffer(Y).reshape(d2_grid_shape)
    mp_z_grid = np.frombuffer(Z).reshape(d3_grid_shape)
    # Copy data to our shared array.
    np.copyto(mp_x_grid, x_array)
    np.copyto(mp_y_grid, y_array)
    np.copyto(mp_z_grid, z_array)

    return X, Y, Z

# A global dictionary storing the variables passed from the initializer.
@jit
def patchwise_compute(sed_thickness,
                      sed_conductivity,
                      ocean_conductivity,
                      topography,
                      coastline,
                       x_grid, y_grid, z_grid):
   
    regridder = Regridder()
    shape2d   = x_grid.shape
    shape3d   = z_grid.shape

    solid_frac_xyz    = np.ones(shape3d)
    sediment_frac_xyz = np.zeros(shape3d) 

    ocean_conductivity_xy    = np.zeros(shape2d)
    sediment_conductivity_xy = np.zeros(shape2d)

    for ix in range(shape2d[0]):
        for iy in range(shape2d[1]):
            x_cell = x_grid[ix:ix+2,iy:iy+2]
            y_cell = y_grid[ix:ix+2,iy:iy+2]
            z_column = -z_grid[ix:ix+2,iy:iy+2,:]
            corrected_aeqd_patch_x, corrected_aeqd_patch_y = aspect.correct_patch_aspect(x_cell,
                                                                                         y_cell)


            new_x_surface, new_y_surface = regridder.get_new_coordinates(corrected_aeqd_patch_x,
                                                                        corrected_aeqd_patch_y)

            alpha      = adecay.get_l1norm_alpha(new_x_surface,new_y_surface)

            topo_patch = topography.get_values(new_x_surface,new_y_surface)
            topo_patch = adecay.decay_data_patch(alpha,topo_patch,average_value=-5e3)

            sed_topo   = sed_thickness.get_sediment_thickness(new_x_surface,new_y_surface)
            sed_topo   = adecay.decay_data_patch(alpha,sed_topo,average_value=0)

            sed_cond   = sed_conductivity.get_sediment_conductivity(new_x_surface,new_y_surface)
            sed_cond   = adecay.decay_data_patch(alpha,sed_cond,average_value=adecay.average)

            ocean_patch = ocean_conductivity.get_ocean_conductivity(new_x_surface,new_y_surface)
            ocean_patch = adecay.decay_data_patch(alpha,ocean_patch,average_value=ocean_conductivity.average)

            boolean_in_land = coastline.identify_land_points(new_x_surface, new_y_surface)
            topo_patch[boolean_in_land==1]=0

            # step 1, calculate volume fractions
            solid_frac_column, sediment_frac_column = volumefractions.get_score_for_column(z_column, 
                                                                new_x_surface, new_y_surface, 
                                                                topo_patch, sed_topo)
            # step 2, calculate surface fractions
            ocean_conductivity_xy[ix,iy]= np.mean(ocean_patch)
            sediment_conductivity_xy[ix,iy]= np.mean(sed_cond)


            z_depth = len(solid_frac_column)
            solid_frac_xyz[ix,iy,:z_depth]    = solid_frac_column
            sediment_frac_xyz[ix,iy,:z_depth] = sediment_frac_column

    return solid_frac_xyz, sediment_frac_xyz, ocean_conductivity_xy, sediment_conductivity_xy

def compute_code_grid(args):
    """
    for every provided ix and iy index, calculates
        * the average ocean and sediment conductivity at that location
        * the average % land per cell by numerical approximation

    It also saves these to file

    Parameters:

    args : iterable
        an iterable argument list with the following (ordered)

        index
        =====
        0 -- list of index tuples to iterate over
        1 -- longitude grid of the entire mesh
        2 -- latitude grid of the entire mesh
        3 -- x mesh-centric coordinates of the entire mesh
        4 -- y mesh-centric coordinates of the entire mesh
        5 -- z depth coordinates of the entire mesh
        6 -- a regridder object
        7 -- an ElevationPatch creator object
        8 -- an interpolator object used to access Grayver's conductivity data


    """

    x_grid = np.frombuffer(var_dict['x_grid']).reshape(var_dict['shape2d'])
    y_grid = np.frombuffer(var_dict['y_grid']).reshape(var_dict['shape2d'])
    z_grid = np.frombuffer(var_dict['z_grid']).reshape(var_dict['shape3d'])
    output_dirs = var_dict['output_dirs']


    (x_patch, y_patch) = args[0]
    ix0 = x_patch[0]
    ix1 = x_patch[1]
    iy0 = y_patch[0]
    iy1 = y_patch[1]

    grayver_data   = args[1]
    elevation_data = args[2]
    coastline_data = args[3]

    x_domain = np.copy(x_grid[ix0:ix1,iy0:iy1])
    y_domain = np.copy(y_grid[ix0:ix1,iy0:iy1])
    z_domain = np.copy(z_grid[ix0:ix1,iy0:iy1,:])

    sediment_thickness_interpolator = grayver_data.get_interpolator(type='sediment thickness',
                                      x_interp_grid=x_domain,
                                      y_interp_grid=y_domain)
    sediment_conductivity_interpolator = grayver_data.get_interpolator(type='sediment conductivity',
                                         x_interp_grid=x_domain,
                                         y_interp_grid=y_domain)
    ocean_conductivity_interpolator = grayver_data.get_interpolator(type='ocean conductivity',
                                         x_interp_grid=x_domain,
                                         y_interp_grid=y_domain)
    elevation_interpolator = elevation_data.get_elevation_interpolator(x_domain, y_domain)

    solid_frac, sediment_frac, ocean_conductivity, sediment_conductivity = patchwise_compute(\
                                sediment_thickness_interpolator,
                                sediment_conductivity_interpolator,
                                ocean_conductivity_interpolator,
                                elevation_interpolator, coastline_data,
                                x_domain, y_domain, z_domain)

    index_string = f'{ix0}:{ix1}_{iy0}:{iy1}'
    np.save(f'{output_dirs["solid_fraction"]}{os.sep}{index_string}_solidfraction',solid_frac)
    np.save(f'{output_dirs["sediment_fraction"]}{os.sep}{index_string}_sedimentfraction',sediment_frac)
    np.save(f'{output_dirs["ocean_conductivity"]}{os.sep}{index_string}_oceanconductivity',ocean_conductivity)
    np.save(f'{output_dirs["sediment_conductivity"]}{os.sep}{index_string}_sedimentconductivity',sediment_conductivity)

    return None


def init_worker(X,Y,Z, shape2d, shape3d, output_dirs):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['x_grid'] = X
    var_dict['y_grid'] = Y
    var_dict['z_grid'] = Z
    var_dict['shape2d'] = shape2d
    var_dict['shape3d'] = shape3d
    var_dict['output_dirs'] = output_dirs

def prepare_code_grid(directory_strings):
    """
    This algorithm assigns parameter codes to the mesh

    Codes are as follows:

    0: land parameter
    1-8: possible sea parameters. 1 indicates a cell is 1/8th in the ocean,
        whereas 8 indicates it is wholly in the ocean.
    9: air
    10: background
    11: edge. Edge nodes occupy the last index plane in every direction


    Codes are assigned to the South, West, top node index.


    """

    mesh_directory      = directory_strings['mesh_dir']
    elevation_directory = directory_strings['elevation_directory']
    shoreline_gpd       = get_na_shapefile(directory_strings['shoreline_shp'])
    output_dirs         = directory_strings['output_dirs']
    grayver_dir         = directory_strings['grayver_dir']
    
    print('loading mesh')
    mesh = meshing.MeshFromFile()
    mesh.load(mesh_directory,type='npy')
    print('creating iterable arguments')
    worker_count = 46
    gebco_data          = elevation_directory
    projection          = projection_definitions()['azimuth_equid']
    projection_func     = convert_lonlat_into_aeqd

    surface_index  = np.argwhere(np.squeeze(mesh.z_grid[0,0,:])==0)
    z_domain_limit = np.argmax(np.squeeze(mesh.z_grid[0,0,:])<30_000)

    x_array = mesh.x_grid[:,:,0]
    y_array = mesh.y_grid[:,:,0]
    z_array = mesh.z_grid[:,:,surface_index:z_domain_limit]

    d2_grid_shape = x_array.shape
    d3_grid_shape = z_array.shape

    x_array_length = list(range(x_array.shape[0]))
    y_array_length = list(range(x_array.shape[1]))

    worker_count_2 = int(np.floor(np.sqrt(2*worker_count)))
    x_divisions = np.array_split(x_array_length, worker_count_2)
    x_division_limits = [(np.amin(x),np.amax(x)+1) for x in x_divisions]
    y_divisions = np.array_split(y_array_length, worker_count_2)
    y_division_limits = [(np.amin(x),np.amax(x)+1) for x in y_divisions]

    iterable_arguments = []

    for x_patch, y_patch in zip(x_division_limits, y_division_limits):
        iterable_arguments.append(
            (x_patch,y_patch),
            GrayverData(projection_func, path=grayver_dir),
            ElevationPatch(gebco_data, projection),
            Coastlines(shoreline_gpd, projection)
        )

    print('creating shared variables')
    x_buffer, y_buffer, z_buffer = return_memory_views_of_arrays(x_array,y_array,z_array)

    print(f'launching {worker_count} workers')
    init_args = (x_buffer, y_buffer, z_buffer, d2_grid_shape, d3_grid_shape, output_dirs)
    with mp.Pool(worker_count, initializer=init_worker, initargs=init_args) as p:
        list(tqdm.tqdm(p.imap(compute_code_grid,args),
                       total=len(args),
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',smoothing=0))


if __name__ == '__main__':
    args = sys.argsv[:]
    #first arg should be target mesh file
    # second arg should be target
    mesh_dir       = args[0]
    gebco_dir      = args[1]
    land_shape_dir = args[2]
    grayver_dir    = args[3]
    #'shoreline/edited_shoreline_us_III.shp'
    mesh_name = mesh_dir.split(os.sep)[-1]
    cwd = os.getcwd()
    directory_strings = {
        'mesh_directory':mesh_dir,
        'elevation_directory':gebco_dir,
        'shoreline_shp':land_shape_dir,
        'grayver_dir':grayver_dir,
        'mesh_name':mesh_name,
        'output_dirs':{}
    }

    output_folder=f'{cwd}{os.sep}{mesh_name}_conductivity_data'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for extension in subfolder_extensions:
        new_path = f'{output_folder}{os.sep}{extension}'
        directory_strings['output_dirs'][extension]=new_path
        if not os.path.exists(new_path):
            os.makedirs(new_path)

    prepare_code_grid(directory_strings)

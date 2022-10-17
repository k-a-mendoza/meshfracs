#import warnings
#warnings.filterwarnings("ignore")
import os
from re import X
import numpy as np
import egi_utils.meshing as meshing
import geopandas as gpd
from pyproj import Proj, transform
import tqdm
from .patchcompute import compute_patches, compute_ratios, compute_aspect_correction_x, compute_aspect_correction_y
import multiprocessing as mp
from multiprocessing import RawArray
from .dataaccess import GrayverInterpolatorFactory, ElevationSplineFactory

from .workerbalance import create_worker_patches
import time
subfolder_extensions = ('solid_fraction','sediment_fraction','ocean_conductivity',
                        'sediment_conductivity','ratio_map','aspect_map_x-','aspect_map_x+',
                        'aspect_map_y-','aspect_map_y+')
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
    debug_mode  = var_dict['debug']
    compute_mode = var_dict['compute_mode']

    (x_patch, y_patch) = args[0]
    ix0 = x_patch[0]
    ix1 = x_patch[1]
    iy0 = y_patch[0]
    iy1 = y_patch[1]
    idx = ix1-ix0
    idy = iy1-iy0

    grayver_data_factory   = args[1]
    elevation_factory      = args[2]

    x_domain = np.copy(x_grid[ix0:ix1+2,iy0:iy1+2])
    y_domain = np.copy(y_grid[ix0:ix1+2,iy0:iy1+2])
    z_domain = np.copy(z_grid[ix0:ix1+2,iy0:iy1+2,:])
    if compute_mode =='default':
        default_compute(output_dirs, debug_mode, ix0, ix1, iy0, iy1, idx, idy, 
                    grayver_data_factory, elevation_factory, x_domain, y_domain, z_domain)
    elif compute_mode =='ratios':
        ratio_compute(output_dirs, ix0, ix1, iy0, iy1, idx, idy, x_domain, y_domain)
    elif compute_mode =='aspect':
        aspect_correction_compute(output_dirs, ix0, ix1, iy0, iy1, idx, idy, x_domain, y_domain)
    return None

def default_compute(output_dirs, debug_mode, ix0, ix1, iy0, iy1, idx, idy, grayver_data_factory, 
                    elevation_factory, x_domain, y_domain, z_domain):

    if debug_mode:
        timing_dict = {
            'data access objects':0,
            'patch calculation':0,
            'array saving':0
            }
        t0 = time.perf_counter()

    sediment_thickness_interpolator = grayver_data_factory.get_interpolator(type='sediment thickness',
                                      x_points=x_domain,
                                      y_points=y_domain)
    sediment_conductivity_interpolator = grayver_data_factory.get_interpolator(type='sediment conductivity',
                                      x_points=x_domain,
                                      y_points=y_domain)
    ocean_conductivity_interpolator = grayver_data_factory.get_interpolator(type='ocean conductivity',
                                      x_points=x_domain,
                                      y_points=y_domain)

    elevation_interpolator = elevation_factory.get_interpolator(x_domain, y_domain)

    if debug_mode:
        dt = time.perf_counter()-t0
        timing_dict['data access objects']+=dt
        t0 = time.perf_counter() 

    solid_frac, sediment_frac, ocean_conductivity, sediment_conductivity = compute_patches(\
                                sediment_thickness_interpolator,
                                sediment_conductivity_interpolator,
                                ocean_conductivity_interpolator,
                                elevation_interpolator,
                                x_domain, y_domain, z_domain,debug_mode)
    solid_frac            = solid_frac[:idx,:idy,:]
    sediment_frac         = sediment_frac[:idx,:idy,:]
    ocean_conductivity    = ocean_conductivity[:idx,:idy]
    sediment_conductivity = sediment_conductivity[:idx,:idy]
    if debug_mode:
        dt = time.perf_counter()-t0
        timing_dict['patch calculation']+=dt
        t0 = time.perf_counter()

    index_string = f'{ix0}:{ix1}_{iy0}:{iy1}'
    
    np.save(f'{output_dirs["solid_fraction"]}{os.sep}{index_string}_solidfraction',solid_frac)
    np.save(f'{output_dirs["sediment_fraction"]}{os.sep}{index_string}_sedimentfraction',sediment_frac)
    np.save(f'{output_dirs["ocean_conductivity"]}{os.sep}{index_string}_oceanconductivity',ocean_conductivity)
    np.save(f'{output_dirs["sediment_conductivity"]}{os.sep}{index_string}_sedimentconductivity',sediment_conductivity)
    if debug_mode:
        dt = time.perf_counter()-t0
        timing_dict['array saving']+=dt
        print('total time (s) per operation\n')
        for key, value in timing_dict.items():
            print(f"\t {key}:{value}")

def ratio_compute(output_dirs, ix0, ix1, iy0, iy1, idx, idy, x_domain, y_domain):

    patch_ratio = compute_ratios(x_domain, y_domain)[:idx,:idy]
    index_string = f'{ix0}:{ix1}_{iy0}:{iy1}'
    np.save(f'{output_dirs["ratio_map"]}{os.sep}{index_string}_ratios',patch_ratio)

def aspect_correction_compute(output_dirs, ix0, ix1, iy0, iy1, idx, idy, x_domain, y_domain):
    index_string = f'{ix0}:{ix1}_{iy0}:{iy1}'
    x_minus, x_plus = compute_aspect_correction_x(x_domain, y_domain)
    y_minus, y_plus = compute_aspect_correction_y(x_domain, y_domain)
    np.save(f'{output_dirs["aspect_map_x-"]}{os.sep}{index_string}_x-',x_minus[:idx,:idy])
    np.save(f'{output_dirs["aspect_map_x+"]}{os.sep}{index_string}_x+',x_plus[:idx,:idy])
    np.save(f'{output_dirs["aspect_map_y-"]}{os.sep}{index_string}_y-',y_minus[:idx,:idy])
    np.save(f'{output_dirs["aspect_map_y+"]}{os.sep}{index_string}_y+',y_plus[:idx,:idy])

def init_worker(X,Y,Z, shape2d, shape3d, output_dirs, debug,compute_mode):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['x_grid'] = X
    var_dict['y_grid'] = Y
    var_dict['z_grid'] = Z
    var_dict['shape2d'] = shape2d
    var_dict['shape3d'] = shape3d
    var_dict['output_dirs'] = output_dirs
    var_dict['debug']=debug
    var_dict['compute_mode']=compute_mode

def prepare_code_grid(directory_strings, debug_mode=False, worker_div_multiplier=1, worker_count=46,
                      compute_mode='default'):
    """
    Initializes memory buffer and launches workers to calculate % Earth, % sediment within the Earth,
    avg sediment conductivity and avg seawater conductivity per cell. 


    """

    mesh_directory      = directory_strings['mesh_directory']
    elevation_directory = directory_strings['elevation_directory']
    output_dirs         = directory_strings['output_dirs']
    conductance_file    = directory_strings['conductance_file']
    
    print(f'loading mesh: {mesh_directory}')
    print(f'current working directory is {os.getcwd()}')
    mesh = meshing.MeshFromFile()
    mesh.load(mesh_directory,type='npy')
    print('creating iterable arguments')
    gebco_data          = elevation_directory
    projection          = projection_definitions()['azimuth_equid']

    surface_index  = np.argwhere(np.squeeze(mesh.z_grid[0,0,:])==0)[0][0]
    z_domain_limit = np.argmax(np.squeeze(mesh.z_grid[0,0,:])>30_000)
    x_array = mesh.x_grid[:,:,0]
    y_array = mesh.y_grid[:,:,0]
    z_array = mesh.z_grid[:,:,surface_index:z_domain_limit]

    d2_grid_shape = x_array.shape
    d3_grid_shape = z_array.shape

    iterable_indices = create_worker_patches(worker_div_multiplier, worker_count, d2_grid_shape, x_array, y_array)
    
    iterable_arguments = []
    for indices in iterable_indices:
        iterable_arguments.append((indices,
                GrayverInterpolatorFactory(projection, file=conductance_file),
                ElevationSplineFactory(gebco_data, projection)))
            
    print('creating shared variables')
    x_buffer, y_buffer, z_buffer = return_memory_views_of_arrays(x_array,y_array,z_array)

    init_args = (x_buffer, y_buffer, z_buffer, d2_grid_shape, d3_grid_shape, output_dirs, debug_mode, compute_mode)

    if debug_mode:
        init_worker(*init_args)
        compute_code_grid(iterable_arguments[0])
        return None
    else:
        print(f'launching {worker_count} workers')
        with mp.Pool(worker_count, initializer=init_worker, initargs=init_args) as p:
            list(tqdm.tqdm(p.imap(compute_code_grid,iterable_arguments),
                       total=len(iterable_arguments),
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',smoothing=0))



def calculate_conductivity(mesh_file, gebco_dir, conductance_file, working_dir, 
                            debug_mode=False,worker_div_multiplier=1,worker_count = 46,compute_mode='default'):
    """
    performs conductivity calculations using the provided files

    Parameters
    ==========

    mesh_file : str
        .npy file of the mesh to perform calculations on. This assumes the mesh file x direction
    is oriented in geographic coordinates (x is E-W, y is N-S)

    gebco_dir : str
        the directory containing the tiles to use for bathymetry data retrieval

    conductance_file : str
        the file containing world conductance data

    debug_mode : bool [optional] 
        whether to run the calculation in debug mode. In debug mode, the calculation is single threaded. 
    Debug mode also results in relative run times being calculated and printed to terminal for every major
    preprocessing operation.

    Returns
    =======
    None
    
    
    """
    #first arg should be target mesh file
    # second arg should be target
    mesh_name = mesh_file.split(os.sep)[-1].split('.')[0]
    directory_strings = {
        'mesh_directory':mesh_file,
        'elevation_directory':gebco_dir,
        'conductance_file':conductance_file,
        'mesh_name':mesh_name,
        'output_dirs':{}
    }

    output_folder=f'{working_dir}{os.sep}{mesh_name}_conductivity'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for extension in subfolder_extensions:
        new_path = f'{output_folder}{os.sep}{extension}'
        directory_strings['output_dirs'][extension]=new_path
        if not os.path.exists(new_path):
            os.makedirs(new_path)

    prepare_code_grid(directory_strings,debug_mode=debug_mode,
                    worker_div_multiplier=worker_div_multiplier, worker_count=worker_count,compute_mode=compute_mode)

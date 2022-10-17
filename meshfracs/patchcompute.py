import numpy as np
from . import aspect
from .datainterp import Regridder
from . import alphadecay as adecay
from . import volumefractions
import time

def compute_patches(sed_thickness,
                      sed_conductivity,
                      ocean_conductivity,
                      topography,
                      x_grid, y_grid, z_grid, debug):
    t0=0.0
    timing_dict = {
            'correct patch aspect':0,
            'new coordinates':0,
            'distance weights':0,
            'new topography':0,
            'sediment thickness':0,
            'ocean conductivity':0,
            'sediment conductivity':0,
            'column scoring':0
        }
    if debug:
        timing_dict = {
            'correct patch aspect':0,
            'new coordinates':0,
            'distance weights':0,
            'new topography':0,
            'sediment thickness':0,
            'ocean conductivity':0,
            'sediment conductivity':0,
            'column scoring':0
        }

    
    regridder = Regridder()
    shape2d   = x_grid.shape
    shape3d   = z_grid.shape

    solid_frac_xyz    = np.ones(shape3d)
    sediment_frac_xyz = np.zeros(shape3d) 

    ocean_conductivity_xy    = np.zeros(shape2d)
    sediment_conductivity_xy = np.zeros(shape2d)
    if debug:
        t0 = time.perf_counter()
    total_iterations = shape2d[0]*shape2d[1]
    for ix in range(shape2d[0]-1):
        for iy in range(shape2d[1]-1):
            
            x_cell = x_grid[ix:ix+2,iy:iy+2]
            y_cell = y_grid[ix:ix+2,iy:iy+2]
            z_column = -z_grid[ix:ix+2,iy:iy+2,:]

            corrected_aeqd_patch_x, corrected_aeqd_patch_y = aspect.correct_patch_aspect(x_cell,
                                                                                         y_cell)
                                            
            if debug:
                dt = time.perf_counter()-t0
                timing_dict['correct patch aspect']+=dt
                t0 = time.perf_counter()
            new_x_surface, new_y_surface = regridder.get_new_coordinates(corrected_aeqd_patch_x,
                                                                         corrected_aeqd_patch_y)


            square_x = np.linspace(0,1,num=new_x_surface.shape[0])
            square_y = np.linspace(0,1,num=new_x_surface.shape[1])
            square_xx, square_yy = np.meshgrid(square_x, square_y,indexing='ij')
            
            if debug:
                dt = time.perf_counter()-t0
                timing_dict['new coordinates']+=dt
                t0 = time.perf_counter()                   

            alpha_sediment      = adecay.get_l1norm_alpha(new_x_surface,new_y_surface,
                                                x0=-9e6,x1=-5e6,x2=5e6,x3=9e6,
                                                y0=-6e6,y1=-3e6,y2=3e6,y3=6e6)

            alpha_ocean         = adecay.get_l1norm_alpha(new_x_surface,new_y_surface,
                                                x0=-5e6,x1=-3e6,x2=3e6,x3=5e6,
                                                y0=-6e6,y1=-3e6,y2=3e6,y3=6e6)

            if debug:
                dt = time.perf_counter()-t0
                timing_dict['distance weights']+=dt
                t0 = time.perf_counter()

            topo_patch = topography.get_values(new_x_surface,new_y_surface)
            topo_patch = adecay.decay_data_patch(alpha_sediment,topo_patch,average_value=-5e3)
            if debug:
                dt = time.perf_counter()-t0
                timing_dict['new topography']+=dt
                t0 = time.perf_counter()

            sed_topo   = sed_thickness.get_values(new_x_surface,new_y_surface)
            sed_topo   = adecay.decay_data_patch(alpha_sediment,sed_topo,average_value=0)
            if debug:
                dt = time.perf_counter()-t0
                timing_dict['sediment thickness']+=dt
                t0 = time.perf_counter()

            sed_cond   = sed_conductivity.get_values(new_x_surface,new_y_surface)
            sed_cond   = adecay.decay_data_patch(alpha_ocean,sed_cond,average_value=sed_conductivity.average)
            if debug:
                dt = time.perf_counter()-t0
                timing_dict['sediment conductivity']+=dt
                t0 = time.perf_counter()

            ocean_patch = ocean_conductivity.get_values(new_x_surface,new_y_surface)
            ocean_patch = adecay.decay_data_patch(alpha_ocean,ocean_patch,average_value=ocean_conductivity.average)
            if debug:
                dt = time.perf_counter()-t0
                timing_dict['ocean conductivity']+=dt
                t0 = time.perf_counter()

            solid_frac_column, sediment_frac_column = volumefractions.get_score_for_column(z_column, 
                                                             square_xx, square_yy, topo_patch, sed_topo)
            if debug:
                dt = time.perf_counter()-t0
                timing_dict['column scoring']+=dt
                t0 = time.perf_counter()                    
            ocean_conductivity_xy[ix,iy]   = np.mean(ocean_patch)
            sediment_conductivity_xy[ix,iy]= np.mean(sed_cond)

            z_depth = len(solid_frac_column)
            solid_frac_xyz[ix,iy,:z_depth]    = solid_frac_column
            sediment_frac_xyz[ix,iy,:z_depth] = sediment_frac_column

    if debug:
        print('total time (s) per operation\n')
        for key, value in timing_dict.items():
            print(f"\t {key}:{value}")
        print("\n")
        print('time per inner loop iteration (s)\n')
        for key, value in timing_dict.items():
            print(f"\t {key}:{value/total_iterations}")

        print("\n")

    return solid_frac_xyz, sediment_frac_xyz, ocean_conductivity_xy, sediment_conductivity_xy

def compute_aspect_correction_x(x_grid, y_grid):

    shape2d   = x_grid.shape
    x_max_value    = np.ones(shape2d)
    x_min_value    = np.ones(shape2d)

    for ix in range(shape2d[0]-1):
        for iy in range(shape2d[1]-1):
            
            x_cell = x_grid[ix:ix+2,iy:iy+2]
            y_cell = y_grid[ix:ix+2,iy:iy+2]

            corrected_aeqd_patch_x, corrected_aeqd_patch_y = aspect.correct_patch_aspect(x_cell,
                                                                                         y_cell)
            x_max_value[ix,iy]=np.amax(corrected_aeqd_patch_x) 
            x_min_value[ix,iy]=np.amin(corrected_aeqd_patch_x)  
            

    return x_max_value, x_min_value

def compute_aspect_correction_y(x_grid, y_grid):

    shape2d   = x_grid.shape
    y_max_value    = np.ones(shape2d)
    y_min_value    = np.ones(shape2d)

    for ix in range(shape2d[0]-1):
        for iy in range(shape2d[1]-1):
            
            x_cell = x_grid[ix:ix+2,iy:iy+2]
            y_cell = y_grid[ix:ix+2,iy:iy+2]

            corrected_aeqd_patch_x, corrected_aeqd_patch_y = aspect.correct_patch_aspect(x_cell,
                                                                                         y_cell)
            y_max_value[ix,iy]=np.amax(corrected_aeqd_patch_y) 
            y_min_value[ix,iy]=np.amin(corrected_aeqd_patch_y)  
            

    return y_max_value, y_min_value


def compute_ratios(x_grid, y_grid):

    shape2d   = x_grid.shape
    ratios    = np.zeros(shape2d)

    for ix in range(shape2d[0]-1):
        for iy in range(shape2d[1]-1):
            
            x_cell = x_grid[ix:ix+2,iy:iy+2]
            y_cell = y_grid[ix:ix+2,iy:iy+2]

            ratio = aspect.get_patch_aspect(x_cell,y_cell)
            ratios[ix,iy]=ratio
                                        
    return ratios
from datainterp import DataInterpolator
import sys
import os
sys.path.append(os.path.join('/home','kmendoza','Desktop','Fall 2021','egi_utils','egi_utils'))
from egi_utils import elevation_data
from numba import jit
from scipy.interpolate import griddata, RectBivariateSpline
import numpy as np
import geopandas as gpd
import os
import h5py

class GrayverData:

    def __init__(self,proj_converter,path=None):
        self.proj_converter = proj_converter
        if path is None:
            file = os.path.join('global conductance model','agrayver','data','conductance_world.h5')
        else:
            file = os.path.join(path,'global conductance model','agrayver','data','conductance_world.h5')
        with h5py.File(file, 'r') as f:
            lat = f['lat'][()]
            lon = f['lon'][()]
            thickness = f['sed_thickness'][()]
            sed_s     = f['sediments'][()]
            ocean_c   = f['average'][()]

        thickness[np.isnan(thickness)] = 0
        thickness[thickness<0]=0
        sed_c= sed_s/thickness
        sed_c[np.isnan(sed_c)]      = np.nanmean(sed_c[sed_c!=0])
        sed_c[sed_c==0]             = np.nanmean(sed_c[sed_c!=0])

        ocean_c[np.isnan(ocean_c)]  = np.nanmean(ocean_c)
        ocean_c[ocean_c==0]         = np.nanmean(ocean_c)

        if np.any(np.isnan(thickness)):
            print(f'nan detected in sediment thickness')
        if np.any(np.isnan(sed_c)):
            print(f'nan detected in sediment conductivity')
        if np.any(np.isnan(ocean_c)):
            print(f'nan detected in ocean conductivity')
            # lat is x, lon is y, then data
        self._ocean_conductivity    = ocean_c
        self._sediment_conductivity = sed_c
        self._sediment_thickness    = thickness

        aeqd_x, aeqd_y = proj_converter(lon, lat)
        self._aeqd_x = aeqd_x
        self._aeqd_y = aeqd_y


    def get_interpolator(self, type, discretization,
                         grid_divisioning=3_000.0,
                         x_points=None,
                         y_points=None):

        x_min = np.amin(x_points)
        x_max = np.amax(x_points)
        x_domain = (x_min,x_max)

        y_min = np.amin(y_points)
        y_max = np.amax(y_points)
        y_domain = (y_min,y_max)

        if type=='sediment thickness':
            data = DataInterpolator(x_points,y_points,self._sediment_thickness.ravel(),
                                    grid_divisioning=100.0, x_domain=x_domain, y_domain=y_domain,
                                    nonzero=grid_divisioning)
        elif type=='sediment conductivity':
            data = DataInterpolator(x_points,y_points,self._sediment_conductivity.ravel(),
                                    grid_divisioning=100.0, x_domain=x_domain, y_domain=y_domain,
                                    nonzero=grid_divisioning)
        elif type=='ocean conductivity':
            data = DataInterpolator(x_points,y_points,self._ocean_conductivity.ravel(),
                                    grid_divisioning=100.0, x_domain=x_domain, y_domain=y_domain,
                                    nonzero=grid_divisioning)
        else:
            return None
        return data




class ElevationPatch:

    def __init__(self, tilelocation, projection):
        netcdf_datasets       = elevation_data.create_netcdf_datasets(projection,tilelocation)
        self.dataset          = elevation_data.TiledDataset(projection,netcdf_datasets,
                                                        filepath=f'GEBCO_cache/cache.csv')
                                                        
        self.dataset.should_multiprocess=False

    def get_elevation_interpolator(self,x_grid, y_grid):
        x = np.arange(np.amin(x_grid),np.amax(x_grid)+500,500)
        y = np.arange(np.amin(y_grid),np.amax(y_grid)+500,500)
        xx, yy = np.meshgrid(x,y)
        patch  = self.dataset._return_elevation_grid(xx.ravel(),yy.ravel())

        data =  RectBivariateSpline(x,y,patch.ravel(),kx=1,ky=1)
        return data

class Coastlines:

    def __init__(self, shapefile, projection):
        self.shapefile=shapefile
        self.shapefile.to_crs(crs=projection)


    def identify_land_points(self, x_grid, y_grid):
        gdf             = gpd.GeoDataFrame(geometry = gpd.points_from_xy(x_grid.ravel(),
                                                                   y_grid.ravel()))
        gdf['na_point'] = False
        points_within   = gpd.sjoin(gdf, self.shapefile, op='within')
        gdf.loc[points_within.index,'na_point']=True
        mask   = gdf.na_point.values.reshape(x_grid.shape)
        points = np.zeros(x_grid.shape)
        points[mask]=1
        return points

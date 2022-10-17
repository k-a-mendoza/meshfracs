import sys
import os
from egi_utils import elevation_data
from numba import jit
from scipy.interpolate import griddata, RectBivariateSpline
from pyproj import transform, Proj
import numpy as np
import geopandas as gpd
from .datainterp import DataInterpolator, RectBivariateFacade
import h5py
wgs_84 = Proj('epsg:4326')


class GrayverInterpolatorFactory:

    def __init__(self,projection,file=None):
        self._preprocessing_done=False
        self.projection = projection
        self.file = file

    def _begin_preprocessing(self):
        with h5py.File(self.file, 'r') as f:
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
        lon_grid, lat_grid = np.meshgrid(lon,lat)
        aeqd_x, aeqd_y = transform(wgs_84, self.projection, lat_grid,lon_grid)
        self._aeqd_x = aeqd_x
        self._aeqd_y = aeqd_y
        self._preprocessing_done = True

    def get_interpolator(self, type, grid_divisioning=5000.0, x_points=None,y_points=None):
        """
        Parameters
        ==========
        type : str
            either 'sediment thickness', 'sediment conductivity', or 'ocean conductivity'

        grid_divisioning : float
            the dxy to make the intermediate interpolated grid. in meters

        x_points : np.ndarray
            array of x domain points spanning the expected interpolator array

        y_points : np.ndarray
            array of y domain points spanning the expected interpolator array 
        
        
        """
        if not self._preprocessing_done:
            self._begin_preprocessing()
        x_domain, y_domain = self.define_domains(x_points, y_points)

        domain_mask = (x_domain[0] < self._aeqd_x) & (x_domain[1] > self._aeqd_x) &\
                      (y_domain[0] < self._aeqd_y) & (y_domain[1] > self._aeqd_y) 

        aeqd_x = self._aeqd_x[domain_mask].ravel()
        aeqd_y = self._aeqd_y[domain_mask].ravel()

        if type=='sediment thickness':
            data = self._sediment_thickness
        elif type=='sediment conductivity':
            data = self._sediment_conductivity
        elif type=='ocean conductivity':
            data = self._ocean_conductivity
        else:
            return None
        data = data[domain_mask].ravel()
        interpolator = DataInterpolator(aeqd_x, aeqd_y, data,
                                    x_domain=x_domain, y_domain=y_domain,
                                    grid_divisioning=grid_divisioning)
        return interpolator

    def define_domains(self, x_points, y_points):
        domain_expansion_x = 0.1
        domain_expansion_y = 0.1
        x_min = np.amin(x_points)
        x_max = np.amax(x_points)
        x_delta  = x_max - np.mean(x_points)
        y_min = np.amin(y_points)
        y_max = np.amax(y_points)
        y_delta  = y_max - np.mean(y_points)

        if x_delta < 50_000:
            x_delta=50_000
            domain_expansion_x=0.5

        if y_delta < 50_000:
            y_delta=50_000
            domain_expansion_y=0.5

        x_domain = (x_min-domain_expansion_x*x_delta,x_max+domain_expansion_x*x_delta)
        y_domain = (y_min-domain_expansion_y*y_delta,y_max+domain_expansion_y*y_delta)
        return x_domain,y_domain

class ElevationSplineFactory:

    def __init__(self, tilelocation, projection):
        self.tile_location = tilelocation
        self.projection = projection
        self._preprocessing_done = False
        
    def _begin_preprocessing(self):
        netcdf_datasets       = elevation_data.create_netcdf_datasets(self.projection,self.tile_location)
        self.dataset          = elevation_data.TiledDataset(self.projection,netcdf_datasets,
                                                        filepath=f'GEBCO_cache/cache.csv')
                                                        
        self.dataset.should_multiprocess=False
        self._preprocessing_done = True

    def get_interpolator(self,x_domain, y_domain):
        if not self._preprocessing_done:
            self._begin_preprocessing()
        x = np.arange(np.amin(x_domain),np.amax(x_domain)+500,500)
        y = np.arange(np.amin(y_domain),np.amax(y_domain)+500,500)
        xx, yy = np.meshgrid(x,y,indexing='ij')
        patch  = self.dataset.get_elevation(xx,yy)
        spline =  RectBivariateSpline(x,y,patch,kx=1,ky=1)
        return RectBivariateFacade(spline)

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

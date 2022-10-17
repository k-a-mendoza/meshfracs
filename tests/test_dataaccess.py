
from meshfracs.dataaccess import ElevationSplineFactory, GrayverInterpolatorFactory
from pyproj import Proj
import os
import numpy as np
user_home = os.path.expanduser("~")

def get_crs():
    longitude_center = -113
    latitude_center  = 39.75
    longitude_center = -113
    latitude_center  = 39.75
    azimuth_equid    = Proj(proj='aeqd', datum='WGS84', lon_0=longitude_center, lat_0=latitude_center, units='m')
    return azimuth_equid

projection = get_crs()
grayver_data_path = f'{user_home}/abetadatabase/world_conductance/grayver2021/conductance_world.h5'

def test_grayverdata_original_data_are_numbers():
    sediment_data_access = GrayverInterpolatorFactory(projection,file=grayver_data_path)
    assert np.all(~np.isnan(sediment_data_access._ocean_conductivity))
    assert np.all(~np.isnan(sediment_data_access._sediment_conductivity))
    assert np.all(~np.isnan(sediment_data_access._sediment_thickness))
    assert np.all(~np.isnan(sediment_data_access._aeqd_x))
    assert np.all(~np.isnan(sediment_data_access._aeqd_y))

def test_grayverdata_original_data_are_not_infinite():
    sediment_data_access = GrayverInterpolatorFactory(projection,file=grayver_data_path)
    assert np.all(~np.isinf(np.abs(sediment_data_access._ocean_conductivity)))
    assert np.all(~np.isinf(np.abs(sediment_data_access._sediment_conductivity)))
    assert np.all(~np.isinf(np.abs(sediment_data_access._sediment_thickness)))
    assert np.all(~np.isinf(np.abs(sediment_data_access._aeqd_x)))
    assert np.all(~np.isinf(np.abs(sediment_data_access._aeqd_y)))

def test_grayverdata_sediment_intermediate_grid_values():
    x_values = np.linspace(-1.2e6,-0.9e6,num=100)
    y_values = np.linspace(-100e3, 100e3,num=100)

    sediment_data_access = GrayverInterpolatorFactory(projection,file=grayver_data_path)
    interpolator = sediment_data_access.get_interpolator(type='sediment thickness',x_points=x_values,
                                                                                  y_points=y_values)

    assert np.all(~np.isinf(np.abs(interpolator.x_array)))
    assert np.all(~np.isnan(np.abs(interpolator.x_array)))
    assert np.all(~np.isinf(np.abs(interpolator.y_array)))
    assert np.all(~np.isnan(np.abs(interpolator.y_array)))
    assert np.all(~np.isinf(np.abs(interpolator.x_grid)))
    assert np.all(~np.isnan(np.abs(interpolator.x_grid)))
    assert np.all(~np.isinf(np.abs(interpolator.y_grid)))
    assert np.all(~np.isnan(np.abs(interpolator.y_grid)))
    assert np.all(~np.isinf(np.abs(interpolator.z_grid)))
    assert np.all(~np.isnan(np.abs(interpolator.z_grid)))


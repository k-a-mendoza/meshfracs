import numpy as np
from numba import jit, njit
from scipy.interpolate import griddata, RectBivariateSpline




class DataInterpolator:
    n_points_max = 1.5e6
    def __init__(self, x_source_points, y_source_points, z_points, grid_divisioning=100.0,
                       x_domain=None,y_domain=None):
        self.grid_dxy= grid_divisioning
        if x_domain is None:
            x_min = np.amin(x_source_points)
            x_max = np.amax(x_source_points)
            x_domain = (x_min, x_max)

        if y_domain is None:
            y_min = np.amin(y_source_points)
            y_max = np.amax(y_source_points)
            y_domain = (y_min,y_max)

        grid_divisioning = self.set_grid_divisioning(x_domain, y_domain)

        x_array, y_array = self.create_arrays(x_domain, y_domain)

        x_grid, y_grid = np.meshgrid(x_array, y_array,indexing='ij')

        data_coordinates = np.stack([x_source_points,y_source_points],axis=1)

        grid_zdata_l       = griddata(data_coordinates, z_points.ravel(),
                             (x_grid.ravel(), y_grid.ravel()), method='linear')
        grid_zdata_n       = griddata(data_coordinates, z_points.ravel(),
                             (x_grid.ravel(), y_grid.ravel()), method='nearest')

        nanmask = np.isnan(grid_zdata_l)
        grid_zdata_l[nanmask]=grid_zdata_n[nanmask]

        self.x_array = x_array
        self.y_array = y_array
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.z_grid = grid_zdata_l.reshape(x_grid.shape)
        self.average = np.mean(self.z_grid)
        self.ev = RectBivariateSpline(self.x_array, self.y_array, self.z_grid, kx=1, ky=1)

    def set_grid_divisioning(self, x_domain, y_domain):
        x_array, y_array = self.create_arrays(x_domain, y_domain)
        n_points = len(x_array)*len(y_array)
        while n_points > self.n_points_max:
            self.grid_dxy*=1.2
            x_array, y_array = self.create_arrays(x_domain, y_domain)
            n_points = len(x_array)*len(y_array)

    def create_arrays(self, x_domain, y_domain):
        points_x = np.arange(x_domain[0],x_domain[1],self.grid_dxy)
        points_y = np.arange(y_domain[0],y_domain[1],self.grid_dxy)
        return points_x, points_y

    def get_values(self, x_grid, y_grid):
        return self.ev.ev(x_grid,y_grid)

    

@njit(fastmath=True)
def interpolate_new_surface(grid_z, xi, yi):
    """
    
    Parameters
    ==========

    grid_x : np.ndarray

    grid_y : np.ndarray

    grid_z : np.ndarray
    
    """
    a0 = grid_z[0,0]
    a1 = grid_z[1,0] - a0
    a2 = grid_z[0,1] - a0
    a3 = grid_z[1,1] - grid_z[1,0] - grid_z[0,1] + a0
    coefficients = np.asarray([a0,a1,a2,a3])

    xyi = xi.ravel()*yi.ravel()
    A = np.zeros((len(xyi),4))
    A[:,0]=1
    A[:,1]=xi.ravel()
    A[:,2]=yi.ravel()
    A[:,3]=xyi
    grid_points = A.dot(coefficients).reshape(xi.shape)
    return grid_points


class RectBivariateFacade:

    def __init__(self,rectspline):
        self.rectspline=rectspline

    def get_values(self, x_grid, y_grid):
        return self.rectspline.ev(x_grid,y_grid)

class Regridder:

    def __init__(self,lower_limit=10, upper_limit= 150, nominal_division=500.0):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.nominal_division = nominal_division
        unit_array = np.linspace(0,1,num=2)
        square_xx, square_yy = np.meshgrid(unit_array,unit_array,indexing='ij')

        self.data2d = np.zeros((4,2))
        self.data2d[:,0]=square_xx.ravel()
        self.data2d[:,1]=square_yy.ravel()

        cube_xx, cube_yy, cube_zz = np.meshgrid(unit_array,unit_array,unit_array,indexing='ij')
        self.data3d = np.zeros((8,3))
        self.data3d[:,0]=cube_xx.ravel()
        self.data3d[:,1]=cube_yy.ravel()
        self.data3d[:,2]=cube_zz.ravel()


    def get_new_coordinates(self, grid_x, grid_y):

        deltax0 = grid_x[1,0] - grid_x[0,0]
        deltax1 = grid_x[1,1] - grid_x[0,1]
        deltay0 = grid_y[0,1] - grid_y[0,0]
        deltay1 = grid_y[1,1] - grid_y[1,0]

        max_x_delta = np.amax([deltax0, deltax1])
        max_y_delta = np.amax([deltay0, deltay1])
        min_delta   = np.amin([max_x_delta, max_y_delta])
        divisions   = int(min_delta//self.nominal_division)

        if divisions < self.lower_limit:
            divisions = self.lower_limit
        elif divisions > self.upper_limit:
            divisions = self.upper_limit

        square_array         = np.linspace(0,1,num=divisions)
        square_xx, square_yy = np.meshgrid(square_array,square_array,indexing='ij')

        super_x_grid = interpolate_new_surface(grid_x, square_xx, square_yy)
        super_y_grid = interpolate_new_surface(grid_y, square_xx, square_yy)

        return super_x_grid, super_y_grid

    

import numpy as np
from numba import jit, njit
from scipy.interpolate import griddata, RectBivariateSpline

class DataInterpolator:

    def __init__(self, x_points, y_points, z_points, grid_divisioning=100.0,
                       x_domain=None,y_domain=None):
        self.grid_dxy= grid_divisioning
        if x_domain is None:
            x_min = np.amin(x_points)
            x_max = np.amax(x_points)
            x_domain = (x_min, x_max)

        if y_domain is None:
            y_min = np.amin(y_points)
            y_max = np.amax(y_points)
            y_domain = (y_min,y_max)

        x_array = np.arange(x_domain[0],x_domain[1]+grid_divisioning,grid_divisioning)
        y_array = np.arange(y_domain[0],y_domain[1]+grid_divisioning,grid_divisioning)
        x_grid,y_grid = np.meshgrid(x_array, y_array)
        data_coordinates = np.concatenate([x_points,y_points])
        grid_zdata = griddata(data_coordinates, z_points.ravel(),
                             (x_grid.ravel(), y_grid.ravel()), method='linear')
        self.rectspline=False
        self.x_array = x_array
        self.y_array = y_array
        self.x_grid = x_array
        self.y_grid = y_array
        self.z_grid = grid_zdata
        self.average = np.mean(self.z_grid)


    @njit
    def get_values(self, x_grid, y_grid):
        ix_position = np.zeros()

        x_array = self.x_array
        y_array = self.y_array
        z_grid  = self.z_grid

        ravel_x = x_grid.ravel()
        ravel_y = y_grid.ravel()

        z_data      = np.zeros(x_grid.shape)
        ix_position = np.zeros(len(ravel_x))
        iy_position = np.zeros(len(ravel_y))

        for i in range(len(ravel_x)):
            x = ravel_x[i]
            ix_position[i]=np.argmin(abs(x-x_array))

        for i in range(len(ravel_y)):
            y = ravel_y[i]
            iy_position[i]=np.argmin(abs(y-y_array))

        for ix, iy in zip(ix_position,iy_position):
            z_data[ix,iy]=z_grid[ix,iy]
        return z_data

    @jit
    def ev(self, x_grid, y_grid):
        if not self.rectspline:
            self.rectspline = RectBivariateSpline(self.x_array,self.y_array,
                                                self.z_grid,kx=1,ky=1)
        return self.rectspline.ev(x_grid,y_grid)


class Regridder:


    def __init__(self,lower_limit=2, upper_limit= 30, nominal_division=500.0):
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

    @jit(nopython=True)
    def _interpolate(self,grid,xi,yi):
        a00 = grid[0,0]
        a01 = grid[0,1] - a00
        a10 = grid[1,0] - a00
        a11 = grid[1,1] - grid[1,0] - grid[0,1] + a00

        result = a00 + a10*xi + a01*yi + a11*xi*yi

        return result


    @jit(nopython=True)
    def get_new_coordinates(self, grid_x, grid_y):

        deltax0 = grid_x[1,0] - grid_x[0,0]
        deltax1 = grid_x[1,1] - grid_x[0,1]
        deltay0 = grid_y[0,1] - grid_y[0,0]
        deltay1 = grid_y[1,1] - grid_y[1,0]

        max_x_delta = np.amax([deltax0, deltax1])
        max_y_delta = np.amax([deltay0, deltay1])
        min_delta   = np.amin([max_x_delta, max_y_delta])
        divisions   = min_delta//self.nominal_division

        if divisions < self.lower_limit:
            divisions = self.lower_limit
        elif divisions > self.upper_limit:
            divisions = self.upper_limit

        square_array         = np.linspace(0,1,num=divisions)
        square_xx, square_yy = np.meshgrid(square_array,square_array,indexing='ij')

        super_x_grid = self._interpolate(grid_x,square_xx,square_yy)
        super_y_grid = self._interpolate(grid_y,square_xx,square_yy)

        return super_x_grid, super_y_grid


    @njit
    def get_3d_cube(self, z_grid, iz, x_grid, y_grid):
        N_VERT     = 9
        total_dz   = z_grid[:,:,iz]-z_grid[:,:,iz+1]
        dz_vert    = total_dz/(N_VERT-1)
        shape2d    = x_grid.shape
        shape3d    = (*shape2d,N_VERT)
        z_grid_new = np.zeros(shape3d)

        for cube_index in range(N_VERT):
            z_offset = cube_index*dz_vert
            z_grid_new[:,:,iz] = self._interpolate(z_grid[:,:,iz]-z_offset,x_grid,y_grid)
        return z_grid_new

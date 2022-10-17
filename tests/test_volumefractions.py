
from meshfracs.volumefractions import get_score_for_column
import numpy as np
import time

def test_volumefraction_returns_proper_length_arrays():
    x_array = np.linspace(0,   1, num=2)
    z_array = np.linspace(1, -10, num=12)
    x_cell, y_cell, z_cell = np.meshgrid(x_array, x_array, z_array, indexing='ij')
    xx_array = np.linspace(0,1,num=5)
    xx_array, yy_array = np.meshgrid(xx_array,xx_array,indexing='ij')

    topo_patch = np.ones(xx_array.shape)*3
    sed_topo   = np.zeros(xx_array.shape)
    result = get_score_for_column(z_cell,xx_array, yy_array, topo_patch, sed_topo)
    assert len(result[0])==len(z_array)
    assert len(result[1])==len(z_array)

def test_volumefraction_returns_default_arrays():
    x_array = np.linspace(0,   1, num=2)
    z_array = np.linspace(1, -10, num=12)
    x_cell, y_cell, z_cell = np.meshgrid(x_array, x_array, z_array, indexing='ij')
    x_cell = x_cell[:,:,0]
    y_cell = y_cell[:,:,0]
    xx_array = np.linspace(0,1,num=5)
    xx_array, yy_array = np.meshgrid(xx_array,xx_array,indexing='ij')

    topo_patch = np.ones(xx_array.shape)*3
    sed_topo   = np.zeros(xx_array.shape)
    result = get_score_for_column(z_cell,xx_array, yy_array, topo_patch, sed_topo)
    assert np.all(result[0]==1)
    assert np.all(result[1]==0)

def test_thick_sediment_above_sealevel_topo():
    x_array = np.linspace(0,   1, num=2)
    z_array = np.linspace(1, -10, num=12)
    x_cell, y_cell, z_cell = np.meshgrid(x_array, x_array, z_array, indexing='ij')
    xx_array = np.linspace(0,1,num=5)
    xx_array, yy_array = np.meshgrid(xx_array,xx_array,indexing='ij')

    topo_patch = np.ones(xx_array.shape)*3
    sed_topo   = np.ones(xx_array.shape)*12
    x_cell = x_cell[:,:,0]
    y_cell = y_cell[:,:,0]
    result = get_score_for_column(z_cell, xx_array, yy_array, topo_patch, sed_topo)
    assert np.all(result[0]==1)
    assert np.sum(result[1]==1)

def test_zero_sediment_below_sealevel_topo():
    x_array = np.linspace(0,   1, num=2)
    z_array = np.linspace(1, -10, num=12)
    x_cell, y_cell, z_cell = np.meshgrid(x_array, x_array, z_array, indexing='ij')
    xx_array = np.linspace(0,1,num=5)
    xx_array, yy_array = np.meshgrid(xx_array,xx_array,indexing='ij')

    topo_patch = np.ones(xx_array.shape)*-1
    sed_topo   = np.ones(xx_array.shape)*0
    x_cell = x_cell[:,:,0]
    y_cell = y_cell[:,:,0]
    result = get_score_for_column(z_cell, xx_array, yy_array, topo_patch, sed_topo)
    assert np.sum(result[0]==1)
    assert np.all(result[1]==0)

def test_volumefrac_speed():
    x_array = np.linspace(0,   1, num=2)
    z_array = np.linspace(1, -10, num=12)
    x_cell, y_cell, z_cell = np.meshgrid(x_array, x_array, z_array, indexing='ij')
    xx_array = np.linspace(0,1,num=5)
    xx_array, yy_array = np.meshgrid(xx_array,xx_array,indexing='ij')

    topo_patch = np.ones(xx_array.shape)*-1
    sed_topo   = np.ones(xx_array.shape)*0
    start = time.perf_counter()
    for i in range(1000):
        result = get_score_for_column(z_cell, xx_array, yy_array, topo_patch, sed_topo)
    stop = time.perf_counter()
    print(f'aspect is ok averaged {1e3*(stop-start)/1000}  ms/it')
    assert True


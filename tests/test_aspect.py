import sys
import os
sys.path.append(os.path.join('/home','kmendoza','Western_US_Mesh_Dev'))
sys.path.append(os.path.join('/home','kmendoza','Desktop','EGI UTILITIES','egi_utils','egi_utils'))
sys.path.append(os.path.join('/home','kmendoza','Desktop','EGI UTILITIES'))
import time
import numpy as np
from meshfracs import aspect
_patch_is_rectangular          = aspect.patch_is_rectangular
_aspect_ratio_exceeds_criteria = aspect.aspect_ratio_exceeds_criteria
_get_max_coordinate_diff       = aspect.get_max_coordinate_diff
_correct_square_aspect_ratio   = aspect.correct_square_aspect_ratio
_quadrilateral_aspect_ratio_correction = aspect.correct_quadrilateral_aspect_ratio


def _get_square_grids():
    x = np.linspace(0,1,num=2)
    square_x, square_y = np.meshgrid(x,x,indexing='ij')
    return square_x, square_y

def test_issquare():
    square_x, square_y =  _get_square_grids()
    assert _patch_is_rectangular(square_x, square_y), 'failed square check'

def test_issquarespeed():
    square_x, square_y =  _get_square_grids()
    start = time.perf_counter()
    for i in range(1000):
         _patch_is_rectangular(square_x, square_y)
    stop = time.perf_counter()
    print(f'issquare speed averaged {1e6*(stop-start)/1000}  (mu s)/it')
    assert True

def test_isrectangulary():
    square_x, square_y =  _get_square_grids()
    assert _patch_is_rectangular(square_x, square_y), 'failed y rect check'


def test_isrectangularx():
    square_x, square_y =  _get_square_grids()
    assert _patch_is_rectangular(square_x, square_y), 'failed x rect check'

def test_notrectangular():
    square_x, square_y =  _get_square_grids()
    square_x[0,0]= square_x[0,0]-0.1
    assert not _patch_is_rectangular(square_x, square_y), 'failed not a rect check'

def test_ratio_is_ok():
    square_x, square_y =  _get_square_grids()
    xdiff = _get_max_coordinate_diff(square_x)
    ydiff = _get_max_coordinate_diff(square_y)  
    assert _aspect_ratio_exceeds_criteria(xdiff,ydiff)==0, 'failed nominal check'

def test_ratio_is_ok_speed():
    square_x, square_y =  _get_square_grids()
    xdiff = _get_max_coordinate_diff(square_x)
    ydiff = _get_max_coordinate_diff(square_y) 
    start = time.perf_counter()
    for i in range(1000):
        _aspect_ratio_exceeds_criteria(xdiff,ydiff)
    stop = time.perf_counter()
    print(f'aspect is ok averaged {1e6*(stop-start)/1000}  (mu s)/it')
    assert True

def test_ratio_x_fail():
    square_x, square_y =  _get_square_grids()
    xdiff = _get_max_coordinate_diff(square_x*5)
    ydiff = _get_max_coordinate_diff(square_y)
    assert _aspect_ratio_exceeds_criteria(xdiff,ydiff)==2, 'failed aspect exceeds xdiff check'

def test_ratio_y_fail():
    square_x, square_y =  _get_square_grids()
    xdiff = _get_max_coordinate_diff(square_x)
    ydiff = _get_max_coordinate_diff(square_y*5)
    assert _aspect_ratio_exceeds_criteria(xdiff,ydiff)==1, 'failed aspect exceeds ydiff check'

def test_square_aspect_correct_x():
    square_x, square_y =  _get_square_grids()
    new_x, new_y = _correct_square_aspect_ratio(square_x*5,square_y)
    xdiff = _get_max_coordinate_diff(new_x)
    ydiff = _get_max_coordinate_diff(new_y)
    assert xdiff/ydiff==4.0, 'failed xdiff is too large check'

def test_square_aspect_correct_x():
    square_x, square_y =  _get_square_grids()
    new_x, new_y = _correct_square_aspect_ratio(square_x,square_y*5)
    xdiff = _get_max_coordinate_diff(new_x)
    ydiff = _get_max_coordinate_diff(new_y)
    assert xdiff/ydiff==1/4, 'failed ydiff is too large check'

def test_square_aspect_correct_speed():
    square_x, square_y =  _get_square_grids()
    square_y*=5
    start = time.perf_counter()
    for i in range(1000):
        new_x, new_y = _correct_square_aspect_ratio(square_x,square_y)
    stop = time.perf_counter()
    print(f'square aspect ratio correction averaged {1e6*(stop-start)/1000}  (mu s)/it')
    assert True

def test_x_aspect_quad_correction():
    square_x, square_y =  _get_square_grids()
    square_x[1,0]+=0.1
    square_x*=9
    new_x, new_y = _quadrilateral_aspect_ratio_correction(square_x,square_y)
    alt_x, alt_y = _quadrilateral_aspect_ratio_correction(new_x,new_y)
    assert np.all(alt_x==new_x) & np.all(alt_y==new_y), 'irregular correction did not work for x direction'

def test_y_aspect_quad_correction():
    square_x, square_y =  _get_square_grids()
    square_y[0,1]+=0.1
    square_y*=9
    new_x, new_y = _quadrilateral_aspect_ratio_correction(square_x,square_y)
    alt_x, alt_y = _quadrilateral_aspect_ratio_correction(new_x,new_y)
    assert np.all(alt_x==new_x) & np.all(alt_y==new_y), 'irregular correction did not work for y direction'

def test_aspect_quad_correction_speed():
    square_x, square_y =  _get_square_grids()
    square_y[0,1]+=0.1
    square_x*=9
    start = time.perf_counter()
    for i in range(1000):
        new_x, new_y = _quadrilateral_aspect_ratio_correction(square_x,square_y)
    stop = time.perf_counter()
    print(f'quadrilateral aspect ratio correction averaged {1e6*(stop-start)/1000}  (mu s)/it')
    assert True, 'irregular correction did not work for y direction'
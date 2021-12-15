import sys
import os
sys.path.append(os.path.join('/home','kmendoza','Western_US_Mesh_Dev'))
sys.path.append(os.path.join('/home','kmendoza','Desktop','EGI UTILITIES','egi_utils','egi_utils'))
sys.path.append(os.path.join('/home','kmendoza','Desktop','EGI UTILITIES'))
import meshfracs
import timeit
import numpy as np
from meshfracs.aspect import _patch_is_rectangular, _aspect_ratio_exceeds_criteria, _get_max_coordinate_diff


def _get_square_grids():
    x = np.linspace(0,1,num=2)
    square_x, square_y = np.meshgrid(x,x,indexing='ij')
    return square_x, square_y

def test_issquare():
    square_x, square_y =  _get_square_grids()
    assert _patch_is_rectangular(square_x, square_y), 'failed square check'

def test_isrectangulary():
    square_x, square_y =  _get_square_grids()
    assert _patch_is_rectangular(square_x, square_y), 'failed y rect check'


def test_isrectangularx():
    square_x, square_y =  _get_square_grids()
    assert _patch_is_rectangular(square_x, square_y), 'failed x rect check'

def test_notrectangular():
    square_x, square_y =  _get_square_grids()
    square_x[0,0]= square_x[0,0]-0.1
    assert _patch_is_rectangular(square_x, square_y), 'failed not a rect check'

def test_ratio_is_ok():
    square_x, square_y =  _get_square_grids()
    xdiff = _get_max_coordinate_diff(square_x)
    ydiff = _get_max_coordinate_diff(square_y)  
    assert _aspect_ratio_exceeds_criteria(xdiff,ydiff)==0, 'failed nominal check'

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
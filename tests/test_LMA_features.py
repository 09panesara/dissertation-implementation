import pytest
from src.generate_LMA_vector import *

def test_calculate_angle():
    ''' Returns angle between ab and bc '''
    a = np.array([0,1,3])
    b= np.array([5,10,3])
    c = np.array([4,5,7])
    angle = calculate_angle(a,b,c)
    assert round(angle, 3) == 1.750

def test_dist_btwn_vectors():
    a = np.array([87, 107, 53])
    b = np.array([93, 102, 78])
    dist = dist_btwn_vectors(a, b)
    assert round(dist, 3) == 26.192

def test_dist_btwn_pt_and_axis():
    a = np.array([10,8,3])
    b= np.array([104,90,10])
    c = np.array([89,93,7])
    dist_to_axis = dist_btwn_pt_and_axis(a, b, c)
    assert round(dist_to_axis, 3) == 99.148







import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt

from src.grid_maker import make_grid
from cartesiangrid.operators import calc_div, calc_grad, zero_interpolator, tree_interpolator


def shock_direction(x_array, y_array, z_array, Tgrid, idx):
    """ Find shock direction according eq.(5) by Schaal14 in the point of coordinates indices idx"""
    gradT = calc_grad(x_array, y_array, z_array, Tgrid, idx, kind_info = 'idx')
    magnitude = np.linalg.norm(gradT)
    ds = - np.array(gradT) / magnitude 
    return ds
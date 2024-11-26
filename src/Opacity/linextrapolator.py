#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:12:06 2024

@author: konstantinos
"""

import numpy as np

def lin_extrapolator(y, V, slope_length = 2 ,extrarows = 3):
    # Low extrapolation
    yslope_low = y[slope_length - 1] - y[0] 
    y_extra_low = [y[0] - yslope_low * (i + 1) for i in range(extrarows)]
    # High extrapolation
    yslope_high = y[-1] - y[-slope_length]
    y_extra_high = [y[-1] + yslope_high * (i + 1) for i in range(extrarows)]
    
    # Stack, reverse low to stack properly
    yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
    
    # 2D low
    Vslope_low = V[slope_length - 1, :] - V[0, :] / yslope_low
    Vextra_low = [V[0, :] + Vslope_low * (y_extra_low[i] - y[0]) for i in range(extrarows)]

    # 2D high
    Vslope_high = V[-1, :] - V[-slope_length, :] / yslope_high
    Vextra_high = [V[-1, :] - Vslope_high * (y_extra_high[i] - y[-1]) for i in range(extrarows)]
    
    Vn = np.vstack([Vextra_low[::-1], V, Vextra_high]) 
    
    return yn, Vn

def extrapolator_flipper(x,y,V, slope_length = 5, extrarows = 60):
    xn, Vn = lin_extrapolator(x, V, slope_length, extrarows)
    yn, Vn = lin_extrapolator(y, Vn.T, slope_length, extrarows)
    return xn, yn, Vn.T
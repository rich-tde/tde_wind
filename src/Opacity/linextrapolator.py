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
    Vslope_low = (V[slope_length - 1, :] - V[0, :]) / yslope_low
    Vextra_low = [V[0, :] + Vslope_low * (y_extra_low[i] - y[0]) for i in range(extrarows)]

    # 2D high
    Vslope_high = (V[-1, :] - V[-slope_length, :]) / yslope_high
    Vextra_high = [V[-1, :] + Vslope_high * (y_extra_high[i] - y[-1]) for i in range(extrarows)]
    
    Vn = np.vstack([Vextra_low[::-1], V, Vextra_high]) 
    
    return yn, Vn

def line(x, m, c):
    return m * x + c
from scipy.optimize import curve_fit

def linfit_extrapolator(y, V, slope_length = 2 ,extrarows = 3):
    # Low extrapolation
    yslope_low = y[slope_length - 1] - y[0] 
    y_extra_low = [y[0] - yslope_low * (i + 1) for i in range(extrarows)]
    
    # High extrapolation
    yslope_high = y[-1] - y[-slope_length]
    y_extra_high = [y[-1] + yslope_high * (i + 1) for i in range(extrarows)]
    
    # Stack, reverse low to stack properly
    yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
    
    # 2D low
    Vextra_low = []
    for i in range(V.shape[1]):  # Loop over columns of V
        params_low, _ = curve_fit(line, y[0:slope_length], V[0:slope_length, i])
        Vslope_low, Vintercept_low = params_low
        Vextra_low.append([line(y_extra_low[j], Vslope_low, Vintercept_low) for j in range(extrarows)])

    # 2D high
    Vextra_high = []
    for i in range(V.shape[1]):  # Loop over columns of V
        params_high, _ = curve_fit(line, y[-slope_length:], V[-slope_length:, i])
        Vslope_high, Vintercept_high = params_high
        Vextra_high.append([line(y_extra_high[j], Vslope_high, Vintercept_high) for j in range(extrarows)])

    # Convert back to arrays
    Vextra_low = np.array(Vextra_low).T  # Transpose back to match dimensions
    Vextra_high = np.array(Vextra_high).T

    Vn = np.vstack([Vextra_low[::-1], V, Vextra_high]) 
    
    return yn, Vn

def extrapolator_flipper(x ,y, V, slope_length = 5, extrarows = 25):
    xn, Vn = lin_extrapolator(x, V, slope_length, extrarows)
    yn, Vn = lin_extrapolator(y, Vn.T, slope_length, extrarows)
    return xn, yn, Vn.T

def extrapolatorfit_flipper(x ,y, V, slope_length = 5, extrarows = 25):
    xn, Vn = linfit_extrapolator(x, V, slope_length, extrarows)
    yn, Vn = linfit_extrapolator(y, Vn.T, slope_length, extrarows)
    return xn, yn, Vn.T
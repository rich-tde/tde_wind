#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:12:06 2024

@author: konstantinos
"""

import numpy as np

def linearpad(D0,z0):
    factor = 100
    dz = z0[-1] - z0[-2]
    # print(np.shape(D0))
    dD = D0[:,-1] - D0[:,-2]
    
    z = [zi for zi in z0]
    z.append(z[-1] + factor*dz)
    z = np.array(z)
    #D = [di for di in D0]

    to_stack = np.add(D0[:,-1], factor*dD)
    to_stack = np.reshape(to_stack, (len(to_stack),1) )
    D = np.hstack((D0, to_stack))
    #D.append(to_stack)
    return np.array(D), z

def pad_interp(x,y,V):
    Vn, xn = linearpad(V, x)
    Vn, xn = linearpad(np.fliplr(Vn), np.flip(xn))
    Vn = Vn.T
    Vn, yn = linearpad(Vn, y)
    Vn, yn = linearpad(np.fliplr(Vn), np.flip(yn))
    Vn = Vn.T
    return xn, yn, Vn

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

def extrapolator_flipper(x ,y, V, slope_length = 5, extrarows = 25):
    xn, Vn = lin_extrapolator(x, V, slope_length, extrarows)
    yn, Vn = lin_extrapolator(y, Vn.T, slope_length, extrarows)
    return xn, yn, Vn.T
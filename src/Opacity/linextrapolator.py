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

def lin_extrapolator(y, V, slope_length, extrarows):
    # Low extrapolation
    deltay_low = y[1] - y[0]
    y_extra_low = [y[0] - deltay_low * (i + 1) for i in range(extrarows)]
    # High extrapolation
    deltay_high= y[-1] - y[-2]    
    y_extra_high = [y[-1] + deltay_high * (i + 1) for i in range(extrarows)]
    
    # Stack, reverse low to stack properly
    yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
    
    # 2D low
    yslope_low = y[slope_length - 1] - y[0]
    Vslope_low = (V[slope_length - 1, :] - V[0, :]) / yslope_low
    Vextra_low = [V[0, :] + Vslope_low * (y_extra_low[i] - y[0]) for i in range(extrarows)]

    # 2D high
    yslope_high = y[-1] - y[-slope_length]
    Vslope_high = (V[-1, :] - V[-slope_length, :]) / yslope_high
    Vextra_high = [V[-1, :] + Vslope_high * (y_extra_high[i] - y[-1]) for i in range(extrarows)]
    
    Vn = np.vstack([Vextra_low[::-1], V, Vextra_high]) 
    
    return yn, Vn

def extrapolator_flipper(x ,y, V, slope_length = 5, extrarowsx = 99, extrarowsy = 100):
    xn, Vn = lin_extrapolator(x, V, slope_length, extrarowsx) 
    yn, Vn = lin_extrapolator(y, Vn.T, slope_length, extrarowsy)
    return xn, yn, Vn.T

def double_extrapolator(x, y, K, slope_length = 5, extrarowsx= 99, extrarowsy= 100):
    # Extend x and y, adding data equally space (this suppose x,y as array equally spaced)
    # Low extrapolation
    deltaxn_low = x[1] - x[0]
    deltayn_low = y[1] - y[0] 
    x_extra_low = [x[0] - deltaxn_low * (i + 1) for i in range(extrarowsx)]
    y_extra_low = [y[0] - deltayn_low * (i + 1) for i in range(extrarowsy)]
    # High extrapolation
    deltaxn_high = x[-1] - x[-2]
    deltayn_high = y[-1] - y[-2]
    x_extra_high = [x[-1] + deltaxn_high * (i + 1) for i in range(extrarowsx)]
    y_extra_high = [y[-1] + deltayn_high * (i + 1) for i in range(extrarowsy)]
    
    # Stack, reverse low to stack properly
    xn = np.concatenate([x_extra_low[::-1], x, x_extra_high])
    yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
    
    # 2D low
    Kn = np.zeros((len(xn), len(yn)))
    for ix, xsel in enumerate(xn):
        for iy, ysel in enumerate(yn):
            if xsel < x[0]:
                deltax = x[slope_length - 1] - x[0]
                if ysel < y[0]:
                    deltay = y[slope_length - 1] - y[0]
                    Kxslope = (K[slope_length - 1, 0] - K[0, 0]) / deltax
                    Kyslope = (K[0, slope_length - 1] - K[0, 0]) / deltay
                    Kn[ix][iy] = K[0, 0] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[0])
                elif ysel > y[-1]:
                    deltay = y[-1] - y[-slope_length] 
                    Kxslope = (K[slope_length - 1, -1] - K[0, -1]) / deltax
                    Kyslope = (K[0, -1] - K[0, -slope_length]) / deltay
                    Kn[ix][iy] = K[0, -1] + Kxslope * (xsel - x[0]) + Kyslope * (ysel - y[-1])
                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = (K[slope_length - 1, iy_inK] - K[0, iy_inK]) / deltax
                    Kn[ix][iy] = K[0, iy_inK] + Kxslope * (xsel - x[0])
            elif xsel > x[-1]:
                deltax = x[-1] - x[-slope_length]
                if ysel < y[0]:
                    deltay = y[slope_length - 1] - y[0]
                    Kxslope = (K[-1, 0] - K[-slope_length, 0]) / deltax
                    Kyslope = (K[-1, slope_length - 1] - K[-1, 0]) / deltay
                    Kn[ix][iy] = K[-1, 0] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[0])
                elif ysel > y[-1]:
                    deltay = y[-1] - y[-slope_length] 
                    Kxslope = (K[-1, -1] - K[-slope_length, -1]) / deltax
                    Kyslope = (K[-1, -1] - K[-1, -slope_length]) / deltay
                    Kn[ix][iy] = K[-1, -1] + Kxslope * (xsel - x[-1]) + Kyslope * (ysel - y[-1])
                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kxslope = (K[-slope_length, iy_inK] - K[-1, iy_inK]) / deltax
                    Kn[ix][iy] = K[-1, iy_inK] + Kxslope * (xsel - x[-1])
            # x is correct, check y
            else:
                ix_inK = np.argmin(np.abs(x - xsel))
                if ysel < y[0]:
                    deltay = y[slope_length - 1] - y[0]
                    Kyslope = (K[ix_inK, slope_length - 1] - K[ix_inK, 0]) / deltay
                    Kn[ix][iy] = K[ix_inK, 0] + Kyslope * (ysel - y[0])
                elif ysel > y[-1]:
                    deltay = y[-1] - y[-slope_length]
                    Kyslope = (K[ix_inK, -1] - K[ix_inK, -slope_length]) / deltay
                    Kn[ix][iy] = K[ix_inK, -1] + Kyslope * (ysel - y[-1])
                else:
                    iy_inK = np.argmin(np.abs(y - ysel))
                    Kn[ix][iy] = K[ix_inK, iy_inK]
    
    return xn, yn, Kn
